"""
BM25 索引持久化缓存。

解决的问题：BM25Retriever 每次查询都从 SQLite 加载全部 chunk → 分词 → 建索引，
耗时随数据量线性增长。本模块将索引缓存到内存 + 磁盘，查询时直接复用。

生命周期：
  - 启动时 warm_up() 预热所有 KB 的索引
  - 文档入库 / 删除 / 重建后调用 invalidate(kb_id) 重建该 KB 的索引
  - 知识库删除时调用 invalidate(kb_id) 清除索引
"""

import os
import pickle
import logging
from dataclasses import dataclass, field

import jieba
from rank_bm25 import BM25Okapi

from app.db.sqlite import get_connection
from app.config import settings

logger = logging.getLogger(__name__)

# 磁盘缓存目录
_CACHE_DIR = os.path.join(os.path.dirname(settings.DATABASE_URL.replace("sqlite:///", "")), "bm25_cache")


@dataclass
class KBIndex:
    """单个知识库的 BM25 索引快照。"""
    bm25: BM25Okapi
    chunks: list[dict]          # chunks_meta 的原始行
    query_tokens_cache: dict = field(default_factory=dict)  # 可选：缓存常见查询的分词结果


class BM25IndexCache:
    """
    BM25 索引缓存管理器。

    用法：
        cache = BM25IndexCache()

        # 查询时（毫秒级）
        index = cache.get(kb_id)
        if index:
            scores = index.bm25.get_scores(query_tokens)

        # 数据变更时（调一次就行）
        cache.invalidate(kb_id)   # 删除索引，下次查询自动重建
        cache.rebuild(kb_id)      # 立即重建
    """

    def __init__(self):
        self._memory_cache: dict[int, KBIndex] = {}   # kb_id → KBIndex

    # ── 查询 ──────────────────────────────────────────────

    def get(self, kb_id: int) -> KBIndex | None:
        """
        获取指定知识库的 BM25 索引。
        优先从内存取，内存没有则从磁盘加载，磁盘也没有则构建。
        """
        # 1. 内存命中
        if kb_id in self._memory_cache:
            return self._memory_cache[kb_id]

        # 2. 尝试从磁盘加载
        index = self._load_from_disk(kb_id)
        if index is not None:
            self._memory_cache[kb_id] = index
            logger.info("BM25 索引从磁盘加载: kb_id=%d, chunks=%d", kb_id, len(index.chunks))
            return index

        # 3. 磁盘也没有，从数据库构建
        index = self._build_from_db(kb_id)
        if index is not None:
            self._memory_cache[kb_id] = index
            self._save_to_disk(kb_id, index)
            logger.info("BM25 索引新构建: kb_id=%d, chunks=%d", kb_id, len(index.chunks))

        return index

    def get_or_none(self, kb_id: int) -> KBIndex | None:
        """只查缓存，不主动构建。用于判断索引是否已就绪。"""
        if kb_id in self._memory_cache:
            return self._memory_cache[kb_id]
        index = self._load_from_disk(kb_id)
        if index is not None:
            self._memory_cache[kb_id] = index
        return index

    # ── 失效 / 重建 ───────────────────────────────────────

    def invalidate(self, kb_id: int):
        """删除指定 KB 的索引（内存 + 磁盘），下次查询时自动重建。"""
        self._memory_cache.pop(kb_id, None)
        cache_path = self._cache_path(kb_id)
        if os.path.exists(cache_path):
            os.remove(cache_path)
            logger.info("BM25 索引已清除: kb_id=%d", kb_id)

    def rebuild(self, kb_id: int):
        """立即重建指定 KB 的索引（内存 + 磁盘）。"""
        self.invalidate(kb_id)
        index = self._build_from_db(kb_id)
        if index is not None:
            self._memory_cache[kb_id] = index
            self._save_to_disk(kb_id, index)
            logger.info("BM25 索引已重建: kb_id=%d, chunks=%d", kb_id, len(index.chunks))

    def warm_up(self):
        """启动时预热：加载或构建所有 KB 的索引。"""
        conn = get_connection()
        rows = conn.execute("SELECT id FROM knowledge_bases").fetchall()
        conn.close()

        for row in rows:
            kb_id = row["id"]
            try:
                self.get(kb_id)
            except Exception as e:
                logger.warning("BM25 索引预热失败: kb_id=%d, error=%s", kb_id, e)

        logger.info("BM25 索引预热完成: 共 %d 个知识库", len(self._memory_cache))

    # ── 内部方法 ──────────────────────────────────────────

    def _build_from_db(self, kb_id: int) -> KBIndex | None:
        """从 SQLite 加载 chunk，分词，构建 BM25 索引。"""
        conn = get_connection()
        rows = conn.execute(
            "SELECT id, doc_id, kb_id, chunk_index, title, text, department, access_level, tags "
            "FROM chunks_meta WHERE kb_id=?",
            (kb_id,),
        ).fetchall()
        conn.close()

        if not rows:
            return None

        chunks = [dict(r) for r in rows]
        texts = [c["text"] for c in chunks]
        tokenized = [list(jieba.cut(t)) for t in texts]
        bm25 = BM25Okapi(tokenized)

        return KBIndex(bm25=bm25, chunks=chunks)

    def _cache_path(self, kb_id: int) -> str:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        return os.path.join(_CACHE_DIR, f"bm25_kb_{kb_id}.pkl")

    def _save_to_disk(self, kb_id: int, index: KBIndex):
        try:
            with open(self._cache_path(kb_id), "wb") as f:
                pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.warning("BM25 索引写入磁盘失败: kb_id=%d, error=%s", kb_id, e)

    def _load_from_disk(self, kb_id: int) -> KBIndex | None:
        path = self._cache_path(kb_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                index = pickle.load(f)
            # 校验：如果磁盘索引的 chunk 数量和数据库不一致，废弃磁盘版本
            conn = get_connection()
            db_count = conn.execute(
                "SELECT COUNT(*) as cnt FROM chunks_meta WHERE kb_id=?", (kb_id,)
            ).fetchone()["cnt"]
            conn.close()
            if db_count != len(index.chunks):
                logger.info(
                    "BM25 磁盘索引过期 (磁盘=%d, 数据库=%d), kb_id=%d, 将重建",
                    len(index.chunks), db_count, kb_id,
                )
                return None
            return index
        except Exception as e:
            logger.warning("BM25 磁盘索引读取失败: kb_id=%d, error=%s", kb_id, e)
            return None


bm25_cache = BM25IndexCache()
