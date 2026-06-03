from app.core.interfaces import BaseRetriever, Chunk, RetrievalContext
from app.db.vec_store import search_similar
from app.db.sqlite import get_connection
from app.core.bm25_cache import bm25_cache


class VectorRetriever(BaseRetriever):
    def __init__(self, embedder):
        self.embedder = embedder

    async def retrieve(self, query: str, ctx: RetrievalContext) -> list[Chunk]:
        embedding = await self.embedder.embed_query(query)
        rows = search_similar(
            query_embedding=embedding,
            kb_id=ctx.kb_id,
            access_levels=ctx.access_levels,
            departments=ctx.user_departments or None,
            limit=ctx.top_k * 2,
            min_score=ctx.min_score,
        )
        return [
            Chunk(
                id=r["id"], doc_id=r["doc_id"], kb_id=r["kb_id"],
                chunk_index=r["chunk_index"], title=r["title"],
                text=r["text"], department=r["department"],
                access_level=r["access_level"],
                score=float(r.get("distance", 0) or 0),
                source="vector",
            )
            for r in rows
        ]


class BM25Retriever(BaseRetriever):
    async def retrieve(self, query: str, ctx: RetrievalContext) -> list[Chunk]:
        import jieba

        # 从缓存获取索引（首次自动从 DB 构建并持久化，后续直接复用）
        index = bm25_cache.get(ctx.kb_id)
        if not index:
            return []

        # 只需要对查询分词一次（不再对全部 chunk 分词）
        query_tokens = list(jieba.cut(query))
        scores = index.bm25.get_scores(query_tokens)

        # 按权限过滤：从缓存的 chunk 列表中筛选符合权限的结果
        results = []
        for i, chunk in enumerate(index.chunks):
            if chunk["access_level"] not in ctx.access_levels:
                continue
            if ctx.user_departments and chunk["department"] and chunk["department"] not in ctx.user_departments:
                continue
            results.append(Chunk(
                id=chunk["id"], doc_id=chunk["doc_id"], kb_id=chunk["kb_id"],
                chunk_index=chunk["chunk_index"], title=chunk["title"],
                text=chunk["text"], department=chunk["department"],
                access_level=chunk["access_level"],
                score=float(scores[i]),
                source="keyword",
            ))
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:ctx.top_k * 2]


class HybridRetriever(BaseRetriever):
    def __init__(self, embedder):
        self.vector = VectorRetriever(embedder)
        self.bm25 = BM25Retriever()
        self.k = 60

    async def retrieve(self, query: str, ctx: RetrievalContext) -> list[Chunk]:
        if ctx.retrieval_mode == "vector":
            return await self.vector.retrieve(query, ctx)[:ctx.top_k]
        if ctx.retrieval_mode == "keyword":
            return await self.bm25.retrieve(query, ctx)[:ctx.top_k]

        vec_results = await self.vector.retrieve(query, ctx)
        bm25_results = await self.bm25.retrieve(query, ctx)

        rrf_scores: dict[int, float] = {}
        chunk_map: dict[int, Chunk] = {}

        for rank, chunk in enumerate(vec_results):
            rrf_scores[chunk.id] = ctx.vector_weight / (self.k + rank + 1)
            chunk_map[chunk.id] = chunk

        for rank, chunk in enumerate(bm25_results):
            score = ctx.bm25_weight / (self.k + rank + 1)
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0) + score
            if chunk.id not in chunk_map:
                chunk_map[chunk.id] = chunk

        merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for chunk_id, rrf_score in merged[:ctx.top_k]:
            c = chunk_map[chunk_id]
            c.score = rrf_score
            results.append(c)
        return results
