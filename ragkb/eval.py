"""eval.py —— 检索质量评估（Day 12 任务 4）

【目标】
把「感觉好像变准了」变成两个可复现的数字：
    1. Recall@k  —— 该找到的，找到了几成
    2. MRR       —— 找到的位置够不够靠前

【为什么必须先有它】
没有这两个数字，明天做查询改写时就无法回答「改完到底好了多少」。

【输入】eval_set.json：7 道可答题 + 3 道不可答题
【输出】一张逐题表 + 三行汇总

【用法】在仓库根目录执行：
    .venv\\Scripts\\python -m project.eval               # 基线：稠密 + BM25 + RRF + Rerank
    .venv\\Scripts\\python -m project.eval --no-rerank   # 对照组：去掉 Rerank
提示：每次运行会重新 ingest 一次（hash 主键 + upsert，幂等，不会重复），
      主要耗时在加载 bge-m3 和给 76 个 chunk 算向量，约 30~90 秒。

【判定口径 —— 写死在这里，改了口径数字就不可比】
    命中        = 任一 gold_snippet 是某个 top_k 文档 page_content 的子串
    Recall@k    = 命中的可答题数 / 可答题总数
    MRR         = 平均( 1 / 第一次命中的排名 )，未命中记 0
    不可答题    = 不参与 Recall/MRR，只单独看「检索层最高分」是否够低
                  （sigmoid 后 < 0.1 视为「库里确实没有」，可作为拒答的代理信号）
"""

import argparse
import json
import math
from pathlib import Path

from langchain_core.documents import Document

from ragkb.config.settings import ENCODING
from ragkb.config.corpus import corpus_files

TOP_K = 5
EVAL_SET_PATH = Path(__file__).with_name("eval_set.json")


# --------------------------------------------------------------------------
# 数据加载（已实现，不用改）
# --------------------------------------------------------------------------
def load_eval_set(path: Path = EVAL_SET_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_eval_path(name: str) -> Path:
    """解析题库路径，三种情况都支持：

    1. 绝对路径 → 直接用
    2. 相对路径，且相对「当前工作目录」能找到 → 用它
    3. 否则按「eval.py 所在目录」找（这样从任何目录启动都能跑）
    """
    p = Path(name)
    if p.is_absolute():
        return p
    if p.exists():
        return p
    return Path(__file__).with_name(name)


# --------------------------------------------------------------------------
# 命中判定
# --------------------------------------------------------------------------
def snippet_hit(docs: list[Document], gold_snippets: list[str]) -> bool:
    """任一 gold_snippet 出现在任一 doc 的 page_content 里 → True。

    提示：
      - 用 `in` 做子串匹配，不要用 `==`
      - gold_snippets 为空（不可答题）时，直接返回 False
      - 只要一个命中就算命中（口径已定死）

    验收（自己先手写两条 Document 试）：
        docs = [Document(page_content="毛泽东思想是马克思主义中国化的成果")]
        snippet_hit(docs, ["马克思主义中国化"]) is True
        snippet_hit(docs, ["西红柿"])            is False
    """
    if not gold_snippets or not docs:
        return False

    texts = [doc.page_content for doc in docs]
    return any(gold in text for gold in gold_snippets for text in texts)


# --------------------------------------------------------------------------
# 首命中排名
# --------------------------------------------------------------------------
def gold_coverage(docs: list[Document], gold_snippets: list[str]) -> tuple[int, int]:
    """严格召回：top_k 覆盖了几个 gold 片段 / 一共几个 gold 片段。

    与 first_hit_rank 的区别：
      first_hit_rank 只要"任意一个 gold 命中"就算命中（题级 HitRate）
      这里统计"命中了几个"，用于计算标准定义的 Recall@k（相关文档覆盖率）
    多跳题需要两个片段，只中一个时 HitRate 满分、Recall 只有 0.5。
    """
    if not gold_snippets:
        return 0, 0
    found = sum(1 for g in gold_snippets if any(g in d.page_content for d in docs))
    return found, len(gold_snippets)


def first_hit_rank(docs: list[Document], gold_snippets: list[str]) -> int:
    """返回第一个命中的文档排名（第 1 条 → 1）；一条都没命中 → 0。

    提示：
      - 排名是「位置 + 1」，否则 1/rank 会除零
      - 用 enumerate 从 1 开始数最省事

    验收：
        3 条 docs，gold 命中第 2 条 → 返回 2
        gold 谁都不命中        → 返回 0
    """
    if not gold_snippets:
        return 0

    for rank, doc in enumerate(docs, start=1):
        if snippet_hit([doc], gold_snippets):
            return rank

    return 0


# --------------------------------------------------------------------------
# 遍历题库
# --------------------------------------------------------------------------
def run_eval(pipeline, eval_set: list[dict], top_k: int = TOP_K, multi: bool = False) -> list[dict]:
    """对每道题调用检索函数（单查询或多查询），收集一条记录：

        {
          "id": 1,
          "type": "heading",          # 取 eval_set 里的 type，没有就写 "?"
          "question": "...",
          "answerable": True,
          "rank": 1,                  # first_hit_rank 的结果
          "hit": True,                # rank > 0
          "top_score": 4.3757,        # docs[0].metadata 里的 rerank_score（没有则为 None）
        }

    提示：
      - 不可答题也要跑（要拿到它的 top_score，用来看「检索层是否足够低」）
      - top_score 取值顺序：先找 rerank_score，再找 rrf_score，都没有就 None
      - 每题之间打印一行进度，方便你看到跑到第几题（模型跑得慢）
    """
    results: list[dict] = []
    total = len(eval_set)

    for index, item in enumerate(eval_set, start=1):
        try:
            if multi:
                docs = pipeline.retrieve_multi(item["question"], top_k=top_k)
            else:
                docs = pipeline.retrieve(item["question"], top_k=top_k)
        except Exception as exc:  # 单题检索失败不该让整份评估崩掉
            print(f"  [{index}/{total}] id={item['id']} 检索失败：{exc}")
            docs = []

        gold_snippets = item.get("gold_snippets") or []
        rank = first_hit_rank(docs, gold_snippets)
        gold_found, gold_total = gold_coverage(docs, gold_snippets)

        top_score = None
        score_kind = None
        if docs:
            meta = docs[0].metadata
            # 注意：不能用 `a or b`，rerank_score 可能是 0.0 / 负数，会被误判成"没有值"
            if "rerank_score" in meta:
                top_score, score_kind = meta["rerank_score"], "rerank"
            elif "rrf_score" in meta:
                top_score, score_kind = meta["rrf_score"], "rrf"

        print(f"  [{index}/{total}] id={item['id']:<3} rank={rank}  {item['question'][:32]}")

        results.append({
            "id": item["id"],
            "type": item.get("type", "?"),
            "question": item["question"],
            "answerable": item["answerable"],
            "rank": rank,
            "hit": rank > 0,
            "gold_found": gold_found,
            "gold_total": gold_total,
            "top_score": top_score,
            "score_kind": score_kind,
        })

    return results


# --------------------------------------------------------------------------
# 汇总指标
# --------------------------------------------------------------------------
def summarize(rows: list[dict]) -> dict:
    """返回：

        {
          "n_answerable": 7,
          "n_hit": 6,
          "recall": 0.857,          # n_hit / n_answerable，分母为 0 时记 0.0
          "mrr": 0.643,             # 平均(1/rank)，未命中该题记 0
          "unanswerable": [         # 不可答题的「检索层最高分」，供人工判断
              {"id": 8, "question": "...", "top_score": -4.74},
              ...
          ],
        }

    提示：
      - 只统计 answerable=True 的题算 Recall 和 MRR
      - 不可答题不参与这两个指标（它们没有 gold_snippets，永远"不命中"，
        混进去会把分数拉低，没有意义）
      - 保留三位小数：round(x, 3)
    """
    answerable = [row for row in rows if row["answerable"]]
    n_answerable = len(answerable)
    n_hit = sum(1 for row in answerable if row["hit"])

    recall = round(n_hit / n_answerable, 3) if n_answerable else 0.0

    # Recall@1：第 1 名就命中。比 Recall@5 更难，天花板没到 1.0 时它最能区分好坏
    # 严格 Recall@k：命中的 gold 片段总数 / gold 片段总数（可答题）
    gold_found = sum(row.get("gold_found", 0) for row in answerable)
    gold_total = sum(row.get("gold_total", 0) for row in answerable)
    recall_strict = round(gold_found / gold_total, 3) if gold_total else 0.0

    n_hit_at_1 = sum(1 for row in answerable if row["rank"] == 1)
    recall_at_1 = round(n_hit_at_1 / n_answerable, 3) if n_answerable else 0.0

    # 未命中的题记 0 分，但分母仍然是「可答题总数」——分母不能跟着命中数变
    mrr_total = sum(1.0 / row["rank"] for row in answerable if row["rank"] > 0)
    mrr = round(mrr_total / n_answerable, 3) if n_answerable else 0.0

    unanswerable = [
        {"id": row["id"], "question": row["question"],
         "top_score": row["top_score"], "score_kind": row.get("score_kind")}
        for row in rows
        if not row["answerable"]
    ]

    # 按题型分组：总分会被简单题拉高，分组才看得出哪类题真的难
    by_type: dict[str, dict] = {}
    for row in answerable:
        bucket = by_type.setdefault(row["type"], {"n": 0, "hit": 0, "rr": 0.0})
        bucket["n"] += 1
        if row["rank"] > 0:
            bucket["hit"] += 1
            bucket["rr"] += 1.0 / row["rank"]
    for bucket in by_type.values():
        bucket["recall"] = round(bucket["hit"] / bucket["n"], 3)
        bucket["mrr"] = round(bucket["rr"] / bucket["n"], 3)
        del bucket["rr"]

    return {
        "n_answerable": n_answerable,
        "n_hit": n_hit,
        "recall": recall,
        "recall_at_1": recall_at_1,
        "recall_strict": recall_strict,
        "gold_found": gold_found,
        "gold_total": gold_total,
        "mrr": mrr,
        "unanswerable": unanswerable,
        "by_type": by_type,
    }


# --------------------------------------------------------------------------
# 以下已实现，不用改
# --------------------------------------------------------------------------
def sigmoid(x: float | None) -> float | None:
    """logits → 0~1；bge-reranker 输出可正可负，负数很正常。"""
    if x is None:
        return None
    return 1.0 / (1.0 + math.exp(-x))


def print_report(rows: list[dict], summary: dict, top_k: int = TOP_K) -> None:
    print()
    print(f"{'id':>3}  {'类型':<12} {'命中':<4} {'排名':>4} {'最高分':>8}  {'题目'}")
    print("-" * 78)

    for r in rows:
        # 用 ASCII 记号：Windows 控制台默认 GBK，✓ / ✗ 会抛 UnicodeEncodeError
        mark = "HIT" if r["hit"] else ("-" if not r["answerable"] else "MISS")
        rank = r["rank"] if r["rank"] else "-"
        score = r["top_score"]
        score_text = f"{score:8.3f}" if isinstance(score, (int, float)) else "       -"
        print(f"{r['id']:>3}  {r['type']:<12} {mark:<4} {rank!s:>4} {score_text}  {r['question']}")

    print("-" * 78)
    print(f"可答题 {summary['n_answerable']} 题：命中 {summary['n_hit']} 题")
    print(f"HitRate@{top_k} = {summary['recall']:.3f}（题级：至少命中 1 个 gold 的题占比）")
    print(f"Recall@{top_k}  = {summary['recall_strict']:.3f}"
          f"（严格：gold 片段覆盖 {summary['gold_found']}/{summary['gold_total']}）")
    print(f"Recall@1 = {summary['recall_at_1']:.3f}    MRR = {summary['mrr']:.3f}")
    print()
    print("按题型分组：")
    for t, b in summary["by_type"].items():
        print(f"  {t:<18} {b['hit']}/{b['n']} 题   Recall={b['recall']:.3f}   MRR={b['mrr']:.3f}")
    print()
    print("不可答题（看检索层最高分是否够低）：")
    for u in summary["unanswerable"]:
        if u.get("score_kind") == "rerank":
            s = sigmoid(u["top_score"])
            s_text = f"{s:.3f}" if s is not None else "-"
            print(f"  id={u['id']:<3} sigmoid(rerank)={s_text:<7} {u['question']}")
        else:
            # rrf_score 量级是 0.01~0.03，跟 rerank 的 logit 不是一回事，不能一起 sigmoid
            raw = f"{u['top_score']:.4f}" if isinstance(u["top_score"], (int, float)) else "-"
            print(f"  id={u['id']:<3} rrf={raw:<9}（未开重排，此分数不可与 rerank 比较） {u['question']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", help="把逐题结果写到这个 json 文件")
    parser.add_argument("--no-rerank", action="store_true", help="对照组：关掉重排")
    parser.add_argument("--no-distractors", action="store_true", help="已废弃（distractors 目录已移除），保留仅为兼容旧命令")
    parser.add_argument("--rebuild", action="store_true", help="入库前清空集合（切语料时必须用，否则测到的是两次的并集）")
    parser.add_argument("--skip-ingest", action="store_true", help="跳过向量入库，只重建 BM25 索引（换了语料就别用）")
    parser.add_argument("--multi-query", action="store_true", help="使用多查询检索")
    parser.add_argument("--eval-set", default=str(EVAL_SET_PATH), help="题库文件（默认 eval_set.json）")
    parser.add_argument("--top-k", type=int, default=TOP_K, help=f"检索返回条数（默认 {TOP_K}）")
    args = parser.parse_args()

    from ragkb.main import build_pipeline

    pipeline = build_pipeline()
    if args.no_rerank:
        pipeline.reranker = None
        print("[模式] 已关闭重排（对照组）")
    else:
        print("[模式] 稠密 + BM25 + RRF + Rerank（基线）")

    if args.rebuild:
        pipeline.vector_store.drop_collection()
        pipeline.vector_store.create_collection()
        print("[重建] 已清空集合并重建")

    files = corpus_files()
    print(f"[语料] {len(files)} 个文件：{[Path(f).name for f in files]}")

    if args.skip_ingest:
        # 集合里已经是这批语料时用这个：只重建内存里的 BM25 索引，秒级完成
        chunks = []
        for f in files:
            chunks.extend(pipeline.splitter.split(pipeline.parser.load(f, ENCODING)))
        pipeline.bm25_retriever.build_index(chunks)
        print(f"[入库] 已跳过（复用集合数据），BM25 索引重建 {len(chunks)} 条")
    else:
        count = pipeline.ingest(files, ENCODING)
        print(f"[入库] {count} 条（幂等，不重复）")

    print(f"[检索模式] {'多查询（Multi-Query）' if args.multi_query else '单查询'}")
    eval_path = resolve_eval_path(args.eval_set)
    print(f"[题库] {eval_path}")

    eval_set = load_eval_set(eval_path)
    print(f"[检索条数] top_k = {args.top_k}")
    rows = run_eval(pipeline, eval_set, args.top_k, multi=args.multi_query)
    summary = summarize(rows)
    print_report(rows, summary, top_k=args.top_k)
    
    if args.dump:
        import json as _json
        Path(args.dump).write_text(_json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[dump] 已写入 {args.dump}")


if __name__ == "__main__":
    main()
