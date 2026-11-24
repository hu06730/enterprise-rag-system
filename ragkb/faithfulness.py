"""忠实度粗测：答案里的"硬事实"是不是真的来自检索到的文档。

两个检查（都是可机械验证的，不需要再调 LLM 当裁判）：
  1. check_citations  —— 答案里的 [n] 编号是否存在（n <= 文档数）
  2. entity_coverage  —— 答案里的数字 / 英文标识符 / 引号术语，有多少能在 context 里找到

用法（真实跑 10 题，会产生 LLM 调用）：
    python -m ragkb.faithfulness                      # 默认按题型轮转取 10 题
    python -m ragkb.faithfulness --limit 5            # 只跑 5 题
    python -m ragkb.faithfulness --ids 1,13,30        # 指定题号
    python -m ragkb.faithfulness --eval-set eval_set.json

注意：本模块的 main() 会连 Milvus + 调 LLM，所以不要放进 pytest；
     tests/test_faithfulness.py 只用假数据测那三个纯函数。
"""

import argparse
import json
import math
import re
from pathlib import Path

from langchain_core.documents import Document

# ---------------------------------------------------------------- 正则与词表

CITE = re.compile(r"[\[【](\d+)[\]】]")                 # [1] / 【2】
NUM = re.compile(r"\d+(?:\.\d+)+|\d{2,}")              # 3.15 / 1992（单个数字忽略）
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{2,}")      # timeit / __all__ / f.close
QUOTED = re.compile(r"[\"'“”‘’「」『』]([^\"'“”‘’「」『』]{2,20})[\"'“”‘’「」『』]")

# 太通用的英文词，当成"事实"会把覆盖率虚高
STOPWORDS = {
    "python", "list", "dict", "str", "int", "the", "a", "of", "for", "and",
    "print", "return", "true", "false", "none", "def", "class", "import",
    "try", "except", "finally", "with", "as", "in", "is", "not", "if", "else",
    # 通用环境/工具名：模型举例时会自然提到，不代表"从文档抄的"
    "bash", "shell", "cmd", "powershell", "windows", "linux", "mac", "macos",
    "unix", "pypi", "github", "pip",
    # 常见的通用变量名（出现在代码示例里，不代表事实）
    "text", "data", "value", "result", "key", "item", "args", "kwargs",
}


# ---------------------------------------------------------------- 纯函数

def extract_facts(answer: str) -> list[str]:
    """抽出**强事实**：数字 + 英文标识符（去重保序）。

    为什么不把引号短语算进来：模型用引号常常是"强调自己的措辞"，不是"引用原文术语"。
    实测里 id=6/12/13 的未命中全部来自引号短语，属于测量假阳性，会冤枉模型。
    """
    facts: list[str] = []
    facts += NUM.findall(answer)
    facts += [w for w in IDENT.findall(answer) if w.lower() not in STOPWORDS]
    return list(dict.fromkeys(facts))


def extract_quoted(answer: str) -> list[str]:
    """抽出**弱引用**（引号里的短语）。只做参考展示，不参与覆盖率计算。"""
    return list(dict.fromkeys(QUOTED.findall(answer)))


def check_citations(answer: str, docs: list[Document]) -> tuple[bool, list[int]]:
    """答案里每个 [n] 是否都在 1..len(docs) 范围内。返回 (是否全有效, 越界编号)。"""
    ids = [int(n) for n in CITE.findall(answer)]
    bad = [n for n in ids if n < 1 or n > len(docs)]
    return len(bad) == 0, bad


def entity_coverage(answer: str, docs: list[Document]) -> tuple[float, list[str]]:
    """强事实中有多少能在 context 中找到。返回 (覆盖率, 未命中的实体)。"""
    facts = extract_facts(answer)
    if not facts:
        return 1.0, []

    context = "\n".join(d.page_content for d in docs).lower()

    hit, miss = [], []
    for f in facts:
        (hit if f.lower() in context else miss).append(f)
    return round(len(hit) / len(facts), 3), miss


# ---------------------------------------------------------------- 选题与运行

def select_questions(path: Path, limit: int = 10, ids: list[int] | None = None) -> list[dict]:
    """从题库里挑可答题。

    ids 给了就按 ids 挑；否则按题型轮转（heading → paraphrase → multihop → ...），
    保证各题型都覆盖到，而不是全挑到同一种简单题。
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    pool = [q for q in data if q.get("answerable")]

    if ids:
        by_id = {q["id"]: q for q in pool}
        return [by_id[i] for i in ids if i in by_id]

    buckets: dict[str, list[dict]] = {}
    for q in pool:
        buckets.setdefault(q["type"], []).append(q)

    picked: list[dict] = []
    while len(picked) < limit and any(buckets.values()):
        for t in sorted(buckets):
            if buckets[t] and len(picked) < limit:
                picked.append(buckets[t].pop(0))
    return picked


def quoted_misses(answer: str, docs: list[Document]) -> list[str]:
    """引号短语里，有哪些在 context 中找不到（仅供参考，不参与计分）。"""
    quoted = extract_quoted(answer)
    if not quoted:
        return []
    context = "\n".join(d.page_content for d in docs).lower()
    return [q for q in quoted if q.lower() not in context]


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute() or p.exists():
        return p
    return Path(__file__).with_name(path_str)


def run(pipeline, questions: list[dict], top_k: int = 5) -> list[dict]:
    """对每道题：检索 → 拼 prompt → 生成 → 两项检查。"""
    from ragkb.interfaces.prompt_builder import build_prompt

    rows = []
    total = len(questions)
    for i, q in enumerate(questions, 1):
        docs = pipeline.retrieve(q["question"], top_k=top_k)
        prompt = build_prompt(q["question"], docs)
        answer = pipeline.generator.generate(prompt)

        ok, bad = check_citations(answer, docs)
        cov, miss = entity_coverage(answer, docs)
        qmiss = quoted_misses(answer, docs)

        rows.append({
            "id": q["id"], "type": q["type"], "question": q["question"],
            "n_docs": len(docs), "answer": answer,
            "citations_ok": ok, "bad_ids": bad,
            "coverage": cov, "misses": miss, "quoted_misses": qmiss,
            # 存下 context，才能用 dump 精确重算（改了抽取规则就不必再调 LLM）
            "doc_contents": [d.page_content for d in docs],
        })
        print(f"  [{i}/{total}] id={q['id']:<3} 引用{'OK' if ok else '✗'} 覆盖率={cov:.2f}  "
              f"未命中={miss if miss else '-'}")
    return rows


def print_report(rows: list[dict]) -> None:
    if not rows:
        print("没有可分析的结果")
        return

    print("\n=== 逐题结果（覆盖率只算强事实：数字 + 英文标识符）===")
    print(f"{'id':>3}  {'类型':<12} {'引用':<4} {'覆盖率':>6}  {'未命中强事实':<20} {'引号短语未命中（参考）'}")
    print("-" * 100)
    for r in rows:
        miss = ", ".join(r["misses"][:3]) if r["misses"] else "-"
        qmiss = ", ".join(r.get("quoted_misses", [])[:2]) if r.get("quoted_misses") else "-"
        print(f"{r['id']:>3}  {r['type']:<12} {'OK' if r['citations_ok'] else '✗':<4} "
              f"{r['coverage']:>6.2f}  {miss:<20} {qmiss}")

    n = len(rows)
    cite_ok = sum(1 for r in rows if r["citations_ok"])
    avg_cov = sum(r["coverage"] for r in rows) / n
    print("-" * 78)
    print(f"引用有效率 = {cite_ok}/{n} = {cite_ok / n:.3f}")
    print(f"平均覆盖率 = {avg_cov:.3f}")

    worst = sorted(rows, key=lambda r: r["coverage"])[:2]
    print("\n覆盖率最低的 2 题（优先人工复核）：")
    for r in worst:
        print(f"  id={r['id']} 覆盖率={r['coverage']:.2f} 未命中={r['misses']}")
        print(f"    问题: {r['question']}")
        print(f"    答案: {r['answer'][:120]}...")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default="eval_set_v2.json")
    ap.add_argument("--limit", type=int, default=10, help="跑多少题（按题型轮转挑选）")
    ap.add_argument("--ids", default="", help="指定题号，逗号分隔，如 1,13,30")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--dump", default="", help="把逐题结果（含答案）写到这个 json，便于事后重新计分")
    args = ap.parse_args()

    path = resolve(args.eval_set)
    if not path.exists():
        raise SystemExit(f"找不到题库: {path}")

    ids = [int(x) for x in args.ids.split(",") if x.strip()] if args.ids else None
    questions = select_questions(path, limit=args.limit, ids=ids)

    if ids:
        got = {q["id"] for q in questions}
        skipped = [i for i in ids if i not in got]
        if skipped:
            print(f"⚠️ 这些题号被跳过（通常是 answerable=false 的不可答题）：{skipped}")

    print(f"题库: {path}")
    print(f"选中 {len(questions)} 题: {[q['id'] for q in questions]}")
    print("提醒：每题会调用一次 LLM（约 1~2 分钟总计）\n")

    from ragkb.main import build_pipeline

    pipeline = build_pipeline()
    # 只重建 BM25 索引，不重新入库（向量库里的数据已经在）
    from ragkb.config.corpus import corpus_files
    chunks = []
    for f in corpus_files():
        chunks.extend(pipeline.splitter.split(pipeline.parser.load(f)))
    pipeline.bm25_retriever.build_index(chunks)

    rows = run(pipeline, questions, top_k=args.top_k)
    print_report(rows)

    if args.dump:
        Path(args.dump).write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n[dump] 已写入 {args.dump}（含答案，可据此重新计分，不必重跑 LLM）")


if __name__ == "__main__":
    main()
