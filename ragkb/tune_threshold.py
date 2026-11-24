"""扫描拒答阈值 τ：sigmoid(rerank 最高分) < τ → 直接拒答，不调 LLM。

原理：重排模型给出的 logit 经 sigmoid 后是 0~1 的"这题能不能答"信号。
     分数越低，说明检索到的内容越答不了这个问题，可以省掉一次 LLM 调用。

用法（在仓库根目录）：
    python -m ragkb.tune_threshold --dump eval_rows.json
    python -m ragkb.tune_threshold --dump eval_rows.json --taus 0.1 0.2 0.3

输入：eval.py --dump 生成的逐题结果（含 answerable / top_score / score_kind）
输出：逐题分数 + 阈值扫描表 + 最优 τ + 边界题（离阈值最近的题）
"""

import argparse
import json
import math
from pathlib import Path

DEFAULT_TAUS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50]
BOUNDARY_MARGIN = 0.15   # 距离 τ 小于这个值就算"边界题"


def sigmoid(x):
    if x is None:
        return None
    # 防止 exp 溢出
    if x < -60:
        return 0.0
    if x > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def resolve(path_str: str) -> Path:
    """相对路径先按当前目录找，找不到再按本文件所在目录找。"""
    p = Path(path_str)
    if p.is_absolute() or p.exists():
        return p
    return Path(__file__).with_name(path_str)


def load_rows(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    # 只保留有 rerank 分数的题：rrf 分数（约 0.02）和 logit 不是一个量纲，混用会算错
    return [r for r in rows if r.get("score_kind") == "rerank" and r.get("top_score") is not None]


def print_scores(rows: list[dict]) -> None:
    print("=== 逐题分数（rerank 通道，sigmoid 后）===")
    for r in sorted(rows, key=lambda x: sigmoid(x["top_score"])):
        tag = "可答  " if r["answerable"] else "不可答"
        print(f'  id={r["id"]:>2}  {r["type"]:<18} {tag}  sigmoid={sigmoid(r["top_score"]):.3f}  {r["question"][:36]}')


def sweep(rows: list[dict], taus: list[float]) -> list[dict]:
    """对每个 τ 统计：误杀(可答被拒) / 拦截(不可答被拒) / 漏放 / 准确率。"""
    out = []
    n = len(rows)
    for tau in taus:
        fn = sum(1 for r in rows if r["answerable"] and sigmoid(r["top_score"]) < tau)
        tn = sum(1 for r in rows if not r["answerable"] and sigmoid(r["top_score"]) < tau)
        fp = sum(1 for r in rows if not r["answerable"] and sigmoid(r["top_score"]) >= tau)
        tp = sum(1 for r in rows if r["answerable"] and sigmoid(r["top_score"]) >= tau)
        out.append({
            "tau": tau,
            "误杀": fn,
            "拦截": tn,
            "漏放": fp,
            "放行可答": tp,
            "准确率": round((tp + tn) / n, 3) if n else 0.0,
            "拒答精确率": round(tn / (tn + fn), 3) if (tn + fn) else None,
            "拒答召回率": round(tn / (tn + fp), 3) if (tn + fp) else None,
        })
    return out


def print_sweep(results: list[dict]) -> None:
    print("\n=== 阈值扫描 ===")
    print(f"{'τ':>6} {'误杀':>4} {'拦截':>4} {'漏放':>4} {'准确率':>7} {'拒答精确率':>10} {'拒答召回率':>10}")
    for r in results:
        prec = "-" if r["拒答精确率"] is None else f"{r['拒答精确率']:.3f}"
        rec = "-" if r["拒答召回率"] is None else f"{r['拒答召回率']:.3f}"
        print(f"{r['tau']:>6} {r['误杀']:>4} {r['拦截']:>4} {r['漏放']:>4} "
              f"{r['准确率']:>7.3f} {prec:>10} {rec:>10}")


def pick_best(results: list[dict]) -> dict:
    """优先零误杀，其次准确率高。"""
    return sorted(results, key=lambda r: (r["误杀"], -r["准确率"]))[0]


def print_boundary(rows: list[dict], tau: float) -> None:
    near = [r for r in rows if r["answerable"] and abs(sigmoid(r["top_score"]) - tau) < BOUNDARY_MARGIN]
    print(f"\n=== 边界题（可答题中，分数距 τ={tau} 小于 {BOUNDARY_MARGIN} 的）===")
    if not near:
        print("  无")
        return
    for r in near:
        s = sigmoid(r["top_score"])
        side = "高于阈值（会被放行）" if s >= tau else "低于阈值（会被误杀！）"
        print(f'  id={r["id"]:>2}  sigmoid={s:.3f}  {side}  {r["question"][:36]}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="eval_rows.json", help="eval.py --dump 生成的 json")
    ap.add_argument("--taus", nargs="*", type=float, default=DEFAULT_TAUS, help="要扫描的阈值列表")
    args = ap.parse_args()

    path = resolve(args.dump)
    if not path.exists():
        raise SystemExit(f"找不到 {path}，请先跑：python -m ragkb.eval --skip-ingest --dump eval_rows.json")

    rows = load_rows(path)
    n_ans = sum(1 for r in rows if r["answerable"])
    n_un = len(rows) - n_ans
    print(f"数据来源: {path}")
    print(f"题目数: {len(rows)}（可答 {n_ans} / 不可答 {n_un}）\n")

    print_scores(rows)
    results = sweep(rows, args.taus)
    print_sweep(results)

    best = pick_best(results)
    print(f"\n>>> 最优 τ = {best['tau']}：误杀 {best['误杀']} 题、拦截 {best['拦截']} 题、"
          f"漏放 {best['漏放']} 题、准确率 {best['准确率']}")
    print_boundary(rows, best["tau"])

    print("\n提醒：τ 是在这批题上挑出来的，属于调参；")
    print("     边界题离阈值越近，换一批题后越可能翻车，需要在更大题库上复验。")


if __name__ == "__main__":
    main()
