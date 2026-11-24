"""校验 eval_set.json：每条 gold_snippet 必须真的存在于「切分后的 chunk」里。

语料来源与 eval.py 保持一致（corpus/ 优先），避免"题库对着旧语料校验"的错位。

用法：.venv\\Scripts\\python -m project.validate_eval_set
"""

import argparse
import json
from pathlib import Path

from ragkb.config.settings import CHUNK_SIZE, CHUNK_OVERLAP, ENCODING
from ragkb.eval import corpus_files
from ragkb.parsers.text_parser import TextParser
from ragkb.splitters.text_splitter import TextSplitter

ROOT = Path(__file__).parent

ap = argparse.ArgumentParser()
ap.add_argument("--eval-set", default="eval_set.json", help="题库文件（默认 eval_set.json）")
args = ap.parse_args()
EVAL_PATH = Path(args.eval_set)
if not EVAL_PATH.is_absolute():
    EVAL_PATH = ROOT / EVAL_PATH

report = []

parser = TextParser()
splitter = TextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

files = corpus_files()
chunks = []
raws = []
for f in files:
    raws.append(Path(f).read_text(encoding=ENCODING))
    chunks.extend(splitter.split(parser.load(f, ENCODING)))

texts = [c.page_content for c in chunks]
raw = "\n".join(raws)

report.append(f"语料文件数 = {len(files)}，chunk 数 = {len(chunks)}（chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}）")
report.append("")

data = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
report.append(f"题目数 = {len(data)}；可答 = {sum(1 for d in data if d['answerable'])}；不可答 = {sum(1 for d in data if not d['answerable'])}")
report.append("")

ok, bad = 0, 0
for item in data:
    q = item["question"]
    snips = item["gold_snippets"]
    if not item["answerable"]:
        report.append(f"[{item['id']:>2}] 不可答  {q}")
        if snips:
            report.append("       -> gold_snippets 应为空！")
            bad += 1
        else:
            ok += 1
        continue

    lines = [f"[{item['id']:>2}] 可答    {q}"]
    for s in snips:
        hits = [i for i, t in enumerate(texts) if s in t]
        in_raw = s in raw
        flag = "OK  " if hits else "MISS"
        lines.append(f"       {flag} 长度{len(s):>3}  命中chunk={len(hits)}个  在原文中={'是' if in_raw else '否'}")
        lines.append(f"            「{s}」")
        ok, bad = (ok + 1, bad) if hits else (ok, bad + 1)
    report.append("\n".join(lines))

report.append("")
report.append(f"结论：片段校验 通过 {ok} / 失败 {bad}")

text = "\n".join(report)
(ROOT / "eval_check_report.txt").write_text(text, encoding="utf-8")
print(text)
