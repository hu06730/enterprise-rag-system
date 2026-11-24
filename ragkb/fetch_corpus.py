"""一次性工具：抓取 Python 官方中文教程，转成纯文本语料放进 corpus/。

用法（在 C:\\Users\\hc\\RAG企业知识库 下）：
    .venv\\Scripts\\python -m project.fetch_corpus
"""

import re
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

BASE = "https://docs.python.org/zh-cn/3/tutorial/"
CHAPTERS = {
    "03_简介": "introduction.html",
    "04_控制流工具": "controlflow.html",
    "05_数据结构": "datastructures.html",
    "06_模块": "modules.html",
    "07_输入与输出": "inputoutput.html",
    "08_错误和异常": "errors.html",
    "09_类": "classes.html",
    "10_标准库概览": "stdlib.html",
    "11_标准库概览2": "stdlib2.html",
    "12_虚拟环境和包": "venv.html",
}

BLOCK = {"p", "li", "pre", "h1", "h2", "h3", "h4", "h5", "h6", "dt", "dd", "td", "th", "div", "br"}
SKIP = {"script", "style", "nav", "header", "footer", "noscript"}

DROP_LINES = {"上一主题", "下一主题", "快速搜索", "目录", "索引", "模块", "下一页", "上一页",
              "本页", "显示源代码", "搜索", "全局模块索引"}


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip_depth = 0
        self.parts: list[str] = []
        self.buf: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in SKIP:
            self.skip_depth += 1
        elif tag in BLOCK:
            self.flush()
        elif tag == "br":
            self.buf.append("\n")

    def handle_endtag(self, tag):
        if tag in SKIP:
            self.skip_depth = max(0, self.skip_depth - 1)
        elif tag in BLOCK:
            self.flush()

    def handle_data(self, data):
        if self.skip_depth:
            return
        self.buf.append(data)

    def flush(self):
        text = "".join(self.buf).strip()
        if text:
            self.parts.append(text)
        self.buf = []


def clean(parts: list[str]) -> str:
    text = "\n\n".join(parts)
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s in DROP_LINES:
            continue
        lines.append(s)
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def main() -> None:
    out_dir = Path(__file__).with_name("corpus")
    out_dir.mkdir(exist_ok=True)

    total = 0
    for name, page in CHAPTERS.items():
        with urllib.request.urlopen(BASE + page, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        parser = TextExtractor()
        parser.feed(html)
        parser.flush()
        text = clean(parser.parts)

        path = out_dir / f"python_{name}.txt"
        path.write_text(text, encoding="utf-8")
        total += len(text)
        print(f"{path.name:<28} {len(text):>7} 字")

    print(f"\n合计 {total} 字，写入 {out_dir}")


if __name__ == "__main__":
    main()
