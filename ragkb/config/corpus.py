"""语料清单：main.py 和 eval.py 共用这一份定义，避免两处各写一套。

语料放在 project/corpus/ 下（目前是 Python 官方中文教程 10 章）。
换语料 = 换这个目录里的文件，代码不用动。
"""

from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"


def corpus_files() -> list[str]:
    """返回 corpus/ 下所有 .txt 的绝对路径（已排序，保证每次入库顺序一致）。"""
    if not CORPUS_DIR.is_dir():
        raise FileNotFoundError(f"语料目录不存在: {CORPUS_DIR}")

    files = sorted(str(p) for p in CORPUS_DIR.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"语料目录里没有 .txt 文件: {CORPUS_DIR}")
    return files
