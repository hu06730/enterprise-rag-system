"""忠实度检查的单元测试：只测纯函数，用假答案，不连 Milvus、不调 LLM。

（真实跑 10 题的入口在 ragkb/faithfulness.py，用 `python -m ragkb.faithfulness` 执行）
"""

from langchain_core.documents import Document

from ragkb.faithfulness import check_citations, entity_coverage, extract_facts


def test_extract_facts_takes_numbers_and_idents():
    facts = extract_facts("用 timeit 测速，Python 3.15 里 __all__ 要写对。")
    assert "timeit" in facts
    assert "3.15" in facts
    assert "__all__" in facts


def test_extract_facts_filters_stopwords():
    """python / list 这类通用词不能当"事实"，否则覆盖率虚高。"""
    facts = extract_facts("python 的 list 和 dict 都很好用")
    assert "python" not in [f.lower() for f in facts]
    assert "list" not in [f.lower() for f in facts]


def test_citations_all_valid():
    docs = [Document(page_content="a"), Document(page_content="b")]
    ok, bad = check_citations("根据文档[1]和[2]，可以得出。", docs)
    assert ok is True
    assert bad == []


def test_citations_out_of_range():
    docs = [Document(page_content="a")]
    ok, bad = check_citations("根据文档[3]可知。", docs)
    assert ok is False
    assert bad == [3]


def test_citations_chinese_brackets_and_multi_digit():
    docs = [Document(page_content=str(i)) for i in range(10)]
    ok, bad = check_citations("见【10】。", docs)     # 10 <= 10，有效
    assert ok is True and bad == []


def test_entity_coverage_flags_hallucinated_ident():
    """f.close 在 context 里有 → 命中；timeit 没有 → 未命中，覆盖率 0.5。"""
    docs = [Document(page_content="不论 try 语句是否触发异常，都会执行 finally 子句；也可以用 f.close() 关闭文件。")]
    cov, miss = entity_coverage("用 f.close() 关闭[1]，再用 timeit 测速。", docs)
    assert "timeit" in miss
    assert "f.close" not in miss
    assert cov == 0.5


def test_entity_coverage_all_found():
    docs = [Document(page_content="使用 timeit 模块可以测量 3.15 版本的运行耗时。")]
    cov, miss = entity_coverage("用 timeit 测 3.15 的耗时。", docs)
    assert cov == 1.0
    assert miss == []


def test_entity_coverage_without_facts():
    """答案里没有硬事实时，按满分处理并注明（避免除零）。"""
    docs = [Document(page_content="随便什么内容")]
    cov, miss = entity_coverage("好的，我明白了。", docs)
    assert cov == 1.0
    assert miss == []
