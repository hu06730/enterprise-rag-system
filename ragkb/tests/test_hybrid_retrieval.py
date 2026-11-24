from langchain_core.documents import Document
from ragkb.retrievers.bm25_retriever import BM25Retriever
from ragkb.retrievers.rrf_fusion import rrf_fuse


def test_bm25_retrieval():
    docs = [
        Document(page_content="我们公司的产品编号是 A-102，用于数据采集"),
        Document(page_content="星辰计划的 Q3 目标是提高转化率"),
        Document(page_content="这是一篇无关的日常报告")
    ]
    retriever = BM25Retriever()
    retriever.build_index(docs)

    # 查询 A-102
    results = retriever.search("A-102", top_k=2)
    assert len(results) > 0
    assert "A-102" in results[0].page_content


def test_rrf_fusion():
    dense = [Document(page_content="A"), Document(page_content="B"), Document(page_content="C")]
    sparse = [Document(page_content="C"), Document(page_content="A"), Document(page_content="D")]

    fused = rrf_fuse(dense, sparse, top_k=3)
    # A 在 dense 排第 1，sparse 排第 2；C 在 sparse 排第 1，dense 排第 3。
    # 具体谁第一取决于计算，但 A 和 C 一定在前两名，B 和 D 会被挤下去
    assert fused[0].page_content in ["A", "C"]
    assert fused[1].page_content in ["A", "C"]