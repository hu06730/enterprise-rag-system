from langchain_core.documents import Document

from ragkb.pipeline.rag_pipeline import RAGPipeline
from ragkb.tests.fakes import RecordingGenerator, FakeParser, FakeSplitter, FakeVectorStore, FakeEmbedding, \
    FakeBM25Retriever


class LowScoreReranker:
    """永远给 -5 分（sigmoid ≈ 0.007，必被拒）"""
    def rerank(self, query, docs, top_k=5):
        return [Document(page_content=d.page_content,
                         metadata={**d.metadata, "rerank_score": -5.0})
                for d in docs[:top_k]]

def test_refuse_when_score_below_threshold():
    gen = RecordingGenerator()
    p = RAGPipeline(parser=FakeParser(), splitter=FakeSplitter(),
                    generator=gen, vector_store=FakeVectorStore(),
                    embedding_model=FakeEmbedding(),
                    bm25_retriever=FakeBM25Retriever(),
                    reranker=LowScoreReranker(),
                    refuse_threshold=0.2)

    answer = p.query("测试问题")

    assert answer == "根据现有资料无法回答。"     # 返回的是固定拒答语
    assert gen.call_count == 0                     # 关键：LLM 一次都没调


class HighScoreReranker:
    """永远给 +5 分（sigmoid ≈ 0.993，远高于阈值）"""
    def rerank(self, query, docs, top_k=5):
        return [Document(page_content=d.page_content,
                         metadata={**d.metadata, "rerank_score": 5.0})
                for d in docs[:top_k]]


def test_answer_when_score_above_threshold():
    """反向用例：阈值不能把正常问题也拒掉。

    只测「该拒的时候拒了」是不够的——如果实现里写错成 `score > threshold` 就拒，
    上面那条测试照样能过。必须同时测「不该拒的时候没拒」。
    """
    gen = RecordingGenerator()
    p = RAGPipeline(parser=FakeParser(), splitter=FakeSplitter(),
                    generator=gen, vector_store=FakeVectorStore(),
                    embedding_model=FakeEmbedding(),
                    bm25_retriever=FakeBM25Retriever(),
                    reranker=HighScoreReranker(),
                    refuse_threshold=0.2)

    answer = p.query("测试问题")

    assert answer == "假答案"                      # 走的是生成器的返回
    assert gen.call_count == 1                     # LLM 被调用了一次
    assert gen.last_prompt is not None