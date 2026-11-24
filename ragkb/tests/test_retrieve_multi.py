from ragkb.pipeline.rag_pipeline import RAGPipeline
from ragkb.tests.fakes import FakeRewriter, FakeParser, FakeSplitter, FakeEmbedding, FakeVectorStore, FakeGenerator, \
    FakeBM25Retriever


def test_multi_query_disabled_by_default():
    # query_rewriter 不传 → 行为和 retrieve 完全一致
    p = RAGPipeline(FakeParser(),FakeSplitter(),FakeGenerator,FakeVectorStore(),FakeEmbedding(),
                    FakeBM25Retriever(),query_rewriter=None)
    assert [d.page_content for d in p.retrieve_multi("q", 3)] == \
           [d.page_content for d in p.retrieve("q", 3)]

def test_multi_query_uses_rewriter():
    rw = FakeRewriter()
    p = RAGPipeline(FakeParser(),FakeSplitter(),FakeGenerator,FakeVectorStore(),FakeEmbedding(),
                    FakeBM25Retriever(),query_rewriter=rw)
    docs = p.retrieve_multi("q", top_k=3)
    assert rw.called is True
    assert len(docs) <= 3
    assert len({d.page_content for d in docs}) == len(docs)   # 无重复

def test_multi_query_empty_rewrite_returns_empty():
    class EmptyRewriter:
        def rewrite(self, query, n=3):
            return []
    p = RAGPipeline(FakeParser(),FakeSplitter(),FakeGenerator,FakeVectorStore(),FakeEmbedding(),
                    FakeBM25Retriever(), query_rewriter=EmptyRewriter())
    assert p.retrieve_multi("q") == []