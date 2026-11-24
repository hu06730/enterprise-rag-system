from ragkb.pipeline.rag_pipeline import RAGPipeline
from ragkb.tests.fakes import FakeParser, FakeSplitter, FakeVectorStore, FakeGenerator, FakeEmbedding, FakeBM25Retriever, \
    FakeReranker
import inspect


def test_ingest_returns_count():
    pipeline = RAGPipeline(parser=FakeParser(), splitter=FakeSplitter(), vector_store=FakeVectorStore(),
                               generator=FakeGenerator(), embedding_model=FakeEmbedding(),
                           bm25_retriever=FakeBM25Retriever(),reranker=FakeReranker())

    count=pipeline.ingest("fake.md")
    assert count>0

def test_query_returns_answer():
    pipeline = RAGPipeline(parser=FakeParser(), splitter=FakeSplitter(), vector_store=FakeVectorStore(),
                           generator=FakeGenerator(), embedding_model=FakeEmbedding(),
                           bm25_retriever=FakeBM25Retriever(),reranker=FakeReranker())

    result = pipeline.query("测试问题")
    assert result =="假答案"

def test_query_empty_docs_refuses_without_llm():
    class EmptyVectorStore(FakeVectorStore):
        def search(self, query_vector, top_k):
            return []

    class EmptyBM25Retriever(FakeBM25Retriever):
        def search(self, query, top_k=5):
            return []

    class RecordingGenerator(FakeGenerator):
        def __init__(self):
            self.last_prompt = None

        def generate(self, prompt):
            self.last_prompt = prompt
            return "假答案"

    gen = RecordingGenerator()
    p = RAGPipeline(
        parser=FakeParser(),
        splitter=FakeSplitter(),
        vector_store=EmptyVectorStore(),
        generator=gen,
        embedding_model=FakeEmbedding(),
        bm25_retriever=EmptyBM25Retriever(),
        reranker=FakeReranker(),
    )

    answer = p.query("测试问题")

    # 契约变更（Day 17）：检索为空时直接在检索层拒答，不再拼 prompt 去调 LLM。
    # 目的：省一次调用，也避免模型对着空上下文自由发挥。
    assert answer == "根据现有资料无法回答。"
    assert gen.last_prompt is None      # 关键：LLM 一次都没被调用

def test_pipeline_does_not_instantiate():
    source = inspect.getsource(RAGPipeline)

    assert "MilvusStore(" not in source
    assert "HuggingFaceEmbeddings(" not in source
    assert "ChatOpenAI(" not in source




