from ragkb.tests.fakes import FakeParser, FakeSplitter, FakeVectorStore, FakeGenerator, FakeEmbedding, FakeBM25Retriever
from ragkb.pipeline.rag_pipeline import RAGPipeline
from ragkb.agent_tools import make_rag_tool


def test_rag_tool_returns_string():
    pipeline = RAGPipeline(
        parser=FakeParser(),
        splitter=FakeSplitter(),
        vector_store=FakeVectorStore(),
        generator=FakeGenerator(),
        embedding_model=FakeEmbedding(),
        bm25_retriever=FakeBM25Retriever()
    )
    tool = make_rag_tool(pipeline)

    assert tool.name == "search_knowledge_base"
    result = tool.invoke({"query": "测试问题"})
    assert result == "假答案"