import pytest
from app.core.interfaces import RetrievalContext


@pytest.mark.asyncio
async def test_vector_retriever_returns_chunks():
    from app.core.retriever import VectorRetriever
    from app.ingestion.embedder import OpenAIEmbedder

    embedder = OpenAIEmbedder()
    retriever = VectorRetriever(embedder=embedder)

    ctx = RetrievalContext(kb_id=1, top_k=5, access_levels=["public", "internal"])
    results = await retriever.retrieve("test query", ctx)
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_bm25_retriever_builds_index():
    from app.core.retriever import BM25Retriever

    retriever = BM25Retriever()
    ctx = RetrievalContext(kb_id=1, top_k=5, access_levels=["public", "internal"])
    results = await retriever.retrieve("test", ctx)
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_hybrid_retriever_merges_results():
    from app.core.retriever import HybridRetriever
    from app.ingestion.embedder import OpenAIEmbedder

    embedder = OpenAIEmbedder()
    retriever = HybridRetriever(embedder=embedder)

    ctx = RetrievalContext(kb_id=1, top_k=10, access_levels=["public", "internal"], retrieval_mode="hybrid")
    results = await retriever.retrieve("test query", ctx)
    assert isinstance(results, list)
    assert all(r.score >= 0 for r in results)
