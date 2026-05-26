import pytest


@pytest.mark.asyncio
async def test_openai_embedder_returns_correct_dimensions():
    from app.ingestion.embedder import OpenAIEmbedder
    embedder = OpenAIEmbedder(model="text-embedding-3-small")
    result = await embedder.embed(["hello world"])
    assert len(result) == 1
    assert len(result[0]) == 1536


@pytest.mark.asyncio
async def test_openai_embedder_single_query():
    from app.ingestion.embedder import OpenAIEmbedder
    embedder = OpenAIEmbedder(model="text-embedding-3-small")
    vec = await embedder.embed_query("test query")
    assert len(vec) == 1536
