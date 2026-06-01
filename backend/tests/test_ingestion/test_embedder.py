import pytest
from app.config import settings


@pytest.mark.asyncio
async def test_openai_embedder_returns_correct_dimensions():
    from app.ingestion.embedder import OpenAIEmbedder
    embedder = OpenAIEmbedder()  # 使用配置的默认模型
    result = await embedder.embed(["hello world"])
    assert len(result) == 1
    assert len(result[0]) == settings.EMBEDDING_DIMENSIONS


@pytest.mark.asyncio
async def test_openai_embedder_single_query():
    from app.ingestion.embedder import OpenAIEmbedder
    embedder = OpenAIEmbedder()  # 使用配置的默认模型
    vec = await embedder.embed_query("test query")
    assert len(vec) == settings.EMBEDDING_DIMENSIONS
