from openai import AsyncOpenAI
from app.core.interfaces import BaseEmbedder
from app.config import settings


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model: str | None = None):
        self.model = model or settings.EMBEDDING_MODEL
        self.client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        resp = await self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

    async def embed_query(self, text: str) -> list[float]:
        results = await self.embed([text])
        return results[0]
