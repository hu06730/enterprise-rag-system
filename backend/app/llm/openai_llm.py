from collections.abc import AsyncIterator
from openai import AsyncOpenAI
from app.core.interfaces import BaseLLM, Message, GenerationResult
from app.config import settings


class OpenAILLM(BaseLLM):
    def __init__(self, model: str | None = None):
        self.model = model or settings.LLM_MODEL
        self.client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
        )

    async def generate(self, messages: list[Message], **kwargs) -> GenerationResult:
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=kwargs.get("temperature", 0.3),
            max_tokens=kwargs.get("max_tokens", 2048),
        )
        choice = resp.choices[0]
        return GenerationResult(
            content=choice.message.content or "",
            model=resp.model,
            usage={
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
            },
        )

    async def generate_stream(self, messages: list[Message]) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            stream=True,
            temperature=0.3,
            max_tokens=2048,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
