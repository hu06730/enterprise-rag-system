import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.config import settings


MOCK_EMBEDDING = [0.1] * settings.EMBEDDING_DIMENSIONS


class MockEmbeddingData:
    def __init__(self, embedding):
        self.embedding = embedding


class MockEmbeddingResponse:
    def __init__(self, texts):
        self.data = [MockEmbeddingData(MOCK_EMBEDDING) for _ in texts]


class MockUsage:
    prompt_tokens = 10
    completion_tokens = 20


class MockMessage:
    def __init__(self, content="OK"):
        self.content = content
        self.role = "assistant"


class MockChoice:
    def __init__(self, content="OK"):
        self.message = MockMessage(content)


class MockChatResponse:
    def __init__(self, content="OK"):
        self.choices = [MockChoice(content)]
        self.model = "mock-model"
        self.usage = MockUsage()


class MockStreamDelta:
    def __init__(self, content):
        self.content = content


class MockStreamChoice:
    def __init__(self, content):
        self.delta = MockStreamDelta(content)


class MockStreamChunk:
    def __init__(self, content):
        self.choices = [MockStreamChoice(content)]


class MockChatStream:
    def __init__(self, content="OK"):
        self.chunks = [MockStreamChunk(c) for c in content]
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self.chunks):
            raise StopAsyncIteration
        chunk = self.chunks[self._index]
        self._index += 1
        return chunk


@pytest.fixture(autouse=True)
def mock_openaiEmbeddings():
    mock_client = AsyncMock()
    mock_client.embeddings.create = AsyncMock(
        side_effect=lambda **kwargs: MockEmbeddingResponse(kwargs.get("input", []))
    )
    with patch("app.ingestion.embedder.AsyncOpenAI", return_value=mock_client):
        yield mock_client


@pytest.fixture(autouse=True)
def mock_openai_chat():
    mock_client = AsyncMock()

    async def mock_create(**kwargs):
        if kwargs.get("stream"):
            return MockChatStream("OK")
        return MockChatResponse("OK")

    mock_client.chat.completions.create = AsyncMock(side_effect=mock_create)
    with patch("app.llm.openai_llm.AsyncOpenAI", return_value=mock_client):
        yield mock_client
