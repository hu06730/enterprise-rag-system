from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections.abc import AsyncIterator


@dataclass
class Message:
    role: str
    content: str


@dataclass
class GenerationResult:
    content: str
    model: str = ""
    usage: dict = field(default_factory=dict)


@dataclass
class Chunk:
    id: int
    doc_id: int
    kb_id: int
    chunk_index: int
    title: str
    text: str
    department: str | None = None
    access_level: str = "internal"
    score: float = 0.0
    source: str = ""


@dataclass
class RetrievalContext:
    kb_id: int
    kb_type: str = "employee"
    user_departments: list[str] = field(default_factory=list)
    user_role: str = "viewer"
    access_levels: list[str] = field(default_factory=lambda: ["public", "internal"])
    top_k: int = 10
    min_score: float = 0.0
    retrieval_mode: str = "hybrid"
    vector_weight: float = 0.5
    bm25_weight: float = 0.5
    rerank_enabled: bool = True


@dataclass
class ParsedDocument:
    text: str
    title: str = ""
    metadata: dict = field(default_factory=dict)
    pages: list[str] = field(default_factory=list)


class BaseRetriever(ABC):
    @abstractmethod
    async def retrieve(self, query: str, ctx: RetrievalContext) -> list[Chunk]: ...


class BaseSplitter(ABC):
    @abstractmethod
    def split(self, text: str, **kwargs) -> list[str]: ...


class BaseLLM(ABC):
    @abstractmethod
    async def generate(self, messages: list[Message], **kwargs) -> GenerationResult: ...

    @abstractmethod
    async def generate_stream(self, messages: list[Message]) -> AsyncIterator[str]: ...


class BaseEmbedder(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]: ...


class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_path: str) -> ParsedDocument: ...
