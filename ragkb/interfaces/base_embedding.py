from abc import ABC, abstractmethod


class BaseEmbedding(ABC):
    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        ...

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...