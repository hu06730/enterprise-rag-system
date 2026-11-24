from abc import ABC, abstractmethod
from langchain_core.documents import Document


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, docs: list[Document], top_k: int =5) -> list[Document]:
        ...