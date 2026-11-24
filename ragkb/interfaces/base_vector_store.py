from abc import ABC, abstractmethod

from langchain_core.documents import Document


class BaseVectorStore(ABC):
    @abstractmethod
    def connect(self) -> None:
        ...


    @abstractmethod
    def create_collection(self) -> None:
        ...



    @abstractmethod
    def upsert(self, data: list[dict]) -> dict:
        ...


    @abstractmethod
    def search(self, query_vector: list[float], top_k: int) -> list[Document]:
        ...


    @abstractmethod
    def drop_collection(self) -> None:
        ...