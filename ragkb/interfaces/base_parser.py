from abc import ABC, abstractmethod

from langchain_core.documents import Document


class BaseParser(ABC):
    @abstractmethod
    def load(self,file_path:str,encoding: str="utf-8") ->list[Document]:
        ...
