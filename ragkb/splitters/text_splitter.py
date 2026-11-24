from langchain_core.documents import Document

from ragkb.interfaces.base_splitter import BaseSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextSplitter(BaseSplitter):
    def __init__(self,chunk_size:int =200,chunk_overlap:int =80):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", " ", ""],
        )

    def split(self,docs:list[Document]) -> list[Document]:
        if not docs:
            raise ValueError("输入文档为空")

        chunks= self.splitter.split_documents(docs)

        if not chunks:
            raise ValueError("切分后为空")

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i

        return chunks