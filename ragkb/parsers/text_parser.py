from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from ragkb.interfaces.base_parser import BaseParser

class TextParser(BaseParser):
    """加载 txt 文件，返回 List[Document]。"""
    def load(self,file_path:str,encoding : str = "utf-8") -> list[Document]:
        loader = TextLoader(
            file_path=file_path,      # ← 必须用传入的参数，不能写死
            encoding=encoding,        # ← 同上
        )
        documents = loader.load()

        if not documents or not any(d.page_content.strip() for d in documents) :
            raise ValueError(f"文件为空或无可读内容: {file_path}")

        # 显式写 source：多文件入库时，这是唯一能区分「这句话来自哪个文件」的字段
        for doc in documents:
            doc.metadata["source"] = file_path

        return documents


