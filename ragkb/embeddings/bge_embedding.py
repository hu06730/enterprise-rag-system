from langchain_community.embeddings import HuggingFaceEmbeddings

from ragkb.interfaces.base_embedding import BaseEmbedding


# model_name="BAAI/bge-m3"
# model_kwargs = {'device': 'cpu'}
# encode_kwargs =  {"normalize_embeddings": True}

class BgeEmbedding(BaseEmbedding):
    def __init__(self, model_name:str= "BAAI/bge-m3",device:str = "cpu", normalize_embeddings:bool =True):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name = model_name,
            model_kwargs ={"device":device,},
            encode_kwargs ={"normalize_embeddings":normalize_embeddings},
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return  self.embedding_model.embed_documents(texts)

    def embed_query(self, query: str) -> list[float]:
        return self.embedding_model.embed_query(query)