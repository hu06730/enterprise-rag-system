from pymilvus import MilvusClient
from ragkb.interfaces.base_vector_store import BaseVectorStore
from langchain_core.documents import Document
import hashlib

def build_milvus_data(docs:list[Document],vectors:list[list[float]],id_field:str = "id",
    vector_field:str = "vector",text_field: str = "text",source_field: str = "source",
    chunk_index_field: str = "chunk_index",) -> list[dict]:

    if len(docs)!=len(vectors):
        raise ValueError("docs,vectors长度不一致")

    data = []
    for doc,vector in zip(docs,vectors):
        source = doc.metadata.get("source","unknown")
        chunk_index = doc.metadata.get("chunk_index",0)
        mixed = f"{source}|{chunk_index}|{doc.page_content}"
        doc_id = hashlib.md5(mixed.encode("utf-8")).hexdigest()
        data.append({
            id_field: doc_id,
            text_field:doc.page_content,
            vector_field:vector,
            source_field:source,
            chunk_index_field:chunk_index,
        })
    return data


class MilvusStore(BaseVectorStore):
    def __init__(self, uri: str, collection_name: str, db_name: str, dimension: int, metric_type: str = "COSINE"):
        self.uri = uri
        self.collection_name = collection_name
        self.db_name = db_name
        self.dimension = dimension
        self.metric_type = metric_type
        self.client = None


    def connect(self) -> None:
        self.client = MilvusClient(self.uri)

        existed_db = self.client.list_databases()
        if self.db_name not in existed_db:
            self.client.create_database(db_name=self.db_name)

        self.client.use_database(self.db_name)

    def create_collection(self) -> None:
        """
        幂等创建集合：不存在则创建，已存在则跳过。
        适用于生产环境，避免误删数据。
        """
        if self.client.has_collection(self.collection_name):
            return  # 集合已存在，直接返回，不做任何破坏性操作

        # 以下为创建集合、索引和加载的完整逻辑
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=self.dimension,
            metric_type=self.metric_type,
            auto_id=False,  # 主键由你提供
            primary_field_name="id",  # 主键字段名
            id_type="string",  # ← 关键：主键是 VARCHAR
            max_length=32,  # sha256 hexdigest 是 64 字符；如果用 md5 就 32
        )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type=self.metric_type,
        )

    def upsert(self, data: list[dict]) -> dict:
        result = self.client.upsert(
            collection_name=self.collection_name,
            data=data,
        )
        self.client.flush(collection_name=self.collection_name)

        return {
            "upsert_count": getattr(result, 'upsert_count', 0),
            "ids": getattr(result, 'primary_keys', []),
        }


    def search(self, query_vector: list[float], top_k: int) -> list[Document]:
        results = self.client.search(
            collection_name=self.collection_name,
            data = [query_vector],
            limit = top_k,
            output_fields=["text", "source", "chunk_index"],
        )

        docs = []
        for hit in results[0]:
            docs.append(Document(
                page_content=hit["entity"]["text"],
                metadata={
                    "source": hit["entity"]["source"],
                    "chunk_index": hit["entity"]["chunk_index"],
                    "score": hit["distance"],
                },
            ))

        return docs

    def drop_collection(self) -> None:
        self.client.drop_collection(self.collection_name)








