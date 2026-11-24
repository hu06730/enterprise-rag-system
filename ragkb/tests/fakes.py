from langchain_core.documents import Document

from ragkb.interfaces.base_embedding import BaseEmbedding
from ragkb.interfaces.base_generator import BaseGenerator
from ragkb.interfaces.base_parser import BaseParser
from ragkb.interfaces.base_splitter import BaseSplitter
from ragkb.interfaces.base_vector_store import BaseVectorStore
from ragkb.interfaces.base_reranker import BaseReranker


class FakeParser(BaseParser):
    def load(self, file_path, encoding="utf-8"):
        return [Document(page_content="假文档内容", metadata={"source": "fake.md"})]

class FakeSplitter(BaseSplitter):
    def split(self, docs):
        for i, d in enumerate(docs):
            d.metadata["chunk_index"] = i
        return docs

class FakeEmbedding(BaseEmbedding):
    def embed_query(self, query):
        return [0.1, 0.2, 0.3]
    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

class FakeVectorStore(BaseVectorStore):
    def connect(self) -> None:
        print("已连接向量库")
        return
    def create_collection(self) -> None:
        print("已创建表")
        return
    def upsert(self, data):
        return {"upsert_count": len(data)}
    def search(self, query_vector, top_k):
        return [Document(page_content="假检索结果", metadata={"source": "fake.md", "chunk_index": 0, "score": 0.9})]
    def drop_collection(self) -> None:
        return

class FakeGenerator(BaseGenerator):
    def generate(self, prompt):
        return "假答案"

class FakeBM25Retriever:
    def build_index(self, docs):
        pass
    def search(self, query, top_k=5):
        return [Document(page_content="假 BM25 结果", metadata={"source": "fake.md"})]

class RecordingGenerator(BaseGenerator):
    """假生成器：不调 LLM，但把收到的 prompt 存下来，供测试检查。"""

    def __init__(self):
        self.last_prompt = None
        self.call_count = 0

    def generate(self, prompt):
        self.last_prompt = prompt
        self.call_count += 1
        return "假答案"


class FakeReranker(BaseReranker):
    """假重排器：不加载模型，用递减分数模拟「越靠前，分越高」。

    除了假装打分，还额外记录三样东西，方便测试断言：
    - called   : 重排有没有被调用（验证它没被 if 跳过）
    - received : 收到的候选文档（验证它接在 RRF 之后）
    - assigned : page_content -> 分数（用来算谁该排第一）
    """

    def __init__(self):
        self.called = False
        self.received = []
        self.assigned = {}

    def rerank(self, query, docs, top_k=5):
        if not docs:
            return []

        self.called = True
        self.received = list(docs)

        scored = []
        for i, doc in enumerate(docs):
            score = 1.0 - i * 0.1
            self.assigned[doc.page_content] = score
            scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        # 与真实 BgeReranker 保持一致：返回新 Document，不改原对象
        return [
            Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "rerank_score": score},
            )
            for doc, score in scored[:top_k]
        ]


class FakeRewriter:
    def __init__(self):
        self.called = False
    def rewrite(self, query, n=3):
        self.called = True
        return [query, "子查询A", "子查询B"]