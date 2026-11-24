import math

from langchain_core.documents import Document

from ragkb.interfaces.base_reranker import BaseReranker
from ragkb.interfaces.base_embedding import BaseEmbedding
from ragkb.interfaces.base_parser import BaseParser
from ragkb.interfaces.base_splitter import BaseSplitter
from ragkb.interfaces.base_vector_store import BaseVectorStore
from ragkb.interfaces.base_generator import BaseGenerator

from ragkb.vectorstores.milvus_store import build_milvus_data
from ragkb.interfaces.prompt_builder import build_prompt
from ragkb.retrievers.bm25_retriever import BM25Retriever

from ragkb.retrievers.rrf_fusion import rrf_fuse,rrf_fuse_many
from ragkb.retrievers.query_rewriter import QueryRewriter

# 拒答时的固定回复：CLI 和 HTTP 接口共用同一个字符串，便于调用方判断
REFUSAL_TEXT = "根据现有资料无法回答。"


class RAGPipeline:
    def __init__(self,parser: BaseParser,splitter: BaseSplitter,generator: BaseGenerator,
                 vector_store: BaseVectorStore,embedding_model:BaseEmbedding,
                 bm25_retriever:BM25Retriever,reranker:BaseReranker | None = None,
                 query_rewriter:QueryRewriter | None = None,refuse_threshold : float = 0.0,) :
        self.refuse_threshold = refuse_threshold
        self.parser = parser
        self.splitter = splitter
        self.generator = generator
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
        self.query_rewriter = query_rewriter

    def ingest(self,file_paths:str | list[str],encoding:str="utf-8") ->int:
        """支持单文件（str）和多文件（list[str]）。

            关键：所有文件先全部 load + split 合并成一个大 chunks 列表，
                 再一次性 build_index + 一次性 upsert。
                 不能循环调 build_index，否则 BM25 只记住最后一个文件。
            """
        # 入口归一化：单个 str 也要当成"一个元素的列表"，
        # 否则 for 会逐字符迭代 "C:\..." → 'C'、':'、'\' ...
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        chunks: list[Document] = []
        for file_path in file_paths:
            docs = self.parser.load(file_path,encoding)

            # extend 而不是 append：append 会得到 [[doc, doc], [doc]] 的嵌套列表
            chunks.extend(self.splitter.split(docs))

        texts = [chunk.page_content for chunk in chunks]

        vectors = self.embedding_model.embed_documents(texts)

        data= build_milvus_data(chunks,vectors)

        self.bm25_retriever.build_index(chunks)

        result = self.vector_store.upsert(data)

        return len(data)


    def retrieve(self,query:str,top_k:int=5) -> list[Document]:
        query_vec = self.embedding_model.embed_query(query)

        # 1. 稠密检索
        dense_docs = self.vector_store.search(query_vec, top_k=15)
        # 2. 稀疏检索
        sparse_docs = self.bm25_retriever.search(query, top_k=15)
        # 3. RRF 融合
        fused_docs = rrf_fuse(dense_docs, sparse_docs, top_k=15)

        # 4.rerank重排
        try:
            if self.reranker is not None:
                final_docs = self.reranker.rerank(query, fused_docs, top_k=top_k)
            else:
                final_docs = fused_docs[:top_k]
        except Exception as e:
            print(f"[Warning] 重排失败，降级使用 RRF 结果: {e}")
            final_docs = fused_docs[:top_k]

        return final_docs

    def retrieve_multi(self, query: str, top_k: int = 5, n: int = 3) -> list[Document]:
        """多查询检索。

            1. query_rewriter 为 None 或没接 → 直接调用 self.retrieve(query, top_k) 退化
            2. 改写得到子查询列表；为空 → return []
            3. 每个子查询：embed → 稠密 search(15) → 稀疏 search(15)  → 收集成 2N 个列表
            4. rrf_fuse_many(所有列表, top_k=15)
            5. rerank(原 query, 融合结果, top_k)   ← 注意：重排用**原始问题**，不是子问题
            6. 重排异常 → 降级取融合前 top_k（沿用你已有的 try/except 模式）
            """

        if self.query_rewriter is None:
            return self.retrieve(query, top_k)

        query_list = self.query_rewriter.rewrite(query,n)

        if not query_list:
            return []

        total_docs = []

        for q in query_list:
            query_vec = self.embedding_model.embed_query(q)
            # 1. 稠密检索
            total_docs.append(self.vector_store.search(query_vec, top_k=15))
            # 2. 稀疏检索
            total_docs.append(self.bm25_retriever.search(q, top_k=15))

        fused_docs = rrf_fuse_many(total_docs, top_k=15)

        try:
            if self.reranker is not None:
                final_docs = self.reranker.rerank(query, fused_docs, top_k=top_k)
            else:
                final_docs = fused_docs[:top_k]
        except Exception as e:
            print(f"[Warning] 重排失败，降级使用 RRF 结果: {e}")
            final_docs = fused_docs[:top_k]

        return final_docs

    def _top_sigmoid(self,docs):
        """把排第一的文档的重排分数换算成 0~1；没有重排分数就返回 None。"""
        if not docs:
            return None

        score = docs[0].metadata.get("rerank_score")
        if score is None:
            return None        # 没开重排 → 不做阈值判断（见下面第四节）

        return 1.0 / (1.0 + math.exp(-score))

    def answer(self, query: str, top_k: int = 5) -> tuple[str, list[Document], bool]:
        """检索 + 生成，同时把"用到了哪些片段"和"是否拒答"一起返回。

        HTTP 接口需要这三样：answer（回答）、docs（来源）、refused（是否拒答）。
        query() 只是它的薄包装——这样 CLI 行为不变，API 又能拿到溯源信息。
        """
        docs = self.retrieve(query, top_k)

        score = self._top_sigmoid(docs)
        if not docs or (score is not None and score < self.refuse_threshold):
            return REFUSAL_TEXT, docs, True

        prompt = build_prompt(query, docs)
        return self.generator.generate(prompt), docs, False

    def query(self,query:str,top_k : int=5) ->str:
        """只要答案字符串（CLI 用）。逻辑全部委托给 answer()。"""
        return self.answer(query, top_k)[0]




