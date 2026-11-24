from langchain_core.documents import Document
import jieba
from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.docs = []
        self.tokenized_corpus = []



    def build_index(self,docs:list[Document]):
        """接收 chunks，构建 BM25 索引"""
        self.docs = docs
        # 1. 对每个文档的分词
        self.tokenized_corpus = [list(jieba.cut(doc.page_content)) for doc in docs]
        # 2. 初始化 BM25
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self,query:str,top_k:int=5) -> list[Document]:
        """检索并返回 Top-K"""
        if not self.docs or not self.bm25:
            return []
        # 1. query 分词
        tokenized_query = list(jieba.cut(query))
        # 2. 获取所有文档的 BM25 分数（这是一个列表，长度和 docs 相同）
        scores = self.bm25.get_scores(tokenized_query)
        # 3. 将分数与文档对应，并按分数降序排序
        doc_score_pairs = list(zip(self.docs, scores))
        # 按照分数降序排
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # 4. 取 Top-K，并将分数写回 metadata
        results = []
        for doc, score in doc_score_pairs[:top_k]:
            # 注意：这里直接修改原 Document 对象可能会影响其他检索（因为内存里同一个对象可能被复用），
            # 但第一版为了简单，先直接加上，或者用 copy.deepcopy(doc)
            doc.metadata['bm25_score'] = float(score)
            results.append(doc)

        return results


