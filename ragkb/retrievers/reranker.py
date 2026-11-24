from FlagEmbedding import FlagReranker
from langchain_core.documents import Document

from ragkb.interfaces.base_reranker import BaseReranker


class BgeReranker(BaseReranker):
    def __init__(self,model_name :str = "BAAI/bge-reranker-base",use_fp16: bool = False) :
        self.reranker = FlagReranker(
            model_name_or_path = model_name,
            use_fp16 = use_fp16,
        )

    def rerank(self,query:str,docs:list[Document],top_k:int = 5) -> list[Document] :
        if not docs:
            return []

        pairs = [(query,doc.page_content) for doc in docs]

        scores = self.reranker.compute_score(pairs)

        if isinstance(scores, float):
            scores = [scores]

        doc_score_pairs = list(zip(docs,scores))
        doc_score_pairs.sort(key=lambda x:x[1],reverse=True)

        results = []
        for doc,score in doc_score_pairs[:top_k]:
            results.append(
                Document(
                    page_content = doc.page_content,
                    metadata ={**doc.metadata,"rerank_score":float(score)},
                )
            )

        return results


