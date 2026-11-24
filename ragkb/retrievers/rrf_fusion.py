from langchain_core.documents import Document


def rrf_fuse(
        dense_docs:list[Document],
        sparse_docs:list[Document],
        k:int=60,
        top_k:int=5,) -> list[Document]:

    # 用一个字典记录每个文档的 RRF 分数。key 用 page_content 去重
    rrf_scores ={}
    # 记录文档原文，方便最后重建 Document
    doc_map ={}

    # 1. 计算稠密检索的 RRF 得分
    for rank, doc in enumerate(dense_docs, start=1):
        key = doc.page_content
        doc_map[key] = doc
        rrf_scores[key] = rrf_scores.get(key,0.0) + (1.0 / (k + rank))

    # 2. 计算稀疏检索的 RRF 得分
    for rank,doc in enumerate(sparse_docs, start=1):
        key = doc.page_content
        doc_map[key] = doc
        rrf_scores[key] = rrf_scores.get(key,0.0) + (1.0 / (k + rank))

    # 3. 按 RRF 分数降序排序
    sorted_items = sorted(rrf_scores.items(), key=lambda x:x[1], reverse=True)

    results = []
    for key, score in sorted_items[:top_k]:
        doc = doc_map[key]
        results.append(Document(page_content=key, metadata={**doc.metadata,"rrf_score":score}))

    return results


def rrf_fuse_many(ranked_lists: list[list[Document]], k: int = 60, top_k: int = 15) -> list[Document]:
    """对 N 个"已排好序"的列表做 RRF 融合。

    做法：遍历每个列表，对每个文档累加 1/(k+rank)；
         同一个文档出现在多个列表里，分数就累加多次（这代表"多路都认为它相关"）。

    输入：[q1稠密, q1稀疏, q2稠密, q2稀疏, q3稠密, q3稀疏]  ← 2N 个列表
    输出：按累加分数降序的 List[Document]，metadata 里带 rrf_score
    """
    # 用一个字典记录每个文档的 RRF 分数。key 用 page_content 去重
    rrf_scores = {}
    # 用一个字典记录每个文档的 命中次数。key 用 page_content 去重
    hit_count = {}
    # 记录文档原文，方便最后重建 Document
    doc_map = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = doc.page_content
            doc_map[key] = doc
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (k + rank))
            hit_count[key] =  hit_count.get(key, 0) + 1

    # 3. 按 RRF 分数降序排序
    sorted_items = sorted(rrf_scores.items(), key=lambda x:x[1], reverse=True)

    results = []

    for key, score in sorted_items[:top_k]:
        doc = doc_map[key]  #如果同一内容在不同列表里 metadata 不同，保留的是最后那次
        results.append(Document(page_content=key, metadata={**doc.metadata,"rrf_score":score,"hit_count":hit_count}))

    return results









