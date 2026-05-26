def recall_at_k(relevant_ids: list[int], retrieved_ids: list[int], k: int) -> float:
    if not relevant_ids:
        return 1.0
    top_k = set(retrieved_ids[:k])
    hits = top_k.intersection(set(relevant_ids))
    return len(hits) / len(relevant_ids)


def mrr(relevant_ids: list[int], retrieved_ids: list[int]) -> float:
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0
