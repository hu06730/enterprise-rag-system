from langchain_core.documents import Document


def build_prompt(query:str,docs:list[Document],max_chars:int= 3000) -> str:

    if not docs:
        return f'''你是一个企业知识库助手，
根据现有资料无法回答该问题，因为未检索到相关文档。
用户问题：{query}
请直接回答“根据现有资料无法回答”.
'''

    def _sort_key(document: Document) -> float:
        """按可用分数降序。显式判断键存在，避免 0.0 被 `or` 当成"没有值"。"""
        meta = document.metadata
        for key in ("rerank_score", "rrf_score", "score"):
            if meta.get(key) is not None:
                return float(meta[key])
        return 0.0

    docs = sorted(docs,
                  key=_sort_key,
                  reverse=True)
    context_parts= []
    total_chars= 0
    sign = False

    for i,doc in enumerate(docs,1):
        source = doc.metadata.get("source", "未知")
        title = doc.metadata.get("title", "无标题")
        piece = f"[{i}] 来源：{source} | 标题：{title}\n{doc.page_content}"

        if total_chars+len(piece)>max_chars:
            sign = True
            break

        context_parts.append(piece)
        total_chars+=len(piece)

    context = "\n\n".join(context_parts)
    if sign:
        context +="\n\n(后续片段因长度限制被截断)"


    prompt = f'''你是一个严谨的企业知识库助手。
请严格根据以下文档片段回答用户问题。
要求：
1. 只使用文档中的信息，不要编造。
2. 如果文档不足以回答，请明确说“根据现有资料无法回答”,
3. 回答时用 [1]、[2] 标注引用的文档编号。

文档：
{context}

用户问题：
{query}
'''
    return prompt

