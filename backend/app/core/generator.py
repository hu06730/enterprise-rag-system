import json
import uuid
from app.core.interfaces import Chunk, Message, RetrievalContext
from app.db.sqlite import get_connection


PROMPT_TEMPLATES = {
    "employee": """你是企业内部知识助手。请严格根据以下参考资料回答问题。
如果参考资料中没有相关信息，请明确说"该问题在现有资料中未找到答案"。
请在你的回答中引用相关来源。

参考资料：
{context}

用户问题：{query}""",

    "customer": """你是一个友好的客服助手。请根据以下参考资料回答客户问题。
回答应该简洁、准确、友好。如果资料中没有答案，请礼貌地说你不确定，并建议联系人工客服。
不要编造任何信息。

参考资料：
{context}

客户问题：{query}""",

    "compliance": """你是合规审计助手。请根据以下参考资料逐条回答，每条回答必须标注具体来源。

要求：
1. 每个事实陈述必须引用来源文档名称和段落
2. 如果多条资料有冲突，请明确指出
3. 不得添加参考资料中没有的信息

参考资料：
{context}

查询：{query}""",
}


class Generator:
    def __init__(self, llm):
        self.llm = llm

    def _load_prompt_template(self, kb_id: int, kb_type: str) -> str:
        conn = get_connection()
        row = conn.execute(
            "SELECT prompt_template FROM kb_config WHERE kb_id=?", (kb_id,)
        ).fetchone()
        conn.close()
        if row and row["prompt_template"]:
            return row["prompt_template"]
        return PROMPT_TEMPLATES.get(kb_type, PROMPT_TEMPLATES["employee"])

    def _assemble_context(self, chunks: list[Chunk], max_tokens: int = 4000) -> tuple[str, list[dict]]:
        sorted_chunks = sorted(chunks, key=lambda x: x.score, reverse=True)
        context_parts = []
        sources = []
        total_chars = 0

        for c in sorted_chunks:
            source_text = f"[来源: {c.title}]\n{c.text}"
            if total_chars + len(source_text) > max_tokens * 3:
                break
            context_parts.append(source_text)
            sources.append({
                "doc_title": c.title,
                "chunk_text": c.text[:200],
                "score": round(c.score, 4),
            })
            total_chars += len(source_text)

        return "\n\n---\n\n".join(context_parts), sources

    async def generate(self, query: str, chunks: list[Chunk], ctx: RetrievalContext) -> dict:
        context, sources = self._assemble_context(chunks)
        template = self._load_prompt_template(ctx.kb_id, ctx.kb_type)
        prompt = template.format(context=context, query=query)

        result = await self.llm.generate([Message(role="user", content=prompt)])
        trace_id = str(uuid.uuid4())[:8]

        return {
            "answer": result.content,
            "sources": sources,
            "confidence": self._estimate_confidence(chunks),
            "trace_id": trace_id,
        }

    async def generate_stream(self, query: str, chunks: list[Chunk], ctx: RetrievalContext):
        context, sources = self._assemble_context(chunks)
        template = self._load_prompt_template(ctx.kb_id, ctx.kb_type)
        prompt = template.format(context=context, query=query)

        yield json.dumps({"type": "sources", "data": sources}) + "\n"
        async for token in self.llm.generate_stream([Message(role="user", content=prompt)]):
            yield json.dumps({"type": "token", "data": token}) + "\n"
        trace_id = str(uuid.uuid4())[:8]
        yield json.dumps({"type": "done", "trace_id": trace_id}) + "\n"

    def _estimate_confidence(self, chunks: list[Chunk]) -> float:
        if not chunks:
            return 0.0
        top_score = max(c.score for c in chunks)
        return round(min(top_score * 10, 1.0), 2)
