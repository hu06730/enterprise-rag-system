import pytest
from app.core.interfaces import Chunk, RetrievalContext


@pytest.mark.asyncio
async def test_generator_returns_answer_with_sources():
    from app.core.generator import Generator
    from app.llm.openai_llm import OpenAILLM

    llm = OpenAILLM()
    gen = Generator(llm=llm)

    chunks = [
        Chunk(id=1, doc_id=1, kb_id=1, chunk_index=0, title="员工手册.pdf",
              text="年假天数为每年15个工作日。", score=0.95, source="vector"),
        Chunk(id=2, doc_id=2, kb_id=1, chunk_index=1, title="FAQ.docx",
              text="申请年假需要提前3天在OA系统提交。", score=0.87, source="keyword"),
    ]
    ctx = RetrievalContext(kb_id=1, kb_type="employee")

    result = await gen.generate(query="年假怎么申请", chunks=chunks, ctx=ctx)
    assert result["answer"] != ""
    assert len(result["sources"]) > 0
    assert result["trace_id"] is not None
