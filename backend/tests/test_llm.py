import pytest


@pytest.mark.asyncio
async def test_openai_llm_generate():
    from app.llm.openai_llm import OpenAILLM
    from app.core.interfaces import Message
    llm = OpenAILLM()
    result = await llm.generate([Message(role="user", content="Reply with just: OK")])
    assert "OK" in result.content


@pytest.mark.asyncio
async def test_openai_llm_stream():
    from app.llm.openai_llm import OpenAILLM
    from app.core.interfaces import Message
    llm = OpenAILLM()
    chunks = []
    async for chunk in llm.generate_stream([Message(role="user", content="Say hello")]):
        chunks.append(chunk)
    assert len(chunks) > 0
    assert "".join(chunks).strip() != ""
