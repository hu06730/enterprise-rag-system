from app.core.interfaces import Message
from app.llm.openai_llm import OpenAILLM


async def judge_faithfulness(answer: str, context: str) -> float:
    llm = OpenAILLM()
    result = await llm.generate([
        Message(role="system", content="""You are an evaluation judge. Score the answer's faithfulness to the context.
Score 1.0 = fully supported, 0.5 = partially, 0.0 = contradicts or unsupported.
Reply with only a number, like 0.85."""),
        Message(role="user", content=f"Context:\n{context}\n\nAnswer:\n{answer}\n\nScore:"),
    ])
    try:
        return float(result.content.strip())
    except ValueError:
        return 0.0


async def judge_relevance(answer: str, query: str) -> float:
    llm = OpenAILLM()
    result = await llm.generate([
        Message(role="system", content="""Score how relevant the answer is to the query.
1.0 = perfectly relevant, 0.0 = completely off-topic.
Reply with only a number, like 0.92."""),
        Message(role="user", content=f"Query: {query}\n\nAnswer: {answer}\n\nScore:"),
    ])
    try:
        return float(result.content.strip())
    except ValueError:
        return 0.0
