import json
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.auth.permissions import get_current_user
from app.core.interfaces import RetrievalContext, Message as CoreMessage
from app.core.intent import classify_intent
from app.core.retriever import HybridRetriever
from app.core.generator import Generator
from app.ingestion.embedder import OpenAIEmbedder
from app.llm.openai_llm import OpenAILLM
from app.db.sqlite import get_connection

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

embedder = OpenAIEmbedder()
llm = OpenAILLM()
retriever = HybridRetriever(embedder=embedder)
generator = Generator(llm=llm)


class ChatRequest(BaseModel):
    kb_id: int
    query: str
    history: list[dict] = []
    stream: bool = False


class FeedbackRequest(BaseModel):
    trace_id: str
    feedback: int
    feedback_text: str = ""


@router.post("/ask")
async def ask(req: ChatRequest, user: dict = Depends(get_current_user)):
    conn = get_connection()
    kb = conn.execute("SELECT * FROM knowledge_bases WHERE id=?", (req.kb_id,)).fetchone()
    config = conn.execute("SELECT * FROM kb_config WHERE kb_id=?", (req.kb_id,)).fetchone()
    conn.close()

    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    intent = classify_intent(req.query)

    ctx = RetrievalContext(
        kb_id=req.kb_id,
        kb_type=kb["kb_type"],
        user_departments=user["departments"],
        user_role=user["role"],
        access_levels=["public", "internal", "restricted"],
        top_k=config["top_k"] if config else 10,
        min_score=config["min_score"] if config else 0.0,
        retrieval_mode=config["retrieval_mode"] if config else "hybrid",
        vector_weight=intent["vector_weight"],
        bm25_weight=intent["bm25_weight"],
        rerank_enabled=bool(config["rerank_enabled"]) if config else True,
    )

    chunks = await retriever.retrieve(req.query, ctx)
    result = await generator.generate(req.query, chunks, ctx)

    conn = get_connection()
    conn.execute(
        "INSERT INTO audit_logs (user_id, kb_id, query_text, answer_text, sources, trace_id) VALUES (?, ?, ?, ?, ?, ?)",
        (user["id"], req.kb_id, req.query, result["answer"],
         json.dumps(result["sources"], ensure_ascii=False), result["trace_id"]),
    )
    conn.commit()
    conn.close()

    return {"code": 0, "data": result, "message": "ok"}


@router.post("/ask/stream")
async def ask_stream(req: ChatRequest, user: dict = Depends(get_current_user)):
    conn = get_connection()
    kb = conn.execute("SELECT * FROM knowledge_bases WHERE id=?", (req.kb_id,)).fetchone()
    config = conn.execute("SELECT * FROM kb_config WHERE kb_id=?", (req.kb_id,)).fetchone()
    conn.close()

    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    intent = classify_intent(req.query)
    ctx = RetrievalContext(
        kb_id=req.kb_id, kb_type=kb["kb_type"],
        user_departments=user["departments"], user_role=user["role"],
        access_levels=["public", "internal", "restricted"],
        top_k=config["top_k"] if config else 10,
        retrieval_mode=config["retrieval_mode"] if config else "hybrid",
        vector_weight=intent["vector_weight"],
        bm25_weight=intent["bm25_weight"],
    )

    chunks = await retriever.retrieve(req.query, ctx)

    async def event_stream():
        async for event in generator.generate_stream(req.query, chunks, ctx):
            yield f"data: {event}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/feedback")
def submit_feedback(req: FeedbackRequest, user: dict = Depends(get_current_user)):
    conn = get_connection()
    conn.execute(
        "UPDATE audit_logs SET feedback=?, feedback_text=? WHERE trace_id=? AND user_id=?",
        (req.feedback, req.feedback_text, req.trace_id, user["id"]),
    )
    conn.commit()
    conn.close()
    return {"code": 0, "data": None, "message": "Feedback recorded"}
