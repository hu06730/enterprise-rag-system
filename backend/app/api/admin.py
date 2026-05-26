import json
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.auth.permissions import get_current_user, require_role
from app.db.sqlite import get_connection

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


class ConfigUpdate(BaseModel):
    chunk_strategy: str | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    retrieval_mode: str | None = None
    top_k: int | None = None
    min_score: float | None = None
    rerank_enabled: bool | None = None
    vector_weight: float | None = None
    bm25_weight: float | None = None
    prompt_template: str | None = None


@router.get("/config/{kb_id}")
def get_config(kb_id: int, user: dict = Depends(require_role("admin", "editor"))):
    conn = get_connection()
    config = conn.execute("SELECT * FROM kb_config WHERE kb_id=?", (kb_id,)).fetchone()
    conn.close()
    if not config:
        raise HTTPException(status_code=404, detail="Not found")
    return {"code": 0, "data": dict(config), "message": "ok"}


@router.put("/config/{kb_id}")
def update_config(kb_id: int, req: ConfigUpdate, user: dict = Depends(require_role("admin"))):
    conn = get_connection()
    fields = {k: v for k, v in req.model_dump().items() if v is not None}
    if fields:
        set_clause = ", ".join(f"{k}=?" for k in fields)
        conn.execute(f"UPDATE kb_config SET {set_clause} WHERE kb_id=?", (*fields.values(), kb_id))
        conn.commit()
    conn.close()
    return {"code": 0, "data": None, "message": "Config updated"}


@router.get("/stats")
def get_stats(user: dict = Depends(require_role("admin", "editor"))):
    conn = get_connection()
    kb_count = conn.execute("SELECT COUNT(*) FROM knowledge_bases").fetchone()[0]
    doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    chunk_count = conn.execute("SELECT COUNT(*) FROM chunks_meta").fetchone()[0]
    query_count = conn.execute("SELECT COUNT(*) FROM audit_logs").fetchone()[0]
    feedback_up = conn.execute("SELECT COUNT(*) FROM audit_logs WHERE feedback=1").fetchone()[0]
    feedback_down = conn.execute("SELECT COUNT(*) FROM audit_logs WHERE feedback=-1").fetchone()[0]
    conn.close()
    return {"code": 0, "data": {
        "knowledge_bases": kb_count,
        "documents": doc_count,
        "chunks": chunk_count,
        "total_queries": query_count,
        "feedback_up": feedback_up,
        "feedback_down": feedback_down,
    }, "message": "ok"}
