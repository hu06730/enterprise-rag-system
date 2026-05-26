import json
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.auth.permissions import get_current_user, require_role
from app.db.sqlite import get_connection

router = APIRouter(prefix="/api/v1/knowledge-bases", tags=["knowledge-bases"])


class KBCreate(BaseModel):
    name: str
    description: str = ""
    kb_type: str = "employee"
    access_level: str = "internal"
    allowed_departments: list[str] = []
    allowed_users: list[int] = []
    chunk_strategy: str = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 50
    retrieval_mode: str = "hybrid"
    top_k: int = 10
    vector_weight: float = 0.5
    bm25_weight: float = 0.5


@router.get("/")
def list_kbs(user: dict = Depends(get_current_user)):
    conn = get_connection()
    rows = conn.execute("SELECT * FROM knowledge_bases").fetchall()
    conn.close()
    return {"code": 0, "data": [dict(r) for r in rows], "message": "ok"}


@router.post("/")
def create_kb(req: KBCreate, user: dict = Depends(require_role("admin", "editor"))):
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO knowledge_bases (name, description, kb_type, access_level, allowed_departments, allowed_users) VALUES (?, ?, ?, ?, ?, ?)",
        (req.name, req.description, req.kb_type, req.access_level,
         json.dumps(req.allowed_departments), json.dumps(req.allowed_users)),
    )
    kb_id = cur.lastrowid
    conn.execute(
        """INSERT INTO kb_config (kb_id, chunk_strategy, chunk_size, chunk_overlap, retrieval_mode, top_k, vector_weight, bm25_weight)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (kb_id, req.chunk_strategy, req.chunk_size, req.chunk_overlap,
         req.retrieval_mode, req.top_k, req.vector_weight, req.bm25_weight),
    )
    conn.commit()
    conn.close()
    return {"code": 0, "data": {"id": kb_id}, "message": "Knowledge base created"}


@router.get("/{kb_id}")
def get_kb(kb_id: int, user: dict = Depends(get_current_user)):
    conn = get_connection()
    kb = conn.execute("SELECT * FROM knowledge_bases WHERE id=?", (kb_id,)).fetchone()
    config = conn.execute("SELECT * FROM kb_config WHERE kb_id=?", (kb_id,)).fetchone()
    conn.close()
    if not kb:
        raise HTTPException(status_code=404, detail="Not found")
    return {"code": 0, "data": {"kb": dict(kb), "config": dict(config) if config else None}, "message": "ok"}


@router.put("/{kb_id}")
def update_kb(kb_id: int, req: KBCreate, user: dict = Depends(require_role("admin", "editor"))):
    conn = get_connection()
    conn.execute(
        "UPDATE knowledge_bases SET name=?, description=?, kb_type=?, access_level=?, allowed_departments=?, allowed_users=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
        (req.name, req.description, req.kb_type, req.access_level,
         json.dumps(req.allowed_departments), json.dumps(req.allowed_users), kb_id),
    )
    conn.execute(
        "UPDATE kb_config SET chunk_strategy=?, chunk_size=?, chunk_overlap=?, retrieval_mode=?, top_k=?, vector_weight=?, bm25_weight=? WHERE kb_id=?",
        (req.chunk_strategy, req.chunk_size, req.chunk_overlap, req.retrieval_mode,
         req.top_k, req.vector_weight, req.bm25_weight, kb_id),
    )
    conn.commit()
    conn.close()
    return {"code": 0, "data": None, "message": "Updated"}


@router.delete("/{kb_id}")
def delete_kb(kb_id: int, user: dict = Depends(require_role("admin"))):
    from app.db.vec_store import delete_chunks_by_kb
    delete_chunks_by_kb(kb_id)
    conn = get_connection()
    conn.execute("DELETE FROM kb_config WHERE kb_id=?", (kb_id,))
    conn.execute("DELETE FROM documents WHERE kb_id=?", (kb_id,))
    conn.execute("DELETE FROM knowledge_bases WHERE id=?", (kb_id,))
    conn.commit()
    conn.close()
    return {"code": 0, "data": None, "message": "Deleted"}
