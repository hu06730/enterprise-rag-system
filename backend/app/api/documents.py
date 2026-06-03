import os
import uuid
from fastapi import APIRouter, UploadFile, File, Form, Depends, BackgroundTasks, HTTPException
from app.auth.permissions import get_current_user, require_role
from app.db.sqlite import get_connection
from app.config import settings
from app.core.bm25_cache import bm25_cache

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    kb_id: int = Form(...),
    file: UploadFile = File(...),
    access_level: str = Form("internal"),
    user: dict = Depends(require_role("admin", "editor")),
):
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    ext = os.path.splitext(file.filename or "unknown")[1]
    stored_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(settings.UPLOAD_DIR, stored_name)

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    file_type = ext.lstrip(".")
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO documents (kb_id, filename, file_path, file_type, access_level, status) VALUES (?, ?, ?, ?, ?, 'pending')",
        (kb_id, file.filename, file_path, file_type, access_level),
    )
    doc_id = cur.lastrowid
    conn.commit()
    conn.close()

    from app.ingestion.pipeline import run_ingestion
    background_tasks.add_task(run_ingestion, doc_id)

    return {"code": 0, "data": {"id": doc_id, "status": "pending"}, "message": "Uploaded, processing in background"}


@router.get("/")
def list_documents(kb_id: int = None, user: dict = Depends(get_current_user)):
    conn = get_connection()
    if kb_id:
        rows = conn.execute("SELECT * FROM documents WHERE kb_id=? ORDER BY created_at DESC", (kb_id,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
    conn.close()
    return {"code": 0, "data": [dict(r) for r in rows], "message": "ok"}


@router.get("/{doc_id}/status")
def get_doc_status(doc_id: int, user: dict = Depends(get_current_user)):
    conn = get_connection()
    doc = conn.execute("SELECT id, filename, status FROM documents WHERE id=?", (doc_id,)).fetchone()
    conn.close()
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")
    return {"code": 0, "data": dict(doc), "message": "ok"}


@router.delete("/{doc_id}")
def delete_document(doc_id: int, user: dict = Depends(require_role("admin", "editor"))):
    from app.db.vec_store import delete_chunks_by_doc
    conn = get_connection()
    doc = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
    if not doc:
        conn.close()
        raise HTTPException(status_code=404, detail="Not found")

    delete_chunks_by_doc(doc_id)
    conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
    conn.commit()
    conn.close()

    # 文档删除，重建该知识库的 BM25 索引
    bm25_cache.rebuild(doc["kb_id"])

    if os.path.exists(doc["file_path"]):
        os.remove(doc["file_path"])
    return {"code": 0, "data": None, "message": "Deleted"}


@router.put("/{doc_id}/reprocess")
async def reprocess(doc_id: int, background_tasks: BackgroundTasks, user: dict = Depends(require_role("admin", "editor"))):
    from app.db.vec_store import delete_chunks_by_doc
    from app.ingestion.pipeline import run_ingestion

    conn = get_connection()
    doc = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
    if not doc:
        conn.close()
        raise HTTPException(status_code=404, detail="Not found")

    delete_chunks_by_doc(doc_id)
    conn.execute("UPDATE documents SET status='pending' WHERE id=?", (doc_id,))
    conn.commit()
    conn.close()

    # 重建 BM25 索引（因为 chunk 被删除了，ingestion 完成后会再次重建）
    bm25_cache.rebuild(doc["kb_id"])

    background_tasks.add_task(run_ingestion, doc_id)
    return {"code": 0, "data": {"id": doc_id, "status": "pending"}, "message": "Reprocessing started"}
