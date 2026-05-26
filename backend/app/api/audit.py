from fastapi import APIRouter, Depends, Query, HTTPException
from app.auth.permissions import get_current_user, require_role
from app.db.sqlite import get_connection

router = APIRouter(prefix="/api/v1/audit", tags=["audit"])


@router.get("/logs")
def list_logs(
    kb_id: int | None = Query(None),
    user_id: int | None = Query(None),
    feedback: int | None = Query(None),
    limit: int = Query(50, le=200),
    offset: int = 0,
    user: dict = Depends(require_role("admin", "editor")),
):
    conn = get_connection()
    query = "SELECT * FROM audit_logs WHERE 1=1"
    params = []
    if kb_id is not None:
        query += " AND kb_id=?"; params.append(kb_id)
    if user_id is not None:
        query += " AND user_id=?"; params.append(user_id)
    if feedback is not None:
        query += " AND feedback=?"; params.append(feedback)
    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"; params.extend([limit, offset])
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return {"code": 0, "data": [dict(r) for r in rows], "message": "ok"}


@router.get("/logs/{log_id}")
def get_log(log_id: int, user: dict = Depends(require_role("admin", "editor"))):
    conn = get_connection()
    row = conn.execute("SELECT * FROM audit_logs WHERE id=?", (log_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    return {"code": 0, "data": dict(row), "message": "ok"}
