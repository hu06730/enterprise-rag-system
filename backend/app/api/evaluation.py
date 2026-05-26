import json
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.auth.permissions import get_current_user, require_role
from app.db.sqlite import get_connection

router = APIRouter(prefix="/api/v1/evaluation", tags=["evaluation"])


class EvalDataCreate(BaseModel):
    kb_id: int
    question: str
    reference_answer: str = ""
    relevant_doc_ids: list[int] = []


@router.post("/dataset")
def add_eval_data(req: EvalDataCreate, user: dict = Depends(require_role("admin", "editor"))):
    conn = get_connection()
    conn.execute(
        "INSERT INTO eval_dataset (kb_id, question, reference_answer, relevant_doc_ids, created_by) VALUES (?, ?, ?, ?, ?)",
        (req.kb_id, req.question, req.reference_answer, json.dumps(req.relevant_doc_ids), user["id"]),
    )
    conn.commit()
    conn.close()
    return {"code": 0, "data": None, "message": "Eval data added"}


@router.get("/dataset/{kb_id}")
def list_eval_data(kb_id: int, user: dict = Depends(get_current_user)):
    conn = get_connection()
    rows = conn.execute("SELECT * FROM eval_dataset WHERE kb_id=? ORDER BY created_at DESC", (kb_id,)).fetchall()
    conn.close()
    return {"code": 0, "data": [dict(r) for r in rows], "message": "ok"}


@router.post("/run/{kb_id}")
async def run_evaluation(kb_id: int, user: dict = Depends(require_role("admin", "editor"))):
    from app.evaluation.evaluator import Evaluator
    evaluator = Evaluator()
    report = await evaluator.run_full_eval(kb_id)
    return {"code": 0, "data": report, "message": "Evaluation complete"}


@router.get("/report/{kb_id}")
def get_report(kb_id: int, user: dict = Depends(get_current_user)):
    return {"code": 0, "data": {"message": "Run POST /run/{kb_id} to generate"}, "message": "ok"}
