import json
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.auth.jwt import hash_password, verify_password, create_token
from app.auth.permissions import get_current_user, require_role
from app.db.sqlite import get_connection

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    role: str = "viewer"
    departments: list[str] = []


@router.post("/login")
def login(req: LoginRequest):
    conn = get_connection()
    user = conn.execute("SELECT * FROM users WHERE username=?", (req.username,)).fetchone()
    conn.close()
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_token(user["id"], user["role"])
    return {
        "code": 0,
        "data": {"token": token, "username": user["username"], "role": user["role"]},
        "message": "ok",
    }


@router.post("/register")
def register(req: RegisterRequest, user: dict = Depends(require_role("admin"))):
    conn = get_connection()
    existing = conn.execute("SELECT id FROM users WHERE username=?", (req.username,)).fetchone()
    if existing:
        conn.close()
        raise HTTPException(status_code=400, detail="Username already exists")

    conn.execute(
        "INSERT INTO users (username, password_hash, role, departments) VALUES (?, ?, ?, ?)",
        (req.username, hash_password(req.password), req.role, json.dumps(req.departments)),
    )
    conn.commit()
    conn.close()
    return {"code": 0, "data": None, "message": "User created"}


@router.get("/me")
def me(user: dict = Depends(get_current_user)):
    return {"code": 0, "data": user, "message": "ok"}
