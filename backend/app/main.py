from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.db.sqlite import init_db
from app.db.vec_store import init_vec
from app.db.seed import seed_intent_rules, seed_admin_user
from app.core.bm25_cache import bm25_cache
from app.api.auth import router as auth_router
from app.api.knowledge_bases import router as kb_router
from app.api.documents import router as doc_router
from app.api.chat import router as chat_router
from app.api.audit import router as audit_router
from app.api.evaluation import router as eval_router
from app.api.admin import router as admin_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    init_vec()
    seed_intent_rules()
    seed_admin_user()
    bm25_cache.warm_up()  # 预热所有知识库的 BM25 索引
    yield


app = FastAPI(title="Enterprise RAG API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(kb_router)
app.include_router(doc_router)
app.include_router(chat_router)
app.include_router(audit_router)
app.include_router(eval_router)
app.include_router(admin_router)


@app.get("/health")
def health():
    return {"status": "ok"}
