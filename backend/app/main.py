from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.db.sqlite import init_db
from app.db.vec_store import init_vec


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    init_vec()
    yield


app = FastAPI(title="Enterprise RAG API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}
