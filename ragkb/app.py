"""FastAPI 服务层：把 RAGPipeline 暴露成 HTTP 接口。

给不熟悉后端的自己留三条备忘：

1. **路由**：`@app.post("/query")` 意思是"有人 POST /query 时，执行下面这个函数"。
2. **请求体**：函数参数声明成 `BaseModel` 子类，FastAPI 会自动把 JSON 解析成对象、
   并做字段校验（不合法直接返回 422，不用自己写 if）。
3. **生命周期（lifespan）**：服务启动时跑一次 yield 之前的代码，关闭时跑 yield 之后。

为什么**不在模块顶层** build_pipeline()：
    那样 `import ragkb.app` 就会连 Milvus、加载 3GB 模型。pytest 收集测试、
    IDE 静态检查、`uvicorn --reload` 都会触发 import → 又慢又容易崩。
    放进 lifespan，才保证"只有服务真正启动时才初始化"。
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from pathlib import Path

from ragkb.config.settings import ENCODING
from ragkb.pipeline.rag_pipeline import REFUSAL_TEXT
from ragkb.service import answer_structured

logger = logging.getLogger("ragkb.app")
# 为什么必须自己配 handler：Python root logger 默认级别是 WARNING，
# 不配的话 logger.info() 会被静默丢弃 —— 表现为"启动时卡住、没日志"，
# 实际是正在入库。这里显式指定 INFO 级别，并把日志格式对齐 uvicorn。
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

# 启动时如果 Milvus 还没就绪（docker compose 常见：app 比 milvus 快），
# 重试这么多次，每次间隔这么多秒。全部失败才让服务退出（fail fast）。
STARTUP_ATTEMPTS = 8
STARTUP_DELAY = 5.0


# ============================================================ 请求 / 响应模型
# 作用：定义 HTTP 接口的"数据格式合同"。FastAPI 会据此自动生成文档（/docs）。
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="用户问题")
    top_k: int = Field(5, ge=1, le=20, description="返回多少条证据片段")


class Citation(BaseModel):
    """一条引用片段。给 Agent 用：它能据此判断"答案有没有证据支持"。"""
    source: str
    chunk_index: int | None = None
    score: float | None = None
    snippet: str = Field(..., description="原文片段前 80 字，可回原文核对")


class QueryResponse(BaseModel):
    answer: str
    refused: bool          # 是否走了拒答（调用方可据此区分"答了"和"没资料"）
    refused_by: str | None = None  # "retrieval" | "generation" | None
    citations: list[Citation]
    confidence: float | None = None  # sigmoid(最高 rerank 分)；未开重排时为 None


class IngestRequest(BaseModel):
    files: list[str] | None = Field(None, description="不给则入库 corpus/ 下所有文件")


class IngestResponse(BaseModel):
    job_id: str
    status: str


class JobStatus(BaseModel):
    job_id: str
    status: str            # running / done / failed
    count: int | None = None
    error: str | None = None


# ============================================================ 生命周期
@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动时准备 pipeline（只做一次），关闭时清理。

    pipeline 里包含 Milvus 连接、embedding 模型、reranker 模型，
    是全进程共享的"重资源"，绝不能每个请求新建。
    """
    # 注意：这两个 import 必须放在 if 里面！
    # ragkb.main 会连带加载 sentence-transformers / torch（约 50 秒），
    # 如果放在外面，测试注入假 pipeline 时也照样要等 50 秒。
    if getattr(app.state, "pipeline", None) is None:
        from ragkb.config.corpus import corpus_files
        from ragkb.main import build_pipeline

        # 依赖没就绪时重试，避免 docker compose 里 app 比 milvus 先起来就崩
        pipeline = None
        for attempt in range(1, STARTUP_ATTEMPTS + 1):
            try:
                pipeline = build_pipeline()
                break
            except Exception as exc:
                logger.warning(
                    "初始化失败（第 %d/%d 次，Milvus 或模型可能未就绪）：%s",
                    attempt, STARTUP_ATTEMPTS, exc,
                )
                if attempt == STARTUP_ATTEMPTS:
                    logger.error("达到最大重试次数，服务退出（fail fast，交给编排系统重启）")
                    raise
                await asyncio.sleep(STARTUP_DELAY)

        # 向量在 Milvus 里（持久化），但 BM25 索引在内存里，进程重启就没了 → 必须重建
        chunks: list = []
        for f in corpus_files():
            chunks.extend(pipeline.splitter.split(pipeline.parser.load(f, ENCODING)))
        pipeline.bm25_retriever.build_index(chunks)

        app.state.pipeline = pipeline
        app.state.chunk_count = len(chunks)

        # 一键部署用：AUTO_INGEST_ON_START=1 时启动即入库（空库首次约 1~3 分钟）
        # 默认关闭，避免每次重启都重算 embedding（虽然幂等，但白等）
        if os.getenv("AUTO_INGEST_ON_START") == "1":
            existing = _collection_count(pipeline.vector_store)
            if existing == 0:
                logger.info("库为空，开始入库（约 1~3 分钟）…")
                n = pipeline.ingest(corpus_files(), ENCODING)
                logger.info("自动入库完成：%d 条", n)
            else:
                logger.info("库中已有 %s 条，跳过入库（重启不再白等）", existing)

    app.state.jobs = {}          # 入库任务表：job_id -> 状态
    yield
    app.state.pipeline = None    # 关闭时释放引用


def _collection_count(store) -> int | None:
    """读集合里的条数；拿不到就返回 None（宁可多入库一次，也不要静默跳过）。"""
    client = getattr(store, "client", None)
    name = getattr(store, "collection_name", None)
    if client is None or name is None:
        return None
    try:
        stats = client.get_collection_stats(name)
        return int(stats.get("row_count", 0))
    except Exception:
        return None


app = FastAPI(title="RAG 企业知识库服务", version="0.1.0", lifespan=lifespan)


# ============================================================ 接口
@app.get("/health")
def health() -> dict[str, Any]:
    """存活检查 + 依赖自检。Docker healthcheck / K8s readiness 都调这个。"""
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="服务尚未就绪")
    return {"status": "ok", "chunks": getattr(app.state, "chunk_count", 0)}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """问答：返回答案 + 引用片段 + 置信度 + 拒答来源。

    业务逻辑全部在 ragkb/service.py（HTTP 与 MCP 共用），这里只做 HTTP 包装。
    """
    try:
        result = answer_structured(app.state.pipeline, req.question, req.top_k)
    except Exception as exc:                       # 检索/生成失败 → 503，不吐 500 堆栈
        raise HTTPException(status_code=503, detail=f"检索或生成失败: {exc}") from exc

    # 检索层拒答时没有可用证据，citations 置空（避免调用方误用无关片段）
    if result["refused_by"] == "retrieval":
        result["citations"] = []
    return QueryResponse(**result)


@app.post("/ingest", response_model=IngestResponse, status_code=202)
def ingest(req: IngestRequest, background: BackgroundTasks) -> IngestResponse:
    """触发入库。

    为什么用后台任务：345 条要算 embedding，约 1~3 分钟。
    同步返回会让 HTTP 请求超时（网关默认 60s），所以：
        立刻返回 202 + job_id，真正干活放到后台，调用方轮询 /ingest/{job_id}。
    幂等性：文档 id = md5(source|chunk_index|content)，重复入库按主键覆盖。
    """
    pipeline = app.state.pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="服务尚未就绪")

    if req.files:
        files = list(req.files)
    else:
        from ragkb.config.corpus import corpus_files
        files = corpus_files()

    job_id = uuid.uuid4().hex[:8]
    app.state.jobs[job_id] = {"job_id": job_id, "status": "running", "count": None, "error": None}

    def _run_ingest() -> None:
        try:
            n = pipeline.ingest(files, ENCODING)
            app.state.jobs[job_id].update(status="done", count=n)
        except Exception as exc:                    # 后台任务异常不能抛给调用方，要记在任务状态里
            app.state.jobs[job_id].update(status="failed", error=str(exc))

    background.add_task(_run_ingest)
    return IngestResponse(job_id=job_id, status="running")


@app.get("/ingest/{job_id}", response_model=JobStatus)
def ingest_status(job_id: str) -> JobStatus:
    """查询入库任务进度。"""
    job = app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"没有这个任务: {job_id}")
    return JobStatus(**job)
