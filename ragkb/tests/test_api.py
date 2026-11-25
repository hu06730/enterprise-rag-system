"""HTTP 接口层测试：用假 pipeline，不连 Milvus、不调 LLM。

关键技巧：TestClient 进入上下文时会执行 lifespan，
但我们在进入之前先把假 pipeline 塞进 app.state，
lifespan 里检查到已有 pipeline 就跳过真实初始化。
"""

import time

from fastapi.testclient import TestClient
from langchain_core.documents import Document

from ragkb.app import app
from ragkb.pipeline.rag_pipeline import REFUSAL_TEXT


class FakePipeline:
    """只实现接口层用到的方法：answer / ingest。"""

    def __init__(self, refuse: bool = False):
        self.refuse = refuse
        self.ingested_files = None

    def answer(self, question, top_k=5):
        if self.refuse:
            return REFUSAL_TEXT, [], True
        docs = [
            Document(page_content="列表最后添加的元素最先被取出",
                     metadata={"source": "python_05_数据结构.txt", "chunk_index": 3, "rerank_score": 5.2}),
            Document(page_content="用 append() 入栈、pop() 出栈",
                     metadata={"source": "python_05_数据结构.txt", "chunk_index": 4, "rerank_score": 4.1}),
        ]
        return f"答案：{question}", docs[:top_k], False

    def ingest(self, files, encoding="utf-8"):
        self.ingested_files = list(files)
        return 345


def make_client(pipeline) -> TestClient:
    """把假 pipeline 预先放进 app.state，再返回 TestClient。"""
    app.state.pipeline = pipeline
    app.state.chunk_count = 345
    return TestClient(app)


def test_health_ok():
    with make_client(FakePipeline()) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_query_returns_answer_and_citations():
    with make_client(FakePipeline()) as client:
        resp = client.post("/query", json={"question": "用列表实现堆栈", "top_k": 2})

    assert resp.status_code == 200
    body = resp.json()
    assert body["refused"] is False
    assert body["answer"] == "答案：用列表实现堆栈"
    assert len(body["citations"]) == 2
    # 可溯源：必须带 source 字段
    assert body["citations"][0]["source"].endswith(".txt")
    assert body["citations"][0]["chunk_index"] == 3


def test_query_refused_when_no_context():
    with make_client(FakePipeline(refuse=True)) as client:
        resp = client.post("/query", json={"question": "Flask 和 Django 哪个好"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["refused"] is True
    assert body["answer"] == REFUSAL_TEXT
    assert body["citations"] == []


def test_query_rejects_empty_question():
    """pydantic 校验：空问题应返回 422，不是 500。"""
    with make_client(FakePipeline()) as client:
        resp = client.post("/query", json={"question": ""})
    assert resp.status_code == 422


def test_query_rejects_bad_top_k():
    with make_client(FakePipeline()) as client:
        resp = client.post("/query", json={"question": "x", "top_k": 999})
    assert resp.status_code == 422


def test_ingest_returns_job_and_polls_status():
    pipeline = FakePipeline()
    with make_client(pipeline) as client:
        resp = client.post("/ingest", json={"files": ["a.txt", "b.txt"]})
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        # BackgroundTasks 在 TestClient 里随响应同步执行完，这里可以直接查到结果
        for _ in range(20):
            status = client.get(f"/ingest/{job_id}").json()
            if status["status"] != "running":
                break
            time.sleep(0.05)

    assert status["status"] == "done"
    assert status["count"] == 345
    assert pipeline.ingested_files == ["a.txt", "b.txt"]


def test_ingest_status_unknown_job_404():
    with make_client(FakePipeline()) as client:
        resp = client.get("/ingest/not-a-real-id")
    assert resp.status_code == 404
