import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.mark.asyncio
async def test_full_flow_login_create_kb_upload_ask():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/v1/auth/login", json={"username": "admin", "password": "admin123"})
        assert resp.status_code == 200
        token = resp.json()["data"]["token"]
        headers = {"Authorization": f"Bearer {token}"}

        resp = await client.post("/api/v1/knowledge-bases/", json={
            "name": "Test KB", "description": "Integration test", "kb_type": "employee",
            "chunk_strategy": "recursive", "chunk_size": 200, "chunk_overlap": 20,
        }, headers=headers)
        assert resp.status_code == 200
        kb_id = resp.json()["data"]["id"]

        resp = await client.get("/api/v1/knowledge-bases/", headers=headers)
        assert resp.status_code == 200
        assert len(resp.json()["data"]) >= 1

        import io
        resp = await client.post(
            "/api/v1/documents/upload",
            data={"kb_id": str(kb_id)},
            files={"file": ("test.txt", io.BytesIO("年假每年15天。\n申请年假需提前3天在OA提交。".encode("utf-8")))},
            headers=headers,
        )
        assert resp.status_code == 200
        doc_id = resp.json()["data"]["id"]

        import asyncio
        await asyncio.sleep(5)

        resp = await client.get(f"/api/v1/documents/{doc_id}/status", headers=headers)
        assert resp.json()["data"]["status"] == "completed"

        resp = await client.post("/api/v1/chat/ask", json={
            "kb_id": kb_id, "query": "年假有多少天",
        }, headers=headers)
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["answer"] != ""
        assert len(data["sources"]) > 0
        assert data["trace_id"] is not None

        resp = await client.get("/api/v1/admin/stats", headers=headers)
        assert resp.status_code == 200
        stats = resp.json()["data"]
        assert stats["knowledge_bases"] >= 1
        assert stats["documents"] >= 1
        assert stats["total_queries"] >= 1
