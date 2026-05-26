import pytest
from pathlib import Path


@pytest.mark.asyncio
async def test_pipeline_processes_txt_and_stores_chunks():
    from app.db.sqlite import init_db, get_connection
    from app.db.vec_store import init_vec
    from app.ingestion.pipeline import run_ingestion

    init_db()
    init_vec()

    conn = get_connection()
    conn.execute(
        "INSERT INTO knowledge_bases (id, name, kb_type) VALUES (1, 'test', 'employee')"
    )
    conn.execute(
        "INSERT INTO kb_config (kb_id, chunk_strategy, chunk_size, chunk_overlap) VALUES (1, 'recursive', 200, 20)"
    )
    conn.execute(
        "INSERT INTO documents (id, kb_id, filename, file_path, file_type, status) VALUES (1, 1, 'test.txt', ?, 'txt', 'pending')",
        (str(Path(__file__).parent / "fixtures" / "ingest_sample.txt"),),
    )
    conn.commit()
    conn.close()

    sample = Path(__file__).parent / "fixtures" / "ingest_sample.txt"
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_text("This is test content for the ingestion pipeline.\nIt has two sentences.", encoding="utf-8")

    await run_ingestion(doc_id=1)

    conn = get_connection()
    count = conn.execute("SELECT COUNT(*) FROM chunks_meta WHERE doc_id=1").fetchone()[0]
    conn.close()
    assert count >= 1
