import json
from app.db.sqlite import get_connection
from app.db.vec_store import insert_chunk_vec_and_meta
from app.ingestion.parsers import get_parser
from app.ingestion.splitter import get_splitter
from app.ingestion.embedder import OpenAIEmbedder


async def run_ingestion(doc_id: int):
    conn = get_connection()
    doc = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
    if not doc:
        conn.close()
        raise ValueError(f"Document {doc_id} not found")

    kb_id = doc["kb_id"]
    config = conn.execute("SELECT * FROM kb_config WHERE kb_id=?", (kb_id,)).fetchone()
    if not config:
        conn.close()
        raise ValueError(f"KB config for kb_id={kb_id} not found")

    conn.execute("UPDATE documents SET status='processing' WHERE id=?", (doc_id,))
    conn.commit()
    conn.close()

    try:
        parser = get_parser(doc["file_path"])
        parsed = parser.parse(doc["file_path"])

        splitter = get_splitter(
            strategy=config["chunk_strategy"],
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
        )
        chunks = splitter.split(parsed.text)

        embedder = OpenAIEmbedder()
        embeddings = await embedder.embed(chunks)

        access_level = doc["access_level"] or "internal"
        tags = doc["metadata_tags"] or "[]"

        for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            insert_chunk_vec_and_meta(
                embedding=embedding,
                doc_id=doc_id,
                kb_id=kb_id,
                chunk_index=i,
                text=chunk_text,
                title=parsed.title,
                access_level=access_level,
                tags=tags,
            )

        conn = get_connection()
        conn.execute("UPDATE documents SET status='completed' WHERE id=?", (doc_id,))
        conn.commit()
        conn.close()
    except Exception as e:
        conn = get_connection()
        conn.execute("UPDATE documents SET status='failed' WHERE id=?", (doc_id,))
        conn.commit()
        conn.close()
        raise e
