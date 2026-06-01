import json
import sqlite3
import sqlite_vec
from app.db.sqlite import get_connection
from app.config import settings


def _get_vec_connection() -> sqlite3.Connection:
    """Return a connection with sqlite_vec extension loaded."""
    conn = get_connection()
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn
    except Exception:
        conn.close()
        raise


def init_vec():
    conn = _get_vec_connection()
    try:
        dim = settings.EMBEDDING_DIMENSIONS
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                embedding float[{dim}]
            )
        """)
        conn.commit()
    finally:
        conn.close()


def insert_chunk_vec_and_meta(
    embedding: list[float],
    doc_id: int,
    kb_id: int,
    chunk_index: int,
    text: str,
    title: str = "",
    department: str | None = None,
    access_level: str = "internal",
    tags: str = "[]",
) -> int:
    conn = _get_vec_connection()
    try:
        cur = conn.execute(
            """INSERT INTO chunks_meta (doc_id, kb_id, chunk_index, title, text, department, access_level, tags)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (doc_id, kb_id, chunk_index, title, text, department, access_level, tags),
        )
        rowid = cur.lastrowid
        vec_json = json.dumps(embedding)
        conn.execute(
            "INSERT INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
            (rowid, vec_json),
        )
        conn.commit()
        return rowid
    finally:
        conn.close()


def delete_chunks_by_doc(doc_id: int):
    conn = _get_vec_connection()
    try:
        conn.execute(
            "DELETE FROM chunks_vec WHERE rowid IN (SELECT id FROM chunks_meta WHERE doc_id=?)",
            (doc_id,),
        )
        conn.execute("DELETE FROM chunks_meta WHERE doc_id=?", (doc_id,))
        conn.commit()
    finally:
        conn.close()


def delete_chunks_by_kb(kb_id: int):
    conn = _get_vec_connection()
    try:
        conn.execute(
            "DELETE FROM chunks_vec WHERE rowid IN (SELECT id FROM chunks_meta WHERE kb_id=?)",
            (kb_id,),
        )
        conn.execute("DELETE FROM chunks_meta WHERE kb_id=?", (kb_id,))
        conn.commit()
    finally:
        conn.close()


def search_similar(
    query_embedding: list[float],
    kb_id: int,
    access_levels: list[str],
    departments: list[str] | None = None,
    limit: int = 20,
    min_score: float = 0.0,
) -> list[dict]:
    conn = _get_vec_connection()
    try:
        vec_json = json.dumps(query_embedding)

        if departments:
            dept_placeholders = ",".join("?" for _ in departments)
            dept_clause = f"AND (m.department IS NULL OR m.department IN ({dept_placeholders}))"
        else:
            dept_clause = ""

        access_placeholders = ",".join("?" for _ in access_levels)

        query = f"""
            SELECT m.id, m.doc_id, m.kb_id, m.chunk_index, m.title, m.text,
                   m.department, m.access_level, m.tags,
                   vec_distance_cosine(v.embedding, ?) as distance
            FROM chunks_vec v
            JOIN chunks_meta m ON v.rowid = m.id
            WHERE m.kb_id = ?
              AND m.access_level IN ({access_placeholders})
              {dept_clause}
              AND vec_distance_cosine(v.embedding, ?) <= ?
            ORDER BY distance
            LIMIT ?
        """

        max_distance = 2.0 if min_score <= 0 else (1.0 - min_score) * 2
        params = [vec_json, kb_id] + access_levels + (departments or []) + [vec_json, max_distance, limit]
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
