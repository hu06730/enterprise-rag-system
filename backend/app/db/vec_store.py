import sqlite3
import sqlite_vec
from app.db.sqlite import get_connection


def init_vec():
    conn = get_connection()
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
            embedding float[1536]
        )
    """)
    conn.commit()
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
    conn = get_connection()
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    cur = conn.execute(
        """INSERT INTO chunks_meta (doc_id, kb_id, chunk_index, title, text, department, access_level, tags)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (doc_id, kb_id, chunk_index, title, text, department, access_level, tags),
    )
    rowid = cur.lastrowid
    vec_json = "[" + ",".join(str(v) for v in embedding) + "]"
    conn.execute(
        "INSERT INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
        (rowid, vec_json),
    )
    conn.commit()
    conn.close()
    return rowid


def delete_chunks_by_doc(doc_id: int):
    conn = get_connection()
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute(
        "DELETE FROM chunks_vec WHERE rowid IN (SELECT id FROM chunks_meta WHERE doc_id=?)",
        (doc_id,),
    )
    conn.execute("DELETE FROM chunks_meta WHERE doc_id=?", (doc_id,))
    conn.commit()
    conn.close()


def delete_chunks_by_kb(kb_id: int):
    conn = get_connection()
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute(
        "DELETE FROM chunks_vec WHERE rowid IN (SELECT id FROM chunks_meta WHERE kb_id=?)",
        (kb_id,),
    )
    conn.execute("DELETE FROM chunks_meta WHERE kb_id=?", (kb_id,))
    conn.commit()
    conn.close()


def search_similar(
    query_embedding: list[float],
    kb_id: int,
    access_levels: list[str],
    departments: list[str] | None = None,
    limit: int = 20,
    min_score: float = 0.0,
) -> list[dict]:
    conn = get_connection()
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    vec_json = "[" + ",".join(str(v) for v in query_embedding) + "]"

    if departments:
        deps_str = ",".join(f"'{d}'" for d in departments)
        dept_clause = f"AND (m.department IS NULL OR m.department IN ({deps_str}))"
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

    params = [vec_json, kb_id] + access_levels + [vec_json, 2.0, limit]
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]
