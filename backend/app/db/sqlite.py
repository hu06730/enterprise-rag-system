import sqlite3
import os
from app.config import settings


def get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(settings.sqlite_path), exist_ok=True)
    conn = sqlite3.connect(settings.sqlite_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'viewer',
            departments TEXT DEFAULT '[]',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS knowledge_bases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT DEFAULT '',
            kb_type TEXT NOT NULL DEFAULT 'employee',
            access_level TEXT NOT NULL DEFAULT 'internal',
            allowed_departments TEXT DEFAULT '[]',
            allowed_users TEXT DEFAULT '[]',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS kb_config (
            kb_id INTEGER PRIMARY KEY,
            chunk_strategy TEXT DEFAULT 'recursive',
            chunk_size INTEGER DEFAULT 512,
            chunk_overlap INTEGER DEFAULT 50,
            retrieval_mode TEXT DEFAULT 'hybrid',
            top_k INTEGER DEFAULT 10,
            min_score REAL DEFAULT 0.0,
            rerank_enabled INTEGER DEFAULT 1,
            vector_weight REAL DEFAULT 0.5,
            bm25_weight REAL DEFAULT 0.5,
            prompt_template TEXT DEFAULT '',
            FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kb_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_type TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            access_level TEXT,
            metadata_tags TEXT DEFAULT '[]',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS chunks_meta (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER NOT NULL,
            kb_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            title TEXT DEFAULT '',
            text TEXT NOT NULL,
            department TEXT,
            access_level TEXT DEFAULT 'internal',
            tags TEXT DEFAULT '[]',
            FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE,
            FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            kb_id INTEGER,
            query_text TEXT NOT NULL,
            answer_text TEXT,
            sources TEXT DEFAULT '[]',
            feedback INTEGER,
            feedback_text TEXT,
            trace_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS eval_dataset (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kb_id INTEGER NOT NULL,
            question TEXT NOT NULL,
            reference_answer TEXT,
            relevant_doc_ids TEXT DEFAULT '[]',
            created_by INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS intent_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            intent_name TEXT NOT NULL,
            keywords TEXT NOT NULL,
            vector_weight REAL NOT NULL DEFAULT 0.5,
            bm25_weight REAL NOT NULL DEFAULT 0.5,
            priority INTEGER DEFAULT 0
        );
    """)
    conn.commit()
    conn.close()
