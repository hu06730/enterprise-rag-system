from app.db.sqlite import get_connection
from app.core.intent import DEFAULT_RULES


def seed_intent_rules():
    conn = get_connection()
    count = conn.execute("SELECT COUNT(*) FROM intent_rules").fetchone()[0]
    if count == 0:
        for rule in DEFAULT_RULES:
            conn.execute(
                "INSERT INTO intent_rules (intent_name, keywords, vector_weight, bm25_weight, priority) VALUES (?, ?, ?, ?, ?)",
                (rule["intent_name"], rule["keywords"], rule["vector_weight"], rule["bm25_weight"], rule["priority"]),
            )
        conn.commit()
    conn.close()


def seed_admin_user():
    conn = get_connection()
    count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    if count == 0:
        from app.auth.jwt import hash_password
        conn.execute(
            "INSERT INTO users (username, password_hash, role, departments) VALUES (?, ?, ?, ?)",
            ("admin", hash_password("admin123"), "admin", "[]"),
        )
        conn.commit()
    conn.close()
