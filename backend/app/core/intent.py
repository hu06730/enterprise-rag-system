from app.db.sqlite import get_connection

DEFAULT_RULES = [
    {
        "intent_name": "conceptual",
        "keywords": "为什么 区别 原理 关系 作用 影响 意义 对比",
        "vector_weight": 0.7,
        "bm25_weight": 0.3,
        "priority": 10,
    },
    {
        "intent_name": "factual_lookup",
        "keywords": "是什么 什么是 多少 多少天 多少钱 定义 有哪些 哪几个",
        "vector_weight": 0.3,
        "bm25_weight": 0.7,
        "priority": 10,
    },
    {
        "intent_name": "procedural",
        "keywords": "怎么做 如何 怎么 流程 步骤 方法 怎样 如何操作 教程",
        "vector_weight": 0.5,
        "bm25_weight": 0.5,
        "priority": 10,
    },
    {
        "intent_name": "compliance",
        "keywords": "规定 标准 条例 政策 依据 合规 制度 规范 要求 必须",
        "vector_weight": 0.2,
        "bm25_weight": 0.8,
        "priority": 10,
    },
]


def load_rules() -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT intent_name, keywords, vector_weight, bm25_weight, priority FROM intent_rules ORDER BY priority DESC"
    ).fetchall()
    conn.close()
    if not rows:
        return DEFAULT_RULES
    return [dict(r) for r in rows]


def classify_intent(query: str) -> dict:
    rules = load_rules()
    for rule in rules:
        keywords = rule["keywords"].split()
        if any(kw in query for kw in keywords):
            return {
                "intent": rule["intent_name"],
                "vector_weight": rule["vector_weight"],
                "bm25_weight": rule["bm25_weight"],
            }
    return {"intent": "default", "vector_weight": 0.5, "bm25_weight": 0.5}
