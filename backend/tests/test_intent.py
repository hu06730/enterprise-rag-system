def setup_module():
    from app.db.sqlite import init_db
    init_db()


def test_intent_classifier_factual_lookup():
    from app.core.intent import classify_intent
    result = classify_intent("年假是多少天")
    assert result["intent"] == "factual_lookup"
    assert result["bm25_weight"] > result["vector_weight"]


def test_intent_classifier_conceptual():
    from app.core.intent import classify_intent
    result = classify_intent("为什么矩阵乘法是这样定义的")
    assert result["intent"] == "conceptual"
    assert result["vector_weight"] > result["bm25_weight"]


def test_intent_classifier_default():
    from app.core.intent import classify_intent
    result = classify_intent("你好啊今天天气")
    assert result["intent"] == "default"
    assert result["vector_weight"] == 0.5
    assert result["bm25_weight"] == 0.5
