from langchain_core.documents import Document
from ragkb.eval import snippet_hit, first_hit_rank

def test_hit_true():
    docs = [Document(page_content="毛泽东思想是马克思主义中国化的成果")]
    assert snippet_hit(docs, ["马克思主义中国化"]) is True

def test_hit_false():
    docs = [Document(page_content="无关内容")]
    assert snippet_hit(docs, ["马克思主义中国化"]) is False

def test_rank():
    docs = [Document(page_content="A"), Document(page_content="B 马克思主义中国化"), Document(page_content="C")]
    assert first_hit_rank(docs, ["马克思主义中国化"]) == 2

def _row(i, found, total, typ="multihop"):
    return {"id": i, "type": typ, "question": f"q{i}", "answerable": True,
            "rank": 1, "hit": True, "gold_found": found, "gold_total": total,
            "top_score": 1.0, "score_kind": "rerank"}


def test_recall_macro_vs_micro():
    """锁定口径：macro 每题等权，micro 按片段加权。

    两题：(1/2) 和 (1/1)
      macro = (0.5 + 1.0) / 2 = 0.75
      micro = (1+1) / (2+1)      = 0.667
    """
    from ragkb.eval import summarize

    s = summarize([_row(1, 1, 2), _row(2, 1, 1)])
    assert s["recall_strict"] == 0.75      # 标准 IR 定义
    assert s["recall_micro"] == 0.667      # 片段池
    assert s["recall"] == 1.0              # HitRate：两题都至少命中 1 个
