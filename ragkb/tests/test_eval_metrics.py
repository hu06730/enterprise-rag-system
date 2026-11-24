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