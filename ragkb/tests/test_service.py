"""服务层单测：只测纯函数与结构化逻辑，不连 Milvus、不调 LLM。"""

from langchain_core.documents import Document

from ragkb.pipeline.rag_pipeline import REFUSAL_TEXT
from ragkb.service import (
    answer_structured,
    build_citations,
    compute_confidence,
    derive_refused_by,
)


# ---------------------------------------------------------------- refused_by

def test_refused_by_retrieval():
    """pipeline 返回 refused=True → 检索层拒答。"""
    assert derive_refused_by(REFUSAL_TEXT, True) == "retrieval"


def test_refused_by_generation():
    """调了 LLM 但模型自己拒答 → 生成层。"""
    assert derive_refused_by(REFUSAL_TEXT, False) == "generation"


def test_refused_by_generation_with_extra_text():
    """模型常在标准句后追加解释，前缀匹配要能识别。"""
    answer = REFUSAL_TEXT + "现有文档片段 [1][2] 未包含该信息。"
    assert derive_refused_by(answer, False) == "generation"


def test_refused_by_none_for_normal_answer():
    assert derive_refused_by("用 append() 入栈，pop() 出栈。", False) is None


# ---------------------------------------------------------------- confidence

def test_confidence_is_sigmoid_of_rerank_score():
    docs = [Document(page_content="x", metadata={"rerank_score": 0.0})]
    assert compute_confidence(docs) == 0.5        # sigmoid(0) = 0.5

    docs = [Document(page_content="x", metadata={"rerank_score": 4.0})]
    assert compute_confidence(docs) > 0.98


def test_confidence_none_without_rerank():
    """没开重排时返回 None，不能用 rrf 分数硬算。"""
    docs = [Document(page_content="x", metadata={"rrf_score": 0.016, "score": 0.7})]
    assert compute_confidence(docs) is None


def test_confidence_none_when_no_docs():
    assert compute_confidence([]) is None


# ---------------------------------------------------------------- citations

def test_build_citations_uses_basename_and_snippet():
    docs = [Document(
        page_content="列表方法使得将列表作为栈来使用非常容易\n第二行",
        metadata={"source": "C:\\Users\\x\\corpus\\python_05_数据结构.txt",
                  "chunk_index": 5, "rerank_score": 5.86},
    )]
    cits = build_citations(docs)

    assert len(cits) == 1
    assert cits[0]["source"] == "python_05_数据结构.txt"   # 只留文件名
    assert cits[0]["chunk_index"] == 5
    assert cits[0]["score"] == 5.86
    assert "\n" not in cits[0]["snippet"]                  # 换行被压平
    assert cits[0]["snippet"]                               # 非空


# ---------------------------------------------------------------- 组合

class _FakePipeline:
    def __init__(self, answer, docs, refused):
        self._answer, self._docs, self._refused = answer, docs, refused
        self.called_with = None

    def answer(self, question, top_k=5):
        self.called_with = (question, top_k)
        return self._answer, self._docs, self._refused


def test_answer_structured_full_payload():
    docs = [Document(page_content="用 append() 入栈", metadata={
        "source": "python_05_数据结构.txt", "chunk_index": 5, "rerank_score": 5.0})]
    p = _FakePipeline("答案正文", docs, False)

    out = answer_structured(p, "用列表实现堆栈", top_k=3)

    assert p.called_with == ("用列表实现堆栈", 3)      # top_k 透传
    assert out["answer"] == "答案正文"
    assert out["refused"] is False
    assert out["refused_by"] is None
    assert 0 < out["confidence"] <= 1
    assert len(out["citations"]) == 1


def test_answer_structured_retrieval_refusal_has_no_citations():
    p = _FakePipeline(REFUSAL_TEXT, [], True)
    out = answer_structured(p, "Flask 和 Django 哪个好")

    assert out["refused"] is True
    assert out["refused_by"] == "retrieval"
    assert out["citations"] == []
    assert out["confidence"] is None
