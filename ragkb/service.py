"""服务层：把 pipeline 的原始输出整理成「给 Agent 消费」的结构化结果。

这一层是 HTTP（app.py）和 MCP（mcp_server.py）**唯一共用**的入口，
目的是让两个接入面返回完全一致的 schema，避免各写一套逻辑。

为什么 refused_by 要在这里算：
    pipeline.answer() 只返回 (answer, docs, refused) —— 它只知道自己"拒没拒"，
    但"谁拒的"是**解释层**的信息，属于服务层职责，不该污染检索/生成逻辑。
"""

import math
from pathlib import Path

from ragkb.pipeline.rag_pipeline import REFUSAL_TEXT

SNIPPET_LEN = 80          # 引用片段只截前 80 字，够回答"出自哪一段"
_SIGMOID_CLAMP = 60.0     # 防止 math.exp 溢出（logit 极端值的保护）


def _sigmoid(x: float) -> float:
    """logit → 0~1。bge-reranker 输出的是 logit，可正可负，必须转换后才能叫"置信度"。"""
    x = max(-_SIGMOID_CLAMP, min(_SIGMOID_CLAMP, float(x)))
    return 1.0 / (1.0 + math.exp(-x))


def derive_refused_by(answer: str, refused: bool) -> str | None:
    """判断这次拒答发生在哪一层。

    - "retrieval" : pipeline 在检索层就拒了（没调 LLM，零成本）
    - "generation": 调了 LLM，模型按 prompt 要求自己说"资料不足"
    - None        : 正常回答
    """
    if refused:
        return "retrieval"
    # 用前缀匹配而不是 ==，因为模型常在标准句后追加解释
    if answer.strip().startswith(REFUSAL_TEXT.rstrip("。")):
        return "generation"
    return None


def build_citations(docs) -> list[dict]:
    """把 Document 列表转成 citations。

    注意：Document 不是字典，不能用 {**doc} 展开；要显式取 metadata 和 page_content。
    不返回字符级 span，因为模型生成的是自然语言，无法可靠对齐到原文偏移量。
    """
    citations = []
    for d in docs:
        raw_source = str(d.metadata.get("source", "unknown"))
        citations.append({
            "source": Path(raw_source).name,          # 只留文件名，避免泄露绝对路径
            "chunk_index": d.metadata.get("chunk_index"),
            # 重排分数优先；没开重排时退回 Milvus 相似度
            "score": d.metadata.get("rerank_score", d.metadata.get("score")),
            "snippet": d.page_content.replace("\n", " ")[:SNIPPET_LEN],
        })
    return citations


def compute_confidence(docs) -> float | None:
    """置信度 = sigmoid(最高 rerank 分)。

    - 没有文档 → None（检索层拒答时应为 None，不要给 0 混淆）
    - 没有 rerank_score（未开重排）→ None，**不要用 rrf 分数硬算**，量纲不同
    """
    if not docs:
        return None
    score = docs[0].metadata.get("rerank_score")
    if score is None:
        return None
    return round(_sigmoid(score), 4)


def answer_structured(pipeline, question: str, top_k: int = 5) -> dict:
    """检索 + 生成 + 解释，返回给 Agent 消费的字典。

    返回:
        {
          "answer": str,
          "citations": [{"source", "chunk_index", "score", "snippet"}],
          "confidence": float | None,
          "refused": bool,
          "refused_by": "retrieval" | "generation" | None,
        }
    """
    answer, docs, refused = pipeline.answer(question, top_k)
    refused_by = derive_refused_by(answer, refused)

    return {
        "answer": answer,
        "citations": build_citations(docs),
        "confidence": compute_confidence(docs),
        "refused": refused_by is not None,
        "refused_by": refused_by,
    }
