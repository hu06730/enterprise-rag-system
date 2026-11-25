"""
两个测试分工：
- test_rerank_success  : 证明「重排真的参与了流程，而且改变了进入 prompt 的顺序」
- test_rerank_fallback : 证明「重排挂了，系统降级回 RRF 结果，且不崩」

全部依赖 tests/fakes.py 里的假对象：
不连 Milvus、不调 LLM、不加载 BGE 重排模型，整个文件 1 秒内跑完。
"""

import logging

from ragkb.interfaces.base_reranker import BaseReranker
from ragkb.pipeline.rag_pipeline import RAGPipeline
from ragkb.tests.fakes import (
    FakeParser,
    FakeSplitter,
    FakeVectorStore,
    FakeEmbedding,
    FakeBM25Retriever,
    FakeReranker,
    RecordingGenerator,
)


def build_pipeline(reranker, generator) -> RAGPipeline:
    """把一个全假对象的 Pipeline 装起来：7 个依赖，7 个假货。"""
    return RAGPipeline(
        parser=FakeParser(),
        splitter=FakeSplitter(),
        vector_store=FakeVectorStore(),
        generator=generator,
        embedding_model=FakeEmbedding(),
        bm25_retriever=FakeBM25Retriever(),
        reranker=reranker,
    )


def first_block(prompt: str) -> str:
    """截出 prompt 里编号 [1] 的那段文档。

    注意：build_prompt 的「要求 3」里也出现了 [1]、[2] 字样，
    所以不能用 prompt.index("[1]")，必须用文档块独有的标记 "[1] 来源："。
    """
    start = prompt.index("[1] 来源：")
    return prompt[start:].split("[2] 来源：")[0]


class ErrorReranker(BaseReranker):
    """一定会炸的重排器：模拟模型没下载成功、显存不够、依赖没装。"""

    def rerank(self, query, docs, top_k=5):
        raise ValueError("模型挂了")


def test_rerank_success():
    reranker = FakeReranker()
    generator = RecordingGenerator()
    pipeline = build_pipeline(reranker, generator)

    answer = pipeline.query("测试问题", top_k=2)

    # 断言 1：重排真的被调用了（而不是被 `if self.reranker` 跳过）
    assert reranker.called is True

    # 断言 2：重排接在 RRF 之后 —— 它收到的每个候选都带 rrf_score
    assert len(reranker.received) == 2
    assert all("rrf_score" in doc.metadata for doc in reranker.received)

    # 断言 3：进 prompt 的第 [1] 段，正是重排分数最高的那段
    highest = max(reranker.assigned, key=reranker.assigned.get)
    assert highest in first_block(generator.last_prompt)

    # 断言 4：只把 top_k=2 条写进 prompt（长度控制生效），且真的产出了答案
    assert generator.last_prompt.count("来源：") == 2
    assert answer == "假答案"


def test_rerank_fallback(caplog):
    generator = RecordingGenerator()
    pipeline = build_pipeline(ErrorReranker(), generator)

    # 断言 1：重排抛异常了，但整条链路不崩，照常返回答案
    #          —— 这就是「降级」的全部意义
    # 注意：降级告警从 print 改成了 logging（MCP 的 stdio 模式下 print 会污染协议），
    #       所以这里用 caplog 捕获，而不是 capsys。
    with caplog.at_level(logging.WARNING, logger="ragkb.pipeline.rag_pipeline"):
        answer = pipeline.query("测试问题", top_k=2)
    assert answer == "假答案"

    # 断言 2：确实走进了 except 分支（记了 Warning），不是碰巧没调用重排
    assert "重排失败" in caplog.text

    # 断言 3：降级后用的是 RRF 的候选，prompt 不是空的
    prompt = generator.last_prompt
    assert prompt is not None
    assert "假检索结果" in prompt or "假 BM25 结果" in prompt

    # 断言 4：RRF 的两条候选都进了 prompt
    #          （说明降级走的是 fused_docs[:top_k]，不是返回空列表）
    assert prompt.count("来源：") == 2
