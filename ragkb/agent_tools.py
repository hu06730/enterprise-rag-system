from langchain.tools import tool
from ragkb.pipeline.rag_pipeline import RAGPipeline


def make_rag_tool(pipeline: RAGPipeline):
    @tool
    def search_knowledge_base(query: str) -> str:
        """搜索企业知识库并回答问题。输入一个完整的问题，返回基于知识库的答案。"""
        return pipeline.query(query)

    return search_knowledge_base