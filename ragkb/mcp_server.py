"""把 RAG 知识库暴露成 MCP 工具，供 Claude Desktop / Cursor 等 Agent 调用。

启动：python -m ragkb.mcp_server

⚠️ stdio 模式下 stdout 是协议通道：本模块及其依赖的库代码**不得 print**，
   日志必须走 stderr（下面的 handler 已显式指定 sys.stderr）。

API 说明（mcp 2.x）：
    mcp 2.0 起 FastMCP 改名为 MCPServer，导入路径也变了：
        from mcp.server.mcpserver import MCPServer
    装饰器 @server.tool()、server.run() 的用法与 1.x 基本一致。
"""

import logging
import sys

from mcp.server.mcpserver import MCPServer

from ragkb.service import answer_structured

# ---------------------------------------------------------------- 日志（stderr）

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)      # 必须 stderr，不能 stdout
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

# ---------------------------------------------------------------- 服务实例

server = MCPServer("ragkb", description="企业知识库检索与问答（RAG）")

_pipeline = None


def get_pipeline():
    """懒加载单例：第一次调用工具时才加载模型（约 3GB / 40 秒）。

    为什么懒加载 + 函数内 import：
      - 模块导入时就加载模型，会让 MCP 客户端握手超时
      - 在函数内 import ragkb.main，避免 "import 本模块" 就拖起 torch
    """
    global _pipeline
    if _pipeline is None:
        from ragkb.main import build_pipeline

        logger.info("首次调用，正在加载 pipeline（约 40 秒）…")
        _pipeline = build_pipeline()
        logger.info("pipeline 就绪")
    return _pipeline


# ---------------------------------------------------------------- 工具定义

@server.tool()
def search_knowledge_base(question: str, top_k: int = 5) -> dict:
    """在企业知识库中检索并回答问题。

    适用：企业文档、技术手册、制度流程、编程教程类问题。
    不适用：实时数据、数学计算、需要访问外部网络的查询。

    参数:
        question: 自然语言问题（中文即可）
        top_k: 候选片段数，默认 5；多跳问题建议 8

    返回字段:
        answer     : 回答正文
        citations  : 引用片段列表（source / chunk_index / score / snippet）
        confidence : 0~1 可信度（sigmoid 后的 rerank 分）；低于 0.2 表示资料不足
        refused    : 是否拒答
        refused_by : "retrieval"（检索没找到）| "generation"（找到但答不了）| null

    注意: refused 为 true 时不要把 answer 当作事实依据，
          应改用其他工具，或向用户说明知识库中没有相关资料。
    """
    return answer_structured(get_pipeline(), question, top_k)


if __name__ == "__main__":
    server.run()          # 默认 stdio 传输
