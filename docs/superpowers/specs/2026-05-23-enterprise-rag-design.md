# 企业知识库 RAG 系统 — 架构设计

## 概述

面向混合场景（员工知识库 + 客服对外 + 合规审计）的企业级 RAG 问答系统。核心原则：零外部服务、单一进程启动、所有关键组件基于接口抽象可替换。

| 维度 | 选择 |
|------|------|
| 场景 | 混合（员工知识库 + 客服 + 合规审计） |
| 技术栈 | Python 全栈（FastAPI） |
| 模型部署 | 先云端 API（OpenAI/Claude），后续可切本地 |
| 数据源 | 非结构化文档（PDF/Word/Markdown/TXT） |
| 交付形式 | REST API + 管理后台 + 用户端问答界面 |

---

## 基础设施

| 组件 | 选型 | 说明 |
|------|------|------|
| 向量存储 | SQLite + sqlite-vec | 向量存在同一 SQLite 文件，使用 vec0 虚拟表 + KNN |
| 元数据/配置/用户 | SQLite | 同上，all-in-one |
| 关键词检索 | rank_bm25 | 纯 Python，内存构建 BM25 索引 |
| 文件存储 | 本地文件夹 `data/uploads/` | 切 MinIO 仅需替换存储适配器 |
| 缓存 | diskcache | 磁盘持久化，重启不丢，线程安全 |
| 异步处理 | FastAPI BackgroundTasks | 同进程异步，无需消息队列 |
| 前端 | React | 独立部署 |

单文件 SQLite 承载：向量、元数据、配置、用户、审计日志、评估数据。一条命令启动。

---

## 数据处理层（Ingestion Pipeline）

### 同步链路

```
用户上传文档 → BackgroundTasks 异步执行
  ├─ 文档状态: processing → completed / failed
  ├─ 进度: GET /documents/{id}/status
  │
  ▼ 解析器（插件注册）
  ├─ PDF  → PyMuPDF
  ├─ Word → python-docx
  ├─ MD   → markdown-it-py（按 AST 标题节点拆）
  └─ TXT  → 按自然段
  │
  ▼ 分块器（策略模式，每个知识库可配置）
  ├─ recursive: 递归字符分割，按分隔符优先级切（\n\n → \n → 。→ . → 空格）
  ├─ markdown:  按标题层级切分，保留标题链作为上下文
  └─ semantic:  SentenceTransformer 计算相邻句子余弦相似度，找谷底作为分界点
      策略参数可配: chunk_size, overlap, min_chunk_size
  │
  ▼ 嵌入: text-embedding-3-small（OpenAI, 1536维），可替换
  │
  ▼ 写入
  └─ chunks_vec (vec0 虚拟表) + chunks_meta (元数据表)，通过 rowid 关联
```

### SQLite-vec 表结构

```sql
-- 向量表
CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding float[1536]);

-- 元数据表（与 chunks_vec 通过 rowid 一一对应）
CREATE TABLE chunks_meta (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL,
    kb_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    title TEXT,
    text TEXT NOT NULL,
    department TEXT,
    access_level TEXT DEFAULT 'internal',
    tags TEXT,            -- JSON array
    FOREIGN KEY (doc_id) REFERENCES documents(id),
    FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id)
);
```

### Metadata 作为权限锚点

每个 chunk 的 `department`、`access_level`、`tags` 字段在检索时直接作为 SQL WHERE 过滤条件注入，无需额外权限过滤服务。

---

## 检索引擎

### 场景差异

| 场景 | 检索倾向 | 默认向量权重 | 默认 BM25 权重 |
|------|----------|:-----------:|:-----------:|
| 员工知识库 | 召回优先 | 0.6 | 0.4 |
| 客服对外 | 精准优先 | 0.5 | 0.5 |
| 合规审计 | 精确匹配 | 0.3 | 0.7 |

### 检索流程

```
用户问题
  │
  ▼ 查询意图识别
  ├─ 关键词规则表判定意图类型:
  │   factual_lookup: "是什么""定义""多少" → BM25 偏高
  │   conceptual:     "为什么""区别""原理" → 向量偏高
  │   procedural:     "怎么做""流程""步骤" → 均衡
  │   compliance:     "规定""标准""依据"   → BM25 偏高
  │   规则表存 SQLite，兜底使用场景默认权重
  │
  ▼ 混合检索
  ├─ 向量路径: sqlite-vec KNN
  │   SELECT m.rowid, m.text, m.title, vec_distance_cosine(v.embedding, ?) as score
  │   FROM chunks_vec v JOIN chunks_meta m ON v.rowid = m.id
  │   WHERE m.kb_id = ? AND m.access_level IN (?) AND m.department IN (?)
  │   ORDER BY score LIMIT ?
  │
  ├─ BM25 路径: rank_bm25 在 chunks_meta.text 上构建倒排索引搜索
  │
  └─ RRF 融合（动态权重）:
      RRF = w_vec/(k + rank_vec) + w_bm25/(k + rank_bm25)   k=60
      → TopK 合并去重
  │
  ▼ 重排序（可选）
  └─ BGE-Reranker / Cohere Rerank 精排，过滤低分
  │
  ▼ 上下文压缩（可选）
  └─ LLM 判断 chunk 相关性，过滤无关 chunk
  │
  ▼ 输出 List[ChunkWithScore] → 送入生成引擎
```

### 场景配置

```sql
CREATE TABLE kb_config (
    kb_id INTEGER PRIMARY KEY,
    chunk_strategy TEXT DEFAULT 'recursive',    -- recursive | markdown | semantic
    chunk_size INTEGER DEFAULT 512,
    chunk_overlap INTEGER DEFAULT 50,
    retrieval_mode TEXT DEFAULT 'hybrid',       -- vector | keyword | hybrid
    top_k INTEGER DEFAULT 10,
    min_score REAL DEFAULT 0.0,
    rerank_enabled INTEGER DEFAULT 1,
    vector_weight REAL DEFAULT 0.5,
    bm25_weight REAL DEFAULT 0.5,
    FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id)
);
```

---

## 生成引擎

### 流程

```
检索结果 List[Chunk]
  │
  ▼ 上下文组装
  ├─ 按 score 降序排列
  ├─ 每个 chunk 标注来源: [文档标题 / 章节]
  └─ 控制总 token ≤ 模型 context × 60%
  │
  ▼ 场景 Prompt 模板
  ├─ employee:   "你是 XX 公司的内部助手。仅使用以下参考资料回答……"
  ├─ customer:   "你是客服助手。回答需友好、准确。不确定就说不知道……"
  └─ compliance: "你是合规助手。回答必须逐条引用原文，标注文档名称和段落……"
      模板存在 SQLite，管理后台可编辑
  │
  ▼ LLM 调用（接口抽象）
  ├─ 默认: OpenAI Chat Completion API
  ├─ 支持 SSE 流式输出
  └─ 可替换: Claude / 通义千问 / DeepSeek
  │
  ▼ 后处理
  ├─ 引用格式规范化
  ├─ 合规场景: 无来源标注 → 标记 warning
  └─ 敏感词过滤（可配）
  │
  ▼ 输出: { answer, sources, confidence, trace_id }
```

### LLM 接口

```python
class BaseLLM(ABC):
    @abstractmethod
    async def generate(self, messages: list[Message], **kwargs) -> GenerationResult: ...
    @abstractmethod
    async def generate_stream(self, messages: list[Message]) -> AsyncIterator[str]: ...
```

---

## 权限与多租户

### 模型

```
User: id, username, password_hash, role(admin/editor/viewer/api), departments(json)
  │
  ▼
KnowledgeBase: id, name, kb_type(employee/customer/compliance),
               access_level(public/internal/restricted),
               allowed_departments(json), allowed_users(json)
  │
  ▼
Document: id, kb_id, filename, access_level(继承或覆盖), metadata_tags
```

### 权限生效链路

```
1. JWT → user_id
2. 查用户 departments + role
3. 组装检索 WHERE:
   kb_id = ? AND access_level IN (user.accessible_levels)
   AND (department IS NULL OR department IN (user.departments))
4. 向量搜索天然只返回有权看的 chunks
5. 审计日志: (user_id, query, sources, feedback, timestamp)
```

权限前置到检索条件中，不做事后过滤。

---

## API 设计

```
/api/v1/
├── /auth
│   ├── POST /login
│   └── POST /register              (admin only)
│
├── /knowledge-bases
│   ├── GET  /                      列表
│   ├── POST /                      创建
│   ├── PUT  /{id}                  更新
│   └── DELETE /{id}                删除
│
├── /documents
│   ├── POST   /upload              上传 → BackgroundTask 异步处理
│   ├── GET    /                    列表 + 处理状态
│   ├── GET    /{id}/status         处理进度
│   ├── DELETE /{id}                删除文档及所有 chunks
│   └── PUT    /{id}/reprocess      重新解析
│
├── /chat
│   ├── POST /ask                   同步问答
│   └── POST /ask/stream            SSE 流式输出
│
├── /audit
│   ├── GET /logs                   查询记录（按时间/用户/知识库）
│   └── GET /logs/{id}              单条详情
│
├── /evaluation
│   ├── POST /run/{kb_id}           运行离线评估
│   └── GET  /report/{kb_id}        查看评估报告
│
└── /admin
    ├── GET  /config/{kb_id}        查看知识库配置
    ├── PUT  /config/{kb_id}        修改配置
    └── GET  /stats                 概览统计
```

### 统一响应格式

```json
// 成功
{"code": 0, "data": {...}, "message": "ok"}

// 问答
{"code": 0, "data": {
    "answer": "...",
    "sources": [{"doc_title": "...", "chunk_text": "...", "score": 0.92}],
    "confidence": 0.85,
    "trace_id": "uuid"
}}

// 错误
{"code": 40001, "data": null, "message": "知识库不存在"}
```

---

## 评估系统

### 离线评估

| 指标 | 说明 |
|------|------|
| Recall@K | 相关文档是否在 TopK 中 |
| MRR | 第一个相关文档排在第几位 |
| Faithfulness | 答案是否忠实于检索内容（LLM-as-Judge） |
| Relevance | 答案是否切题（LLM-as-Judge） |
| Correctness | 答案事实是否正确（LLM-as-Judge） |

### 评估数据集

```sql
CREATE TABLE eval_dataset (
    id INTEGER PRIMARY KEY,
    kb_id INTEGER NOT NULL,
    question TEXT NOT NULL,
    reference_answer TEXT,
    relevant_doc_ids TEXT,          -- JSON: [1, 5, 8]
    created_by INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 在线反馈

```
用户 👍/👎 + 可选文字反馈
  → 写入 audit_logs
  → 管理后台可视化统计
  → 低分回答排查 → 调整策略 → 重新评估
```

---

## 前端

### 用户端问答界面

- 知识库选择下拉
- 对话区域，每条 AI 回答标注来源片段（可点击展开）
- 支持多轮对话（携带历史上下文）
- SSE 流式打字效果
- 反馈按钮（👍/👎）

### 管理后台

- 知识库管理：创建/编辑，配置场景类型与权限
- 文档管理：拖拽上传，列表显示处理状态，支持删除/重处理
- 配置管理：分块策略参数、RRF 权重、Prompt 模板编辑
- 审计日志：按时间/用户/知识库筛选
- 评估报告：触发离线评估，查看指标变化趋势
- 统计概览：文档数、chunk 数、API 调用量、反馈分布

### 技术选型

React + TypeScript，前后端分离，独立部署。

---

## 项目目录结构

```
rag-system/
├── backend/
│   ├── app/
│   │   ├── main.py                      # FastAPI 入口
│   │   ├── config.py                    # 环境变量 + 数据库配置读取
│   │   │
│   │   ├── api/
│   │   │   ├── auth.py
│   │   │   ├── knowledge_bases.py
│   │   │   ├── documents.py
│   │   │   ├── chat.py
│   │   │   ├── audit.py
│   │   │   ├── evaluation.py
│   │   │   └── admin.py
│   │   │
│   │   ├── core/                        # RAG 核心逻辑
│   │   │   ├── interfaces.py            # 所有抽象基类
│   │   │   ├── intent.py                # 查询意图识别
│   │   │   ├── retriever.py             # HybridRetriever
│   │   │   ├── generator.py             # 生成引擎
│   │   │   ├── rewrite.py               # 查询重写
│   │   │   └── reranker.py              # 重排序
│   │   │
│   │   ├── ingestion/
│   │   │   ├── pipeline.py              # 编排 + BackgroundTasks
│   │   │   ├── parsers.py               # PDF/Word/MD/TXT
│   │   │   ├── splitter.py              # recursive / markdown / semantic
│   │   │   └── embedder.py
│   │   │
│   │   ├── llm/
│   │   │   ├── base.py
│   │   │   ├── openai_llm.py
│   │   │   └── claude_llm.py
│   │   │
│   │   ├── evaluation/
│   │   │   ├── evaluator.py
│   │   │   ├── metrics.py
│   │   │   ├── llm_judge.py
│   │   │   └── feedback.py
│   │   │
│   │   ├── models/                      # SQLAlchemy ORM
│   │   │   ├── user.py
│   │   │   ├── knowledge_base.py
│   │   │   ├── document.py
│   │   │   ├── chunk.py
│   │   │   ├── eval_dataset.py
│   │   │   └── audit_log.py
│   │   │
│   │   ├── services/                    # 业务逻辑层
│   │   │   ├── knowledge_base_service.py
│   │   │   ├── document_service.py
│   │   │   └── audit_service.py
│   │   │
│   │   ├── auth/
│   │   │   ├── jwt.py
│   │   │   └── permissions.py
│   │   │
│   │   └── db/
│   │       ├── sqlite.py                # 连接管理
│   │       └── vec_store.py             # sqlite-vec 封装
│   │
│   ├── tests/
│   │   ├── test_ingestion/
│   │   ├── test_retrieval/
│   │   ├── test_generation/
│   │   └── test_evaluation/
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── user-portal/                     # 问答界面 (React)
│   └── admin-portal/                    # 管理后台 (React)
│
├── data/
│   ├── uploads/                         # 原始文档
│   ├── cache/                           # diskcache 目录
│   └── rag.db                           # SQLite 数据库（向量+元数据+配置）
│
└── eval_datasets/                       # 评估问答对样本
    └── sample_qa.jsonl
```

---

## 核心接口抽象

```python
# 检索器
class BaseRetriever(ABC):
    @abstractmethod
    async def retrieve(self, query: str, ctx: RetrievalContext) -> list[Chunk]: ...

# 分块器
class BaseSplitter(ABC):
    @abstractmethod
    def split(self, text: str, **kwargs) -> list[str]: ...

# LLM
class BaseLLM(ABC):
    @abstractmethod
    async def generate(self, msgs: list[Message], **kw) -> GenerationResult: ...
    @abstractmethod
    async def generate_stream(self, msgs: list[Message]) -> AsyncIterator[str]: ...

# 嵌入
class BaseEmbedder(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

# 文档解析器
class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_path: str) -> ParsedDocument: ...
```

---

## 扩展路径

| 当前（起步） | 未来（有需求时） |
|-------------|------------------|
| SQLite + sqlite-vec | PostgreSQL + pgvector / Milvus 分布式 |
| rank_bm25 | Elasticsearch |
| diskcache | Redis |
| FastAPI BackgroundTasks | Celery + RabbitMQ |
| 本地文件夹 | MinIO / S3 |
| 同步处理 pipeline | 消息队列异步 pipeline |
| 关键词规则意图识别 | 轻量分类模型 |
| OpenAI API | Claude / 本地 DeepSeek / Qwen |
| 单实例 | 多实例 + 负载均衡 |

每次替换只影响适配器实现，接口及上层代码不变。

---

*版本: v1.0 | 日期: 2026-05-23 | 状态: 设计已完成，待评审*
