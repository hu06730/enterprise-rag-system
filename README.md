# 企业知识库 RAG 系统

基于 RAG（检索增强生成）的企业级智能问答系统，支持多知识库管理、文档上传、流式对话、审计日志和评估系统。

## ✨ 功能特性

### 🤖 智能问答
- SSE 流式输出，打字机效果
- 混合检索：向量检索 + BM25 关键词检索 + RRF 融合
- 查询意图识别，动态调整检索权重
- 引用来源展示，支持反馈（点赞/踩）

### 📚 知识库管理
- 多知识库支持（员工/客服/合规）
- 可配置分块策略（递归/Markdown/语义）
- 可配置检索模式（混合/向量/关键词）
- 权限控制：部门级别访问控制

### 📄 文档处理
- 支持 PDF、Word、Markdown、TXT 格式
- 异步处理管道（解析 → 分块 → 向量化）
- 实时状态监控（等待/处理中/完成/失败）
- 支持重新处理和删除

### 📊 评估系统
- 测试数据集管理
- 自动评估：Recall@K、MRR 指标
- LLM-as-Judge：忠实度、相关性评分
- 评估报告生成

### 🔐 权限管理
- JWT 认证
- 三级角色：管理员 / 编辑者 / 查看者
- 基于角色的访问控制（RBAC）

## 🛠️ 技术栈

### 后端
| 组件 | 技术 |
|------|------|
| 框架 | FastAPI |
| 数据库 | SQLite + sqlite-vec（向量存储） |
| LLM | 阿里云百炼（qwen-plus） |
| Embedding | 百炼 text-embedding-v3 |
| 文档解析 | PyMuPDF、python-docx、markdown |
| 分块 | 递归/Markdown/语义分块 |
| 检索 | BM25（rank_bm25）+ 向量检索 + RRF |

### 前端
| 组件 | 技术 |
|------|------|
| 框架 | Next.js 14（App Router） |
| 语言 | TypeScript |
| 样式 | Tailwind CSS |
| 组件库 | shadcn/ui |
| 状态管理 | Zustand |
| HTTP | 原生 fetch + ReadableStream（SSE） |

## 📁 项目结构

```
├── backend/
│   ├── app/
│   │   ├── api/            # API 路由（7个模块）
│   │   ├── auth/           # JWT 认证和权限
│   │   ├── core/           # 核心组件（检索器、生成器、意图识别）
│   │   ├── db/             # SQLite 数据库和向量存储
│   │   ├── evaluation/     # 评估引擎
│   │   ├── ingestion/      # 文档处理管道
│   │   └── llm/            # LLM 接口
│   ├── tests/              # 测试用例
│   ├── config.py           # 配置管理
│   └── main.py             # FastAPI 入口
├── frontend/
│   ├── app/                # Next.js 页面
│   │   ├── admin/          # 管理后台页面
│   │   ├── chat/           # 聊天页面
│   │   └── login/          # 登录页面
│   ├── components/         # React 组件
│   │   ├── admin/          # 管理后台组件
│   │   ├── chat/           # 聊天组件
│   │   └── ui/             # shadcn/ui 组件
│   ├── lib/                # 工具函数
│   ├── stores/             # Zustand 状态管理
│   └── types/              # TypeScript 类型定义
├── docs/                   # 设计文档
└── .env.example            # 环境变量模板
```

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/hu06730/enterprise-rag-system.git
cd enterprise-rag-system
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入百炼 API Key：

```env
OPENAI_API_KEY=sk-你的百炼API密钥
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v3
EMBEDDING_DIMENSIONS=1024
LLM_MODEL=qwen-plus
```

### 3. 启动后端

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

后端 API 文档：http://localhost:8000/docs

### 4. 启动前端

```bash
cd frontend
npm install
npm run dev
```

前端地址：http://localhost:3000

### 5. 登录系统

- 用户名：`admin`
- 密码：`admin123`

## 📖 API 文档

启动后端后访问：
- Swagger UI：http://localhost:8000/docs
- ReDoc：http://localhost:8000/redoc

### 主要 API 端点

| 模块 | 端点 | 说明 |
|------|------|------|
| 认证 | POST /api/v1/auth/login | 用户登录 |
| 知识库 | GET/POST /api/v1/knowledge-bases/ | 知识库 CRUD |
| 文档 | POST /api/v1/documents/upload | 上传文档 |
| 对话 | POST /api/v1/chat/ask/stream | 流式问答 |
| 审计 | GET /api/v1/audit/logs | 审计日志 |
| 评估 | POST /api/v1/evaluation/run/{kb_id} | 运行评估 |
| 管理 | GET /api/v1/admin/stats | 系统统计 |

## 🧪 运行测试

```bash
cd backend
python -m pytest tests/ -v
```

## 📝 设计文档

- [架构设计文档](docs/superpowers/specs/2026-05-23-enterprise-rag-design.md)
- [实现计划](docs/superpowers/plans/2026-05-23-enterprise-rag-backend.md)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License
