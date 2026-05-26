# 企业知识库 RAG 后端 — 实施计划

> **给执行代理的说明：** 必需子技能——使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 来按任务逐步执行此计划。步骤使用复选框（`- [ ]`）语法来追踪进度。

**目标：** 构建企业知识库 RAG 系统的完整后端——零外部服务，一条命令启动。

**架构：** FastAPI 服务采用分层模块架构：数据管道（解析 → 分块 → 嵌入 → 存储），混合检索（BM25 + 向量 + 基于意图感知权重的 RRF 融合），生成引擎（Prompt 模板 + LLM 抽象），全部基于单个 SQLite 文件，使用 sqlite-vec 进行向量搜索。

**技术栈：** Python 3.11+，FastAPI，SQLite + sqlite-vec，rank_bm25，diskcache，OpenAI SDK，SentenceTransformers（可选，仅用于语义分块）

**设计文档：** `docs/superpowers/specs/2026-05-23-enterprise-rag-design.md`

---

### 任务 1：项目脚手架与配置

**涉及文件：**
- 新建：`backend/requirements.txt`
- 新建：`backend/app/__init__.py`
- 新建：`backend/app/config.py`
- 新建：`backend/app/main.py`（最小化版本）
- 新建：`.env.example`

- [ ] **步骤 1：创建 requirements.txt**

写入 `backend/requirements.txt`：
```
fastapi==0.115.6
uvicorn[standard]==0.34.0
sqlalchemy==2.0.36
sqlite-vec==0.1.6
pymupdf==1.25.1
python-docx==1.1.2
markdown-it-py==3.0.0
openai==1.58.1
rank-bm25==0.2.2
diskcache==5.6.3
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.18
sentence-transformers==3.3.1
httpx==0.28.1
pytest==8.3.4
pytest-asyncio==0.25.0
python-dotenv==1.0.1
```

- [ ] **步骤 2：创建 .env.example**

```bash
# .env.example
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
# DATABASE_URL 使用绝对路径；参见 config.py 中的默认值
DATABASE_URL=
UPLOAD_DIR=data/uploads
CACHE_DIR=data/cache
SECRET_KEY=change-me-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=1440
```

- [ ] **步骤 3：编写 config.py**

写入 `backend/app/config.py`：
```python
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/data/rag.db")
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", str(BASE_DIR / "data" / "uploads"))
    CACHE_DIR: str = os.getenv("CACHE_DIR", str(BASE_DIR / "data" / "cache"))
    SECRET_KEY: str = os.getenv("SECRET_KEY", "change-me-in-production")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_EXPIRE_MINUTES: int = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))

    @property
    def sqlite_path(self) -> str:
        return self.DATABASE_URL.replace("sqlite:///", "")


settings = Settings()
```

- [ ] **步骤 4：编写最小化 main.py**

写入 `backend/app/main.py`：
```python
from fastapi import FastAPI
from app.config import settings

app = FastAPI(title="Enterprise RAG API", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}
```

- [ ] **步骤 5：创建数据目录并验证启动**

执行：`cd backend && mkdir -p data/uploads data/cache && python -c "from app.main import app; print('OK')"`
预期输出：`OK`

- [ ] **步骤 6：验证 uvicorn 能否启动**

执行：`cd backend && timeout 3 uvicorn app.main:app --port 8000 || true`
预期结果：服务器启动，然后超时。无崩溃。

---

### 任务 2：数据库层 — SQLite 连接与模式

**涉及文件：**
- 新建：`backend/app/db/__init__.py`
- 新建：`backend/app/db/sqlite.py`
- 新建：`backend/app/db/vec_store.py`
- 修改：`backend/app/main.py`（添加 lifespan 以在启动时初始化数据库）

- [ ] **步骤 1：编写 db/sqlite.py**

写入 `backend/app/db/sqlite.py`：
```python
import sqlite3
import os
from app.config import settings


def get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(settings.sqlite_path), exist_ok=True)
    conn = sqlite3.connect(settings.sqlite_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'viewer',
            departments TEXT DEFAULT '[]',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS knowledge_bases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT DEFAULT '',
            kb_type TEXT NOT NULL DEFAULT 'employee',
            access_level TEXT NOT NULL DEFAULT 'internal',
            allowed_departments TEXT DEFAULT '[]',
            allowed_users TEXT DEFAULT '[]',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS kb_config (
            kb_id INTEGER PRIMARY KEY,
            chunk_strategy TEXT DEFAULT 'recursive',
            chunk_size INTEGER DEFAULT 512,
            chunk_overlap INTEGER DEFAULT 50,
            retrieval_mode TEXT DEFAULT 'hybrid',
            top_k INTEGER DEFAULT 10,
            min_score REAL DEFAULT 0.0,
            rerank_enabled INTEGER DEFAULT 1,
            vector_weight REAL DEFAULT 0.5,
            bm25_weight REAL DEFAULT 0.5,
            prompt_template TEXT DEFAULT '',
            FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kb_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_type TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            access_level TEXT,
            metadata_tags TEXT DEFAULT '[]',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS chunks_meta (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER NOT NULL,
            kb_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            title TEXT DEFAULT '',
            text TEXT NOT NULL,
            department TEXT,
            access_level TEXT DEFAULT 'internal',
            tags TEXT DEFAULT '[]',
            FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE,
            FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            kb_id INTEGER,
            query_text TEXT NOT NULL,
            answer_text TEXT,
            sources TEXT DEFAULT '[]',
            feedback INTEGER,
            feedback_text TEXT,
            trace_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS eval_dataset (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kb_id INTEGER NOT NULL,
            question TEXT NOT NULL,
            reference_answer TEXT,
            relevant_doc_ids TEXT DEFAULT '[]',
            created_by INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS intent_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            intent_name TEXT NOT NULL,
            keywords TEXT NOT NULL,
            vector_weight REAL NOT NULL DEFAULT 0.5,
            bm25_weight REAL NOT NULL DEFAULT 0.5,
            priority INTEGER DEFAULT 0
        );
    """)
    conn.commit()
    conn.close()
```

- [ ] **步骤 2：编写 db/vec_store.py**

写入 `backend/app/db/vec_store.py`：
```python
import sqlite3
import sqlite_vec
from app.db.sqlite import get_connection


def init_vec():
    conn = get_connection()
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
            embedding float[1536]
        )
    """)
    conn.commit()
    conn.close()


def insert_chunk_vec_and_meta(
    embedding: list[float],
    doc_id: int,
    kb_id: int,
    chunk_index: int,
    text: str,
    title: str = "",
    department: str | None = None,
    access_level: str = "internal",
    tags: str = "[]",
) -> int:
    conn = get_connection()
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    cur = conn.execute(
        """INSERT INTO chunks_meta (doc_id, kb_id, chunk_index, title, text, department, access_level, tags)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (doc_id, kb_id, chunk_index, title, text, department, access_level, tags),
    )
    rowid = cur.lastrowid
    vec_json = "[" + ",".join(str(v) for v in embedding) + "]"
    conn.execute(
        "INSERT INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
        (rowid, vec_json),
    )
    conn.commit()
    conn.close()
    return rowid


def delete_chunks_by_doc(doc_id: int):
    conn = get_connection()
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute(
        "DELETE FROM chunks_vec WHERE rowid IN (SELECT id FROM chunks_meta WHERE doc_id=?)",
        (doc_id,),
    )
    conn.execute("DELETE FROM chunks_meta WHERE doc_id=?", (doc_id,))
    conn.commit()
    conn.close()


def delete_chunks_by_kb(kb_id: int):
    conn = get_connection()
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute(
        "DELETE FROM chunks_vec WHERE rowid IN (SELECT id FROM chunks_meta WHERE kb_id=?)",
        (kb_id,),
    )
    conn.execute("DELETE FROM chunks_meta WHERE kb_id=?", (kb_id,))
    conn.commit()
    conn.close()


def search_similar(
    query_embedding: list[float],
    kb_id: int,
    access_levels: list[str],
    departments: list[str] | None = None,
    limit: int = 20,
    min_score: float = 0.0,
) -> list[dict]:
    conn = get_connection()
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    vec_json = "[" + ",".join(str(v) for v in query_embedding) + "]"

    if departments:
        deps_str = ",".join(f"'{d}'" for d in departments)
        dept_clause = f"AND (m.department IS NULL OR m.department IN ({deps_str}))"
    else:
        dept_clause = ""

    access_placeholders = ",".join("?" for _ in access_levels)

    query = f"""
        SELECT m.id, m.doc_id, m.kb_id, m.chunk_index, m.title, m.text,
               m.department, m.access_level, m.tags,
               vec_distance_cosine(v.embedding, ?) as distance
        FROM chunks_vec v
        JOIN chunks_meta m ON v.rowid = m.id
        WHERE m.kb_id = ?
          AND m.access_level IN ({access_placeholders})
          {dept_clause}
          AND vec_distance_cosine(v.embedding, ?) <= ?
        ORDER BY distance
        LIMIT ?
    """

    params = [vec_json, kb_id] + access_levels + [vec_json, 2.0, limit]
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]
```

- [ ] **步骤 3：更新 main.py 以在启动时初始化数据库**

修改 `backend/app/main.py`：
```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.db.sqlite import init_db
from app.db.vec_store import init_vec


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    init_vec()
    yield


app = FastAPI(title="Enterprise RAG API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}
```

- [ ] **步骤 4：验证数据库初始化是否正常**

执行：`cd backend && python -c "from app.main import app; print('DB OK')"`
预期输出：`DB OK`（创建 `data/rag.db`）

执行：`cd backend && python -c "
from app.db.sqlite import get_connection
conn = get_connection()
tables = conn.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall()
print([t['name'] for t in tables])
conn.close()
"`
预期输出：打印包含 `users`、`knowledge_bases`、`chunks_meta`、`chunks_vec` 等的列表。

---

### 任务 3：核心接口定义

**涉及文件：**
- 新建：`backend/app/core/__init__.py`
- 新建：`backend/app/core/interfaces.py`

- [ ] **步骤 1：编写 core/interfaces.py**

写入 `backend/app/core/interfaces.py`：
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections.abc import AsyncIterator


@dataclass
class Message:
    role: str
    content: str


@dataclass
class GenerationResult:
    content: str
    model: str = ""
    usage: dict = field(default_factory=dict)


@dataclass
class Chunk:
    id: int
    doc_id: int
    kb_id: int
    chunk_index: int
    title: str
    text: str
    department: str | None = None
    access_level: str = "internal"
    score: float = 0.0
    source: str = ""


@dataclass
class RetrievalContext:
    kb_id: int
    kb_type: str = "employee"
    user_departments: list[str] = field(default_factory=list)
    user_role: str = "viewer"
    access_levels: list[str] = field(default_factory=lambda: ["public", "internal"])
    top_k: int = 10
    min_score: float = 0.0
    retrieval_mode: str = "hybrid"
    vector_weight: float = 0.5
    bm25_weight: float = 0.5
    rerank_enabled: bool = True


@dataclass
class ParsedDocument:
    text: str
    title: str = ""
    metadata: dict = field(default_factory=dict)
    pages: list[str] = field(default_factory=list)


class BaseRetriever(ABC):
    @abstractmethod
    async def retrieve(self, query: str, ctx: RetrievalContext) -> list[Chunk]: ...


class BaseSplitter(ABC):
    @abstractmethod
    def split(self, text: str, **kwargs) -> list[str]: ...


class BaseLLM(ABC):
    @abstractmethod
    async def generate(self, messages: list[Message], **kwargs) -> GenerationResult: ...

    @abstractmethod
    async def generate_stream(self, messages: list[Message]) -> AsyncIterator[str]: ...


class BaseEmbedder(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]: ...


class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_path: str) -> ParsedDocument: ...
```

- [ ] **步骤 2：验证导入**

执行：`cd backend && python -c "from app.core.interfaces import BaseRetriever, BaseLLM, BaseEmbedder, BaseParser, BaseSplitter; print('OK')"`
预期输出：`OK`

---

### 任务 4：文档解析器

**涉及文件：**
- 新建：`backend/app/ingestion/__init__.py`
- 新建：`backend/app/ingestion/parsers.py`
- 新建：`backend/tests/test_ingestion/__init__.py`
- 新建：`backend/tests/test_ingestion/test_parsers.py`

- [ ] **步骤 1：编写 TXT 解析器的失败测试**

写入 `backend/tests/test_ingestion/test_parsers.py`：
```python
import pytest
from pathlib import Path

SAMPLE_DIR = Path(__file__).parent / "fixtures"


def test_txt_parser_returns_parsed_document():
    from app.ingestion.parsers import TxtParser
    parser = TxtParser()
    filepath = SAMPLE_DIR / "sample.txt"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text("Hello RAG.\nThis is a test document.", encoding="utf-8")

    result = parser.parse(str(filepath))
    assert result.text == "Hello RAG.\nThis is a test document."
    assert result.title == "sample.txt"
```

- [ ] **步骤 2：运行测试 — 预期失败**

执行：`cd backend && python -m pytest tests/test_ingestion/test_parsers.py::test_txt_parser_returns_parsed_document -v`
预期结果：FAIL — `ModuleNotFoundError` 或 `ImportError`

- [ ] **步骤 3：实现解析器**

写入 `backend/app/ingestion/parsers.py`：
```python
import os
from pathlib import Path
from app.core.interfaces import BaseParser, ParsedDocument


class TxtParser(BaseParser):
    def parse(self, file_path: str) -> ParsedDocument:
        with open(file_path, encoding="utf-8") as f:
            text = f.read()
        return ParsedDocument(
            text=text,
            title=os.path.basename(file_path),
        )


class PdfParser(BaseParser):
    def parse(self, file_path: str) -> ParsedDocument:
        import fitz
        doc = fitz.open(file_path)
        pages = [page.get_text() for page in doc]
        text = "\n\n".join(pages)
        doc.close()
        return ParsedDocument(
            text=text,
            title=os.path.basename(file_path),
            metadata={"page_count": len(pages)},
            pages=pages,
        )


class WordParser(BaseParser):
    def parse(self, file_path: str) -> ParsedDocument:
        from docx import Document
        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        text = "\n\n".join(paragraphs)
        return ParsedDocument(
            text=text,
            title=os.path.basename(file_path),
            metadata={"paragraph_count": len(paragraphs)},
        )


class MdParser(BaseParser):
    def parse(self, file_path: str) -> ParsedDocument:
        with open(file_path, encoding="utf-8") as f:
            text = f.read()
        return ParsedDocument(
            text=text,
            title=os.path.basename(file_path),
            metadata={"format": "markdown"},
        )


PARSER_REGISTRY: dict[str, type[BaseParser]] = {
    ".txt": TxtParser,
    ".pdf": PdfParser,
    ".docx": WordParser,
    ".doc": WordParser,
    ".md": MdParser,
    ".markdown": MdParser,
}


def get_parser(file_path: str) -> BaseParser:
    ext = Path(file_path).suffix.lower()
    parser_cls = PARSER_REGISTRY.get(ext)
    if parser_cls is None:
        raise ValueError(f"No parser registered for extension: {ext}")
    return parser_cls()
```

- [ ] **步骤 4：运行测试 — 预期通过**

执行：`cd backend && python -m pytest tests/test_ingestion/test_parsers.py::test_txt_parser_returns_parsed_document -v`
预期结果：PASS

- [ ] **步骤 5：添加 MD 解析器测试**

追加到 `backend/tests/test_ingestion/test_parsers.py`：
```python
def test_md_parser():
    from app.ingestion.parsers import MdParser
    filepath = SAMPLE_DIR / "sample.md"
    filepath.write_text("# Title\n\nContent here.", encoding="utf-8")
    result = MdParser().parse(str(filepath))
    assert "# Title" in result.text
    assert result.metadata["format"] == "markdown"
```

执行：`cd backend && python -m pytest tests/test_ingestion/test_parsers.py::test_md_parser -v`
预期结果：PASS

- [ ] **步骤 6：添加 get_parser 测试**

追加到 `backend/tests/test_ingestion/test_parsers.py`：
```python
def test_get_parser_returns_correct_type():
    from app.ingestion.parsers import get_parser, TxtParser, PdfParser, WordParser, MdParser
    assert isinstance(get_parser("a.txt"), TxtParser)
    assert isinstance(get_parser("a.pdf"), PdfParser)
    assert isinstance(get_parser("a.docx"), WordParser)
    assert isinstance(get_parser("a.md"), MdParser)


def test_get_parser_raises_for_unknown():
    import pytest
    from app.ingestion.parsers import get_parser
    with pytest.raises(ValueError, match="No parser"):
        get_parser("a.xyz")
```

执行：`cd backend && python -m pytest tests/test_ingestion/test_parsers.py::test_get_parser_returns_correct_type tests/test_ingestion/test_parsers.py::test_get_parser_raises_for_unknown -v`
预期结果：两个都 PASS

---

### 任务 5：分块器

**涉及文件：**
- 新建：`backend/app/ingestion/splitter.py`
- 新建：`backend/tests/test_ingestion/test_splitter.py`

- [ ] **步骤 1：编写递归分块器的失败测试**

写入 `backend/tests/test_ingestion/test_splitter.py`：
```python
def test_recursive_splitter_chunks_by_separators():
    from app.ingestion.splitter import RecursiveSplitter
    splitter = RecursiveSplitter(chunk_size=50, chunk_overlap=10)
    text = "第一段内容。\n\n第二段，这里有更多内容。\n\n第三段。"
    chunks = splitter.split(text)
    assert len(chunks) >= 2
    assert all(len(c) <= 50 for c in chunks)
```

- [ ] **步骤 2：运行测试 — 预期失败**

执行：`cd backend && python -m pytest tests/test_ingestion/test_splitter.py::test_recursive_splitter_chunks_by_separators -v`
预期结果：FAIL

- [ ] **步骤 3：实现分块器**

写入 `backend/app/ingestion/splitter.py`：
```python
import re
from app.core.interfaces import BaseSplitter


class RecursiveSplitter(BaseSplitter):
    SEPARATORS = ["\n\n", "\n", "。", ". ", ".", "；", ";", " ", ""]

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str, **kwargs) -> list[str]:
        return self._split_text(text, self.SEPARATORS)

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        if not text.strip():
            return []
        sep = separators[0]
        remaining_seps = separators[1:]
        if sep == "":
            return self._split_by_length(text)
        if sep in text:
            parts = text.split(sep)
        else:
            return self._split_text(text, remaining_seps)
        chunks = []
        current = ""
        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                if len(part) > self.chunk_size:
                    sub = self._split_text(part, remaining_seps)
                    if current and self.chunk_overlap > 0 and sub:
                        sub[0] = current[-self.chunk_overlap:] + sub[0]
                    chunks.extend(sub)
                    current = ""
                else:
                    current = part
        if current.strip():
            chunks.append(current.strip())
        return chunks

    def _split_by_length(self, text: str) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end].strip())
            start = end - self.chunk_overlap
        return chunks


class MarkdownSplitter(BaseSplitter):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str, **kwargs) -> list[str]:
        chunks = []
        sections = re.split(r'(?=^#{1,3} )', text, flags=re.MULTILINE)
        for section in sections:
            if not section.strip():
                continue
            content = section.strip()
            if len(content) <= self.chunk_size:
                chunks.append(content)
            else:
                paragraphs = content.split("\n\n")
                current = ""
                for p in paragraphs:
                    candidate = current + ("\n\n" if current else "") + p
                    if len(candidate) <= self.chunk_size:
                        current = candidate
                    else:
                        if current:
                            chunks.append(current)
                        current = p
                if current:
                    chunks.append(current)
        return chunks


class SemanticSplitter(BaseSplitter):
    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        min_chunk_size: int = 100,
        similarity_threshold: float = 0.5,
    ):
        self.model_name = model_name
        self.min_chunk_size = min_chunk_size
        self.similarity_threshold = similarity_threshold
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def split(self, text: str, **kwargs) -> list[str]:
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [text] if text.strip() else []
        embeddings = self.model.encode(sentences)
        import numpy as np
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1]) + 1e-8
            )
            similarities.append(float(sim))
        split_points = [0]
        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold:
                split_points.append(i + 1)
        split_points.append(len(sentences))
        chunks = []
        for i in range(len(split_points) - 1):
            chunk = "".join(sentences[split_points[i]:split_points[i + 1]])
            if chunk.strip() and len(chunk) >= self.min_chunk_size:
                chunks.append(chunk.strip())
        if not chunks:
            return [text.strip()]
        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        return re.split(r'(?<=[。！？.!?\n])\s*', text)


SPLITTER_REGISTRY: dict[str, type[BaseSplitter]] = {
    "recursive": RecursiveSplitter,
    "markdown": MarkdownSplitter,
    "semantic": SemanticSplitter,
}


def get_splitter(strategy: str, **kwargs) -> BaseSplitter:
    cls = SPLITTER_REGISTRY.get(strategy)
    if cls is None:
        raise ValueError(f"Unknown splitter strategy: {strategy}")
    return cls(**kwargs)
```

- [ ] **步骤 4：运行测试 — 预期通过**

执行：`cd backend && python -m pytest tests/test_ingestion/test_splitter.py::test_recursive_splitter_chunks_by_separators -v`
预期结果：PASS

- [ ] **步骤 5：添加 Markdown 分块器测试**

追加到 `backend/tests/test_ingestion/test_splitter.py`：
```python
def test_markdown_splitter_preserves_headings():
    from app.ingestion.splitter import MarkdownSplitter
    splitter = MarkdownSplitter(chunk_size=500, chunk_overlap=0)
    text = "## Section A\n\nContent for A.\n\n## Section B\n\nContent for B."
    chunks = splitter.split(text)
    assert any("Section A" in c for c in chunks)
    assert any("Section B" in c for c in chunks)


def test_get_splitter_raises_for_unknown():
    import pytest
    from app.ingestion.splitter import get_splitter
    with pytest.raises(ValueError, match="Unknown splitter"):
        get_splitter("nonexistent")
```

执行：`cd backend && python -m pytest tests/test_ingestion/test_splitter.py -v`
预期结果：3 个 PASS

---

### 任务 6：嵌入服务

**涉及文件：**
- 新建：`backend/app/ingestion/embedder.py`
- 新建：`backend/tests/test_ingestion/test_embedder.py`

- [ ] **步骤 1：编写失败测试**

写入 `backend/tests/test_ingestion/test_embedder.py`：
```python
import pytest


@pytest.mark.asyncio
async def test_openai_embedder_returns_correct_dimensions():
    from app.ingestion.embedder import OpenAIEmbedder
    embedder = OpenAIEmbedder(model="text-embedding-3-small")
    result = await embedder.embed(["hello world"])
    assert len(result) == 1
    assert len(result[0]) == 1536


@pytest.mark.asyncio
async def test_openai_embedder_single_query():
    from app.ingestion.embedder import OpenAIEmbedder
    embedder = OpenAIEmbedder(model="text-embedding-3-small")
    vec = await embedder.embed_query("test query")
    assert len(vec) == 1536
```

- [ ] **步骤 2：运行测试 — 预期失败**

执行：`cd backend && python -m pytest tests/test_ingestion/test_embedder.py -v`
预期结果：FAIL

- [ ] **步骤 3：实现嵌入服务**

写入 `backend/app/ingestion/embedder.py`：
```python
from openai import AsyncOpenAI
from app.core.interfaces import BaseEmbedder
from app.config import settings


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model: str | None = None):
        self.model = model or settings.EMBEDDING_MODEL
        self.client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        resp = await self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

    async def embed_query(self, text: str) -> list[float]:
        results = await self.embed([text])
        return results[0]
```

- [ ] **步骤 4：运行测试 — 预期通过**

执行：`cd backend && python -m pytest tests/test_ingestion/test_embedder.py -v`
预期结果：2 个 PASS（需要在 .env 中配置 OPENAI_API_KEY）

---

### 任务 7：数据管道编排

**涉及文件：**
- 新建：`backend/app/ingestion/pipeline.py`
- 新建：`backend/tests/test_ingestion/test_pipeline.py`

- [ ] **步骤 1：编写失败测试**

写入 `backend/tests/test_ingestion/test_pipeline.py`：
```python
import pytest
from pathlib import Path


@pytest.mark.asyncio
async def test_pipeline_processes_txt_and_stores_chunks():
    from app.db.sqlite import init_db, get_connection
    from app.db.vec_store import init_vec
    from app.ingestion.pipeline import run_ingestion

    init_db()
    init_vec()

    conn = get_connection()
    conn.execute(
        "INSERT INTO knowledge_bases (id, name, kb_type) VALUES (1, 'test', 'employee')"
    )
    conn.execute(
        "INSERT INTO kb_config (kb_id, chunk_strategy, chunk_size, chunk_overlap) VALUES (1, 'recursive', 200, 20)"
    )
    conn.execute(
        "INSERT INTO documents (id, kb_id, filename, file_path, file_type, status) VALUES (1, 1, 'test.txt', ?, 'txt', 'pending')",
        (str(Path(__file__).parent / "fixtures" / "ingest_sample.txt"),),
    )
    conn.commit()
    conn.close()

    sample = Path(__file__).parent / "fixtures" / "ingest_sample.txt"
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_text("This is test content for the ingestion pipeline.\nIt has two sentences.", encoding="utf-8")

    await run_ingestion(doc_id=1)

    conn = get_connection()
    count = conn.execute("SELECT COUNT(*) FROM chunks_meta WHERE doc_id=1").fetchone()[0]
    conn.close()
    assert count >= 1
```

- [ ] **步骤 2：运行测试 — 预期失败**

执行：`cd backend && python -m pytest tests/test_ingestion/test_pipeline.py -v`
预期结果：FAIL

- [ ] **步骤 3：实现管道**

写入 `backend/app/ingestion/pipeline.py`：
```python
import json
from app.db.sqlite import get_connection
from app.db.vec_store import insert_chunk_vec_and_meta
from app.ingestion.parsers import get_parser
from app.ingestion.splitter import get_splitter
from app.ingestion.embedder import OpenAIEmbedder


async def run_ingestion(doc_id: int):
    conn = get_connection()
    doc = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
    if not doc:
        conn.close()
        raise ValueError(f"Document {doc_id} not found")

    kb_id = doc["kb_id"]
    config = conn.execute("SELECT * FROM kb_config WHERE kb_id=?", (kb_id,)).fetchone()
    if not config:
        conn.close()
        raise ValueError(f"KB config for kb_id={kb_id} not found")

    conn.execute("UPDATE documents SET status='processing' WHERE id=?", (doc_id,))
    conn.commit()
    conn.close()

    try:
        parser = get_parser(doc["file_path"])
        parsed = parser.parse(doc["file_path"])

        splitter = get_splitter(
            strategy=config["chunk_strategy"],
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
        )
        chunks = splitter.split(parsed.text)

        embedder = OpenAIEmbedder()
        embeddings = await embedder.embed(chunks)

        access_level = doc["access_level"] or "internal"
        tags = doc["metadata_tags"] or "[]"

        for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            insert_chunk_vec_and_meta(
                embedding=embedding,
                doc_id=doc_id,
                kb_id=kb_id,
                chunk_index=i,
                text=chunk_text,
                title=parsed.title,
                access_level=access_level,
                tags=tags,
            )

        conn = get_connection()
        conn.execute("UPDATE documents SET status='completed' WHERE id=?", (doc_id,))
        conn.commit()
        conn.close()
    except Exception as e:
        conn = get_connection()
        conn.execute("UPDATE documents SET status='failed' WHERE id=?", (doc_id,))
        conn.commit()
        conn.close()
        raise e
```

- [ ] **步骤 4：运行测试 — 预期通过**

执行：`cd backend && python -m pytest tests/test_ingestion/test_pipeline.py::test_pipeline_processes_txt_and_stores_chunks -v`
预期结果：PASS（需要 OPENAI_API_KEY）

---

### 任务 8：LLM 适配器

**涉及文件：**
- 新建：`backend/app/llm/__init__.py`
- 新建：`backend/app/llm/base.py`
- 新建：`backend/app/llm/openai_llm.py`
- 新建：`backend/tests/test_llm.py`

- [ ] **步骤 1：编写 LLM 基类**

写入 `backend/app/llm/base.py`：
```python
from app.core.interfaces import BaseLLM as BaseLLMInterface
from app.core.interfaces import Message, GenerationResult

__all__ = ["BaseLLMInterface", "Message", "GenerationResult"]
```

- [ ] **步骤 2：编写 OpenAI LLM 适配器**

写入 `backend/app/llm/openai_llm.py`：
```python
from collections.abc import AsyncIterator
from openai import AsyncOpenAI
from app.core.interfaces import BaseLLM, Message, GenerationResult
from app.config import settings


class OpenAILLM(BaseLLM):
    def __init__(self, model: str | None = None):
        self.model = model or settings.LLM_MODEL
        self.client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
        )

    async def generate(self, messages: list[Message], **kwargs) -> GenerationResult:
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=kwargs.get("temperature", 0.3),
            max_tokens=kwargs.get("max_tokens", 2048),
        )
        choice = resp.choices[0]
        return GenerationResult(
            content=choice.message.content or "",
            model=resp.model,
            usage={
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
            },
        )

    async def generate_stream(self, messages: list[Message]) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            stream=True,
            temperature=0.3,
            max_tokens=2048,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
```

- [ ] **步骤 3：编写测试**

写入 `backend/tests/test_llm.py`：
```python
import pytest


@pytest.mark.asyncio
async def test_openai_llm_generate():
    from app.llm.openai_llm import OpenAILLM
    from app.core.interfaces import Message
    llm = OpenAILLM()
    result = await llm.generate([Message(role="user", content="Reply with just: OK")])
    assert "OK" in result.content


@pytest.mark.asyncio
async def test_openai_llm_stream():
    from app.llm.openai_llm import OpenAILLM
    from app.core.interfaces import Message
    llm = OpenAILLM()
    chunks = []
    async for chunk in llm.generate_stream([Message(role="user", content="Say hello")]):
        chunks.append(chunk)
    assert len(chunks) > 0
    assert "".join(chunks).strip() != ""
```

执行：`cd backend && python -m pytest tests/test_llm.py -v`
预期结果：2 个 PASS

---

### 任务 9：查询意图分类器

**涉及文件：**
- 新建：`backend/app/core/intent.py`
- 新建：`backend/tests/test_intent.py`

- [ ] **步骤 1：编写失败测试**

写入 `backend/tests/test_intent.py`：
```python
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
```

- [ ] **步骤 2：运行测试 — 预期失败**

执行：`cd backend && python -m pytest tests/test_intent.py -v`
预期结果：FAIL

- [ ] **步骤 3：实现意图分类器**

写入 `backend/app/core/intent.py`：
```python
from app.db.sqlite import get_connection

DEFAULT_RULES = [
    {
        "intent_name": "factual_lookup",
        "keywords": "是什么 什么是 多少 多少天 多少钱 定义 有哪些 哪几个",
        "vector_weight": 0.3,
        "bm25_weight": 0.7,
        "priority": 10,
    },
    {
        "intent_name": "conceptual",
        "keywords": "为什么 区别 原理 关系 作用 影响 意义 对比",
        "vector_weight": 0.7,
        "bm25_weight": 0.3,
        "priority": 10,
    },
    {
        "intent_name": "procedural",
        "keywords": "怎么做 如何 怎么 流程 步骤 方法 怎样 如何操作 教程",
        "vector_weight": 0.5,
        "bm25_weight": 0.5,
        "priority": 10,
    },
    {
        "intent_name": "compliance",
        "keywords": "规定 标准 条例 政策 依据 合规 制度 规范 要求 必须",
        "vector_weight": 0.2,
        "bm25_weight": 0.8,
        "priority": 10,
    },
]


def load_rules() -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT intent_name, keywords, vector_weight, bm25_weight, priority FROM intent_rules ORDER BY priority DESC"
    ).fetchall()
    conn.close()
    if not rows:
        return DEFAULT_RULES
    return [dict(r) for r in rows]


def classify_intent(query: str) -> dict:
    rules = load_rules()
    for rule in rules:
        keywords = rule["keywords"].split()
        if any(kw in query for kw in keywords):
            return {
                "intent": rule["intent_name"],
                "vector_weight": rule["vector_weight"],
                "bm25_weight": rule["bm25_weight"],
            }
    return {"intent": "default", "vector_weight": 0.5, "bm25_weight": 0.5}
```

- [ ] **步骤 4：运行测试 — 预期通过**

执行：`cd backend && python -m pytest tests/test_intent.py -v`
预期结果：3 个 PASS

- [ ] **步骤 5：种子数据 — 默认意图规则入库**

写入 `backend/app/db/seed.py`：
```python
from app.db.sqlite import get_connection
from app.core.intent import DEFAULT_RULES


def seed_intent_rules():
    conn = get_connection()
    count = conn.execute("SELECT COUNT(*) FROM intent_rules").fetchone()[0]
    if count == 0:
        for rule in DEFAULT_RULES:
            conn.execute(
                "INSERT INTO intent_rules (intent_name, keywords, vector_weight, bm25_weight, priority) VALUES (?, ?, ?, ?, ?)",
                (rule["intent_name"], rule["keywords"], rule["vector_weight"], rule["bm25_weight"], rule["priority"]),
            )
        conn.commit()
    conn.close()
```

更新 `backend/app/main.py` 调用种子数据：
```python
from app.db.seed import seed_intent_rules

# 在 lifespan 中，init_db() 和 init_vec() 之后：
seed_intent_rules()
```

---

### 任务 10：检索引擎（向量 + BM25 + 混合检索 + RRF）

**涉及文件：**
- 新建：`backend/app/core/retriever.py`
- 新建：`backend/tests/test_retrieval/__init__.py`
- 新建：`backend/tests/test_retrieval/test_retriever.py`

- [ ] **步骤 1：编写检索测试**

写入 `backend/tests/test_retrieval/test_retriever.py`：
```python
import pytest
from app.core.interfaces import RetrievalContext


@pytest.mark.asyncio
async def test_vector_retriever_returns_chunks():
    from app.core.retriever import VectorRetriever
    from app.ingestion.embedder import OpenAIEmbedder

    embedder = OpenAIEmbedder()
    retriever = VectorRetriever(embedder=embedder)

    ctx = RetrievalContext(kb_id=1, top_k=5, access_levels=["public", "internal"])
    results = await retriever.retrieve("test query", ctx)
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_bm25_retriever_builds_index():
    from app.core.retriever import BM25Retriever

    retriever = BM25Retriever()
    ctx = RetrievalContext(kb_id=1, top_k=5, access_levels=["public", "internal"])
    results = await retriever.retrieve("test", ctx)
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_hybrid_retriever_merges_results():
    from app.core.retriever import HybridRetriever
    from app.ingestion.embedder import OpenAIEmbedder

    embedder = OpenAIEmbedder()
    retriever = HybridRetriever(embedder=embedder)

    ctx = RetrievalContext(kb_id=1, top_k=10, access_levels=["public", "internal"], retrieval_mode="hybrid")
    results = await retriever.retrieve("test query", ctx)
    assert isinstance(results, list)
    assert all(r.score >= 0 for r in results)
```

- [ ] **步骤 2：运行测试 — 预期失败**

执行：`cd backend && python -m pytest tests/test_retrieval/test_retriever.py -v`
预期结果：FAIL

- [ ] **步骤 3：实现检索引擎**

写入 `backend/app/core/retriever.py`：
```python
from app.core.interfaces import BaseRetriever, Chunk, RetrievalContext
from app.db.vec_store import search_similar
from app.db.sqlite import get_connection


class VectorRetriever(BaseRetriever):
    def __init__(self, embedder):
        self.embedder = embedder

    async def retrieve(self, query: str, ctx: RetrievalContext) -> list[Chunk]:
        embedding = await self.embedder.embed_query(query)
        rows = search_similar(
            query_embedding=embedding,
            kb_id=ctx.kb_id,
            access_levels=ctx.access_levels,
            departments=ctx.user_departments or None,
            limit=ctx.top_k * 2,
            min_score=ctx.min_score,
        )
        return [
            Chunk(
                id=r["id"], doc_id=r["doc_id"], kb_id=r["kb_id"],
                chunk_index=r["chunk_index"], title=r["title"],
                text=r["text"], department=r["department"],
                access_level=r["access_level"],
                score=float(r.get("distance", 0) or 0),
                source="vector",
            )
            for r in rows
        ]


class BM25Retriever(BaseRetriever):
    async def retrieve(self, query: str, ctx: RetrievalContext) -> list[Chunk]:
        from rank_bm25 import BM25Okapi
        import jieba

        conn = get_connection()
        dept_clause = ""
        params = [ctx.kb_id]
        if ctx.user_departments:
            placeholders = ",".join("?" for _ in ctx.user_departments)
            dept_clause = f"AND (department IS NULL OR department IN ({placeholders}))"
            params.extend(ctx.user_departments)

        access_placeholders = ",".join("?" for _ in ctx.access_levels)
        params.extend(ctx.access_levels)

        rows = conn.execute(
            f"SELECT id, doc_id, kb_id, chunk_index, title, text, department, access_level "
            f"FROM chunks_meta WHERE kb_id=? {dept_clause} AND access_level IN ({access_placeholders})",
            params,
        ).fetchall()
        conn.close()

        if not rows:
            return []

        texts = [r["text"] for r in rows]
        tokenized = [list(jieba.cut(t)) for t in texts]
        bm25 = BM25Okapi(tokenized)
        query_tokens = list(jieba.cut(query))
        scores = bm25.get_scores(query_tokens)

        results = []
        for i, r in enumerate(rows):
            results.append(Chunk(
                id=r["id"], doc_id=r["doc_id"], kb_id=r["kb_id"],
                chunk_index=r["chunk_index"], title=r["title"],
                text=r["text"], department=r["department"],
                access_level=r["access_level"],
                score=float(scores[i]),
                source="keyword",
            ))
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:ctx.top_k * 2]


class HybridRetriever(BaseRetriever):
    def __init__(self, embedder):
        self.vector = VectorRetriever(embedder)
        self.bm25 = BM25Retriever()
        self.k = 60

    async def retrieve(self, query: str, ctx: RetrievalContext) -> list[Chunk]:
        if ctx.retrieval_mode == "vector":
            return await self.vector.retrieve(query, ctx)[:ctx.top_k]
        if ctx.retrieval_mode == "keyword":
            return await self.bm25.retrieve(query, ctx)[:ctx.top_k]

        vec_results = await self.vector.retrieve(query, ctx)
        bm25_results = await self.bm25.retrieve(query, ctx)

        rrf_scores: dict[int, float] = {}
        chunk_map: dict[int, Chunk] = {}

        for rank, chunk in enumerate(vec_results):
            rrf_scores[chunk.id] = ctx.vector_weight / (self.k + rank + 1)
            chunk_map[chunk.id] = chunk

        for rank, chunk in enumerate(bm25_results):
            score = ctx.bm25_weight / (self.k + rank + 1)
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0) + score
            if chunk.id not in chunk_map:
                chunk_map[chunk.id] = chunk

        merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for chunk_id, rrf_score in merged[:ctx.top_k]:
            c = chunk_map[chunk_id]
            c.score = rrf_score
            results.append(c)
        return results
```

- [ ] **步骤 4：安装 jieba（如未安装）**

执行：`cd backend && pip install jieba`

- [ ] **步骤 5：运行测试 — 预期通过**

执行：`cd backend && python -m pytest tests/test_retrieval/test_retriever.py -v`
预期结果：3 个 PASS

---

### 任务 11：生成引擎

**涉及文件：**
- 新建：`backend/app/core/generator.py`
- 新建：`backend/tests/test_generation/__init__.py`
- 新建：`backend/tests/test_generation/test_generator.py`

- [ ] **步骤 1：编写测试**

写入 `backend/tests/test_generation/test_generator.py`：
```python
import pytest
from app.core.interfaces import Chunk, RetrievalContext


@pytest.mark.asyncio
async def test_generator_returns_answer_with_sources():
    from app.core.generator import Generator
    from app.llm.openai_llm import OpenAILLM

    llm = OpenAILLM()
    gen = Generator(llm=llm)

    chunks = [
        Chunk(id=1, doc_id=1, kb_id=1, chunk_index=0, title="员工手册.pdf",
              text="年假天数为每年15个工作日。", score=0.95, source="vector"),
        Chunk(id=2, doc_id=2, kb_id=1, chunk_index=1, title="FAQ.docx",
              text="申请年假需要提前3天在OA系统提交。", score=0.87, source="keyword"),
    ]
    ctx = RetrievalContext(kb_id=1, kb_type="employee")

    result = await gen.generate(query="年假怎么申请", chunks=chunks, ctx=ctx)
    assert result["answer"] != ""
    assert len(result["sources"]) > 0
    assert result["trace_id"] is not None
```

- [ ] **步骤 2：运行测试 — 预期失败**

执行：`cd backend && python -m pytest tests/test_generation/test_generator.py -v`
预期结果：FAIL

- [ ] **步骤 3：实现生成引擎**

写入 `backend/app/core/generator.py`：
```python
import json
import uuid
from app.core.interfaces import Chunk, Message, RetrievalContext
from app.db.sqlite import get_connection


PROMPT_TEMPLATES = {
    "employee": """你是企业内部知识助手。请严格根据以下参考资料回答问题。
如果参考资料中没有相关信息，请明确说"该问题在现有资料中未找到答案"。
请在你的回答中引用相关来源。

参考资料：
{context}

用户问题：{query}""",

    "customer": """你是一个友好的客服助手。请根据以下参考资料回答客户问题。
回答应该简洁、准确、友好。如果资料中没有答案，请礼貌地说你不确定，并建议联系人工客服。
不要编造任何信息。

参考资料：
{context}

客户问题：{query}""",

    "compliance": """你是合规审计助手。请根据以下参考资料逐条回答，每条回答必须标注具体来源。

要求：
1. 每个事实陈述必须引用来源文档名称和段落
2. 如果多条资料有冲突，请明确指出
3. 不得添加参考资料中没有的信息

参考资料：
{context}

查询：{query}""",
}


class Generator:
    def __init__(self, llm):
        self.llm = llm

    def _load_prompt_template(self, kb_id: int, kb_type: str) -> str:
        conn = get_connection()
        row = conn.execute(
            "SELECT prompt_template FROM kb_config WHERE kb_id=?", (kb_id,)
        ).fetchone()
        conn.close()
        if row and row["prompt_template"]:
            return row["prompt_template"]
        return PROMPT_TEMPLATES.get(kb_type, PROMPT_TEMPLATES["employee"])

    def _assemble_context(self, chunks: list[Chunk], max_tokens: int = 4000) -> tuple[str, list[dict]]:
        sorted_chunks = sorted(chunks, key=lambda x: x.score, reverse=True)
        context_parts = []
        sources = []
        total_chars = 0

        for c in sorted_chunks:
            source_text = f"[来源: {c.title}]\n{c.text}"
            if total_chars + len(source_text) > max_tokens * 3:
                break
            context_parts.append(source_text)
            sources.append({
                "doc_title": c.title,
                "chunk_text": c.text[:200],
                "score": round(c.score, 4),
            })
            total_chars += len(source_text)

        return "\n\n---\n\n".join(context_parts), sources

    async def generate(self, query: str, chunks: list[Chunk], ctx: RetrievalContext) -> dict:
        context, sources = self._assemble_context(chunks)
        template = self._load_prompt_template(ctx.kb_id, ctx.kb_type)
        prompt = template.format(context=context, query=query)

        result = await self.llm.generate([Message(role="user", content=prompt)])
        trace_id = str(uuid.uuid4())[:8]

        return {
            "answer": result.content,
            "sources": sources,
            "confidence": self._estimate_confidence(chunks),
            "trace_id": trace_id,
        }

    async def generate_stream(self, query: str, chunks: list[Chunk], ctx: RetrievalContext):
        context, sources = self._assemble_context(chunks)
        template = self._load_prompt_template(ctx.kb_id, ctx.kb_type)
        prompt = template.format(context=context, query=query)

        yield json.dumps({"type": "sources", "data": sources}) + "\n"
        async for token in self.llm.generate_stream([Message(role="user", content=prompt)]):
            yield json.dumps({"type": "token", "data": token}) + "\n"
        trace_id = str(uuid.uuid4())[:8]
        yield json.dumps({"type": "done", "trace_id": trace_id}) + "\n"

    def _estimate_confidence(self, chunks: list[Chunk]) -> float:
        if not chunks:
            return 0.0
        top_score = max(c.score for c in chunks)
        return round(min(top_score * 10, 1.0), 2)
```

- [ ] **步骤 4：运行测试 — 预期通过**

执行：`cd backend && python -m pytest tests/test_generation/test_generator.py -v`
预期结果：PASS

---

### 任务 12：认证模块 — JWT 与权限

**涉及文件：**
- 新建：`backend/app/auth/__init__.py`
- 新建：`backend/app/auth/jwt.py`
- 新建：`backend/app/auth/permissions.py`

- [ ] **步骤 1：编写 JWT 模块**

写入 `backend/app/auth/jwt.py`：
```python
from datetime import datetime, timedelta, timezone
from jose import jwt
from passlib.context import CryptContext
from app.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_token(user_id: int, role: str) -> str:
    payload = {
        "sub": str(user_id),
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=settings.JWT_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
```

- [ ] **步骤 2：编写权限模块**

写入 `backend/app/auth/permissions.py`：
```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.auth.jwt import decode_token
from app.db.sqlite import get_connection

security = HTTPBearer()


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    try:
        payload = decode_token(credentials.credentials)
        user_id = int(payload["sub"])
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    conn = get_connection()
    user = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    conn.close()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    import json
    return {
        "id": user["id"],
        "username": user["username"],
        "role": user["role"],
        "departments": json.loads(user["departments"] or "[]"),
    }


def require_role(*roles: str):
    def checker(user: dict = Depends(get_current_user)):
        if user["role"] not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return user
    return checker
```

---

### 任务 13：API 路由 — 认证端点

**涉及文件：**
- 新建：`backend/app/api/__init__.py`
- 新建：`backend/app/api/auth.py`

- [ ] **步骤 1：编写认证路由**

写入 `backend/app/api/auth.py`：
```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.auth.jwt import hash_password, verify_password, create_token
from app.auth.permissions import get_current_user, require_role
from app.db.sqlite import get_connection
import json

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    role: str = "viewer"
    departments: list[str] = []


@router.post("/login")
def login(req: LoginRequest):
    conn = get_connection()
    user = conn.execute("SELECT * FROM users WHERE username=?", (req.username,)).fetchone()
    conn.close()
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_token(user["id"], user["role"])
    return {
        "code": 0,
        "data": {"token": token, "username": user["username"], "role": user["role"]},
        "message": "ok",
    }


@router.post("/register")
def register(req: RegisterRequest, user: dict = Depends(require_role("admin"))):
    conn = get_connection()
    existing = conn.execute("SELECT id FROM users WHERE username=?", (req.username,)).fetchone()
    if existing:
        conn.close()
        raise HTTPException(status_code=400, detail="Username already exists")

    conn.execute(
        "INSERT INTO users (username, password_hash, role, departments) VALUES (?, ?, ?, ?)",
        (req.username, hash_password(req.password), req.role, json.dumps(req.departments)),
    )
    conn.commit()
    conn.close()
    return {"code": 0, "data": None, "message": "User created"}


@router.get("/me")
def me(user: dict = Depends(get_current_user)):
    return {"code": 0, "data": user, "message": "ok"}
```

---

### 任务 14：API 路由 — 知识库 CRUD

**涉及文件：**
- 新建：`backend/app/api/knowledge_bases.py`

- [ ] **步骤 1：编写知识库路由**

写入 `backend/app/api/knowledge_bases.py`：
```python
import json
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.auth.permissions import get_current_user, require_role
from app.db.sqlite import get_connection

router = APIRouter(prefix="/api/v1/knowledge-bases", tags=["knowledge-bases"])


class KBCreate(BaseModel):
    name: str
    description: str = ""
    kb_type: str = "employee"
    access_level: str = "internal"
    allowed_departments: list[str] = []
    allowed_users: list[int] = []
    chunk_strategy: str = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 50
    retrieval_mode: str = "hybrid"
    top_k: int = 10
    vector_weight: float = 0.5
    bm25_weight: float = 0.5


@router.get("/")
def list_kbs(user: dict = Depends(get_current_user)):
    conn = get_connection()
    rows = conn.execute("SELECT * FROM knowledge_bases").fetchall()
    conn.close()
    return {"code": 0, "data": [dict(r) for r in rows], "message": "ok"}


@router.post("/")
def create_kb(req: KBCreate, user: dict = Depends(require_role("admin", "editor"))):
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO knowledge_bases (name, description, kb_type, access_level, allowed_departments, allowed_users) VALUES (?, ?, ?, ?, ?, ?)",
        (req.name, req.description, req.kb_type, req.access_level,
         json.dumps(req.allowed_departments), json.dumps(req.allowed_users)),
    )
    kb_id = cur.lastrowid
    conn.execute(
        """INSERT INTO kb_config (kb_id, chunk_strategy, chunk_size, chunk_overlap, retrieval_mode, top_k, vector_weight, bm25_weight)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (kb_id, req.chunk_strategy, req.chunk_size, req.chunk_overlap,
         req.retrieval_mode, req.top_k, req.vector_weight, req.bm25_weight),
    )
    conn.commit()
    conn.close()
    return {"code": 0, "data": {"id": kb_id}, "message": "Knowledge base created"}


@router.get("/{kb_id}")
def get_kb(kb_id: int, user: dict = Depends(get_current_user)):
    conn = get_connection()
    kb = conn.execute("SELECT * FROM knowledge_bases WHERE id=?", (kb_id,)).fetchone()
    config = conn.execute("SELECT * FROM kb_config WHERE kb_id=?", (kb_id,)).fetchone()
    conn.close()
    if not kb:
        raise HTTPException(status_code=404, detail="Not found")
    return {"code": 0, "data": {"kb": dict(kb), "config": dict(config) if config else None}, "message": "ok"}


@router.put("/{kb_id}")
def update_kb(kb_id: int, req: KBCreate, user: dict = Depends(require_role("admin", "editor"))):
    conn = get_connection()
    conn.execute(
        "UPDATE knowledge_bases SET name=?, description=?, kb_type=?, access_level=?, allowed_departments=?, allowed_users=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
        (req.name, req.description, req.kb_type, req.access_level,
         json.dumps(req.allowed_departments), json.dumps(req.allowed_users), kb_id),
    )
    conn.execute(
        "UPDATE kb_config SET chunk_strategy=?, chunk_size=?, chunk_overlap=?, retrieval_mode=?, top_k=?, vector_weight=?, bm25_weight=? WHERE kb_id=?",
        (req.chunk_strategy, req.chunk_size, req.chunk_overlap, req.retrieval_mode,
         req.top_k, req.vector_weight, req.bm25_weight, kb_id),
    )
    conn.commit()
    conn.close()
    return {"code": 0, "data": None, "message": "Updated"}


@router.delete("/{kb_id}")
def delete_kb(kb_id: int, user: dict = Depends(require_role("admin"))):
    from app.db.vec_store import delete_chunks_by_kb
    delete_chunks_by_kb(kb_id)
    conn = get_connection()
    conn.execute("DELETE FROM kb_config WHERE kb_id=?", (kb_id,))
    conn.execute("DELETE FROM documents WHERE kb_id=?", (kb_id,))
    conn.execute("DELETE FROM knowledge_bases WHERE id=?", (kb_id,))
    conn.commit()
    conn.close()
    return {"code": 0, "data": None, "message": "Deleted"}
```

---

### 任务 15：API 路由 — 文档上传与管理

**涉及文件：**
- 新建：`backend/app/api/documents.py`

- [ ] **步骤 1：编写文档路由**

写入 `backend/app/api/documents.py`：
```python
import os
import uuid
from fastapi import APIRouter, UploadFile, File, Form, Depends, BackgroundTasks, HTTPException
from app.auth.permissions import get_current_user, require_role
from app.db.sqlite import get_connection
from app.config import settings

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    kb_id: int = Form(...),
    file: UploadFile = File(...),
    access_level: str = Form("internal"),
    user: dict = Depends(require_role("admin", "editor")),
):
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    ext = os.path.splitext(file.filename or "unknown")[1]
    stored_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(settings.UPLOAD_DIR, stored_name)

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    file_type = ext.lstrip(".")
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO documents (kb_id, filename, file_path, file_type, access_level, status) VALUES (?, ?, ?, ?, ?, 'pending')",
        (kb_id, file.filename, file_path, file_type, access_level),
    )
    doc_id = cur.lastrowid
    conn.commit()
    conn.close()

    from app.ingestion.pipeline import run_ingestion
    background_tasks.add_task(run_ingestion, doc_id)

    return {"code": 0, "data": {"id": doc_id, "status": "pending"}, "message": "Uploaded, processing in background"}


@router.get("/")
def list_documents(kb_id: int = None, user: dict = Depends(get_current_user)):
    conn = get_connection()
    if kb_id:
        rows = conn.execute("SELECT * FROM documents WHERE kb_id=? ORDER BY created_at DESC", (kb_id,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
    conn.close()
    return {"code": 0, "data": [dict(r) for r in rows], "message": "ok"}


@router.get("/{doc_id}/status")
def get_doc_status(doc_id: int, user: dict = Depends(get_current_user)):
    conn = get_connection()
    doc = conn.execute("SELECT id, filename, status FROM documents WHERE id=?", (doc_id,)).fetchone()
    conn.close()
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")
    return {"code": 0, "data": dict(doc), "message": "ok"}


@router.delete("/{doc_id}")
def delete_document(doc_id: int, user: dict = Depends(require_role("admin", "editor"))):
    from app.db.vec_store import delete_chunks_by_doc
    conn = get_connection()
    doc = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
    if not doc:
        conn.close()
        raise HTTPException(status_code=404, detail="Not found")

    delete_chunks_by_doc(doc_id)
    conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
    conn.commit()
    conn.close()

    if os.path.exists(doc["file_path"]):
        os.remove(doc["file_path"])
    return {"code": 0, "data": None, "message": "Deleted"}


@router.put("/{doc_id}/reprocess")
async def reprocess(doc_id: int, background_tasks: BackgroundTasks, user: dict = Depends(require_role("admin", "editor"))):
    from app.db.vec_store import delete_chunks_by_doc
    from app.ingestion.pipeline import run_ingestion

    conn = get_connection()
    doc = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
    if not doc:
        conn.close()
        raise HTTPException(status_code=404, detail="Not found")

    delete_chunks_by_doc(doc_id)
    conn.execute("UPDATE documents SET status='pending' WHERE id=?", (doc_id,))
    conn.commit()
    conn.close()

    background_tasks.add_task(run_ingestion, doc_id)
    return {"code": 0, "data": {"id": doc_id, "status": "pending"}, "message": "Reprocessing started"}
```

---

### 任务 16：API 路由 — 对话与反馈

**涉及文件：**
- 新建：`backend/app/api/chat.py`

- [ ] **步骤 1：编写对话路由**

写入 `backend/app/api/chat.py`：
```python
import json
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.auth.permissions import get_current_user
from app.core.interfaces import RetrievalContext, Message as CoreMessage
from app.core.intent import classify_intent
from app.core.retriever import HybridRetriever
from app.core.generator import Generator
from app.ingestion.embedder import OpenAIEmbedder
from app.llm.openai_llm import OpenAILLM
from app.db.sqlite import get_connection

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

embedder = OpenAIEmbedder()
llm = OpenAILLM()
retriever = HybridRetriever(embedder=embedder)
generator = Generator(llm=llm)


class ChatRequest(BaseModel):
    kb_id: int
    query: str
    history: list[dict] = []
    stream: bool = False


class FeedbackRequest(BaseModel):
    trace_id: str
    feedback: int
    feedback_text: str = ""


@router.post("/ask")
async def ask(req: ChatRequest, user: dict = Depends(get_current_user)):
    conn = get_connection()
    kb = conn.execute("SELECT * FROM knowledge_bases WHERE id=?", (req.kb_id,)).fetchone()
    config = conn.execute("SELECT * FROM kb_config WHERE kb_id=?", (req.kb_id,)).fetchone()
    conn.close()

    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    intent = classify_intent(req.query)

    ctx = RetrievalContext(
        kb_id=req.kb_id,
        kb_type=kb["kb_type"],
        user_departments=user["departments"],
        user_role=user["role"],
        access_levels=["public", "internal", "restricted"],
        top_k=config["top_k"] if config else 10,
        min_score=config["min_score"] if config else 0.0,
        retrieval_mode=config["retrieval_mode"] if config else "hybrid",
        vector_weight=intent["vector_weight"],
        bm25_weight=intent["bm25_weight"],
        rerank_enabled=bool(config["rerank_enabled"]) if config else True,
    )

    chunks = await retriever.retrieve(req.query, ctx)
    result = await generator.generate(req.query, chunks, ctx)

    conn = get_connection()
    conn.execute(
        "INSERT INTO audit_logs (user_id, kb_id, query_text, answer_text, sources, trace_id) VALUES (?, ?, ?, ?, ?, ?)",
        (user["id"], req.kb_id, req.query, result["answer"],
         json.dumps(result["sources"], ensure_ascii=False), result["trace_id"]),
    )
    conn.commit()
    conn.close()

    return {"code": 0, "data": result, "message": "ok"}


@router.post("/ask/stream")
async def ask_stream(req: ChatRequest, user: dict = Depends(get_current_user)):
    conn = get_connection()
    kb = conn.execute("SELECT * FROM knowledge_bases WHERE id=?", (req.kb_id,)).fetchone()
    config = conn.execute("SELECT * FROM kb_config WHERE kb_id=?", (req.kb_id,)).fetchone()
    conn.close()

    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    intent = classify_intent(req.query)
    ctx = RetrievalContext(
        kb_id=req.kb_id, kb_type=kb["kb_type"],
        user_departments=user["departments"], user_role=user["role"],
        access_levels=["public", "internal", "restricted"],
        top_k=config["top_k"] if config else 10,
        retrieval_mode=config["retrieval_mode"] if config else "hybrid",
        vector_weight=intent["vector_weight"],
        bm25_weight=intent["bm25_weight"],
    )

    chunks = await retriever.retrieve(req.query, ctx)

    async def event_stream():
        async for event in generator.generate_stream(req.query, chunks, ctx):
            yield f"data: {event}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/feedback")
def submit_feedback(req: FeedbackRequest, user: dict = Depends(get_current_user)):
    conn = get_connection()
    conn.execute(
        "UPDATE audit_logs SET feedback=?, feedback_text=? WHERE trace_id=? AND user_id=?",
        (req.feedback, req.feedback_text, req.trace_id, user["id"]),
    )
    conn.commit()
    conn.close()
    return {"code": 0, "data": None, "message": "Feedback recorded"}
```

---

### 任务 17：API 路由 — 审计、评估、管理

**涉及文件：**
- 新建：`backend/app/api/audit.py`
- 新建：`backend/app/api/evaluation.py`
- 新建：`backend/app/api/admin.py`

- [ ] **步骤 1：编写审计路由**

写入 `backend/app/api/audit.py`：
```python
from fastapi import APIRouter, Depends, Query
from app.auth.permissions import get_current_user, require_role
from app.db.sqlite import get_connection

router = APIRouter(prefix="/api/v1/audit", tags=["audit"])


@router.get("/logs")
def list_logs(
    kb_id: int | None = Query(None),
    user_id: int | None = Query(None),
    feedback: int | None = Query(None),
    limit: int = Query(50, le=200),
    offset: int = 0,
    user: dict = Depends(require_role("admin", "editor")),
):
    conn = get_connection()
    query = "SELECT * FROM audit_logs WHERE 1=1"
    params = []
    if kb_id is not None:
        query += " AND kb_id=?"; params.append(kb_id)
    if user_id is not None:
        query += " AND user_id=?"; params.append(user_id)
    if feedback is not None:
        query += " AND feedback=?"; params.append(feedback)
    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"; params.extend([limit, offset])
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return {"code": 0, "data": [dict(r) for r in rows], "message": "ok"}


@router.get("/logs/{log_id}")
def get_log(log_id: int, user: dict = Depends(require_role("admin", "editor"))):
    conn = get_connection()
    row = conn.execute("SELECT * FROM audit_logs WHERE id=?", (log_id,)).fetchone()
    conn.close()
    if not row:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Not found")
    return {"code": 0, "data": dict(row), "message": "ok"}
```

- [ ] **步骤 2：编写评估路由**

写入 `backend/app/api/evaluation.py`：
```python
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.auth.permissions import get_current_user, require_role
from app.db.sqlite import get_connection
import json

router = APIRouter(prefix="/api/v1/evaluation", tags=["evaluation"])


class EvalDataCreate(BaseModel):
    kb_id: int
    question: str
    reference_answer: str = ""
    relevant_doc_ids: list[int] = []


@router.post("/dataset")
def add_eval_data(req: EvalDataCreate, user: dict = Depends(require_role("admin", "editor"))):
    conn = get_connection()
    conn.execute(
        "INSERT INTO eval_dataset (kb_id, question, reference_answer, relevant_doc_ids, created_by) VALUES (?, ?, ?, ?, ?)",
        (req.kb_id, req.question, req.reference_answer, json.dumps(req.relevant_doc_ids), user["id"]),
    )
    conn.commit()
    conn.close()
    return {"code": 0, "data": None, "message": "Eval data added"}


@router.get("/dataset/{kb_id}")
def list_eval_data(kb_id: int, user: dict = Depends(get_current_user)):
    conn = get_connection()
    rows = conn.execute("SELECT * FROM eval_dataset WHERE kb_id=? ORDER BY created_at DESC", (kb_id,)).fetchall()
    conn.close()
    return {"code": 0, "data": [dict(r) for r in rows], "message": "ok"}


@router.post("/run/{kb_id}")
async def run_evaluation(kb_id: int, user: dict = Depends(require_role("admin", "editor"))):
    from app.evaluation.evaluator import Evaluator
    evaluator = Evaluator()
    report = await evaluator.run_full_eval(kb_id)
    return {"code": 0, "data": report, "message": "Evaluation complete"}


@router.get("/report/{kb_id}")
def get_report(kb_id: int, user: dict = Depends(get_current_user)):
    return {"code": 0, "data": {"message": "Run POST /run/{kb_id} to generate"}, "message": "ok"}
```

- [ ] **步骤 3：编写管理路由**

写入 `backend/app/api/admin.py`：
```python
import json
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.auth.permissions import get_current_user, require_role
from app.db.sqlite import get_connection

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


class ConfigUpdate(BaseModel):
    chunk_strategy: str | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    retrieval_mode: str | None = None
    top_k: int | None = None
    min_score: float | None = None
    rerank_enabled: bool | None = None
    vector_weight: float | None = None
    bm25_weight: float | None = None
    prompt_template: str | None = None


@router.get("/config/{kb_id}")
def get_config(kb_id: int, user: dict = Depends(require_role("admin", "editor"))):
    conn = get_connection()
    config = conn.execute("SELECT * FROM kb_config WHERE kb_id=?", (kb_id,)).fetchone()
    conn.close()
    if not config:
        raise HTTPException(status_code=404, detail="Not found")
    return {"code": 0, "data": dict(config), "message": "ok"}


@router.put("/config/{kb_id}")
def update_config(kb_id: int, req: ConfigUpdate, user: dict = Depends(require_role("admin"))):
    conn = get_connection()
    fields = {k: v for k, v in req.model_dump().items() if v is not None}
    if fields:
        set_clause = ", ".join(f"{k}=?" for k in fields)
        conn.execute(f"UPDATE kb_config SET {set_clause} WHERE kb_id=?", (*fields.values(), kb_id))
        conn.commit()
    conn.close()
    return {"code": 0, "data": None, "message": "Config updated"}


@router.get("/stats")
def get_stats(user: dict = Depends(require_role("admin", "editor"))):
    conn = get_connection()
    kb_count = conn.execute("SELECT COUNT(*) FROM knowledge_bases").fetchone()[0]
    doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    chunk_count = conn.execute("SELECT COUNT(*) FROM chunks_meta").fetchone()[0]
    query_count = conn.execute("SELECT COUNT(*) FROM audit_logs").fetchone()[0]
    feedback_up = conn.execute("SELECT COUNT(*) FROM audit_logs WHERE feedback=1").fetchone()[0]
    feedback_down = conn.execute("SELECT COUNT(*) FROM audit_logs WHERE feedback=-1").fetchone()[0]
    conn.close()
    return {"code": 0, "data": {
        "knowledge_bases": kb_count,
        "documents": doc_count,
        "chunks": chunk_count,
        "total_queries": query_count,
        "feedback_up": feedback_up,
        "feedback_down": feedback_down,
    }, "message": "ok"}
```

---

### 任务 18：评估引擎

**涉及文件：**
- 新建：`backend/app/evaluation/__init__.py`
- 新建：`backend/app/evaluation/evaluator.py`
- 新建：`backend/app/evaluation/metrics.py`
- 新建：`backend/app/evaluation/llm_judge.py`

- [ ] **步骤 1：编写指标计算**

写入 `backend/app/evaluation/metrics.py`：
```python
def recall_at_k(relevant_ids: list[int], retrieved_ids: list[int], k: int) -> float:
    if not relevant_ids:
        return 1.0
    top_k = set(retrieved_ids[:k])
    hits = top_k.intersection(set(relevant_ids))
    return len(hits) / len(relevant_ids)


def mrr(relevant_ids: list[int], retrieved_ids: list[int]) -> float:
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0
```

- [ ] **步骤 2：编写 LLM 评判器**

写入 `backend/app/evaluation/llm_judge.py`：
```python
from app.core.interfaces import Message
from app.llm.openai_llm import OpenAILLM


async def judge_faithfulness(answer: str, context: str) -> float:
    llm = OpenAILLM()
    result = await llm.generate([
        Message(role="system", content="""You are an evaluation judge. Score the answer's faithfulness to the context.
Score 1.0 = fully supported, 0.5 = partially, 0.0 = contradicts or unsupported.
Reply with only a number, like 0.85."""),
        Message(role="user", content=f"Context:\n{context}\n\nAnswer:\n{answer}\n\nScore:"),
    ])
    try:
        return float(result.content.strip())
    except ValueError:
        return 0.0


async def judge_relevance(answer: str, query: str) -> float:
    llm = OpenAILLM()
    result = await llm.generate([
        Message(role="system", content="""Score how relevant the answer is to the query.
1.0 = perfectly relevant, 0.0 = completely off-topic.
Reply with only a number, like 0.92."""),
        Message(role="user", content=f"Query: {query}\n\nAnswer: {answer}\n\nScore:"),
    ])
    try:
        return float(result.content.strip())
    except ValueError:
        return 0.0
```

- [ ] **步骤 3：编写评估器**

写入 `backend/app/evaluation/evaluator.py`：
```python
import json
from app.db.sqlite import get_connection
from app.core.interfaces import RetrievalContext
from app.core.retriever import HybridRetriever
from app.core.generator import Generator
from app.ingestion.embedder import OpenAIEmbedder
from app.llm.openai_llm import OpenAILLM
from app.evaluation.metrics import recall_at_k, mrr
from app.evaluation.llm_judge import judge_faithfulness, judge_relevance


class Evaluator:
    def __init__(self):
        self.embedder = OpenAIEmbedder()
        self.llm = OpenAILLM()
        self.retriever = HybridRetriever(embedder=self.embedder)
        self.generator = Generator(llm=self.llm)

    async def run_full_eval(self, kb_id: int) -> dict:
        conn = get_connection()
        kb = conn.execute("SELECT * FROM knowledge_bases WHERE id=?", (kb_id,)).fetchone()
        config = conn.execute("SELECT * FROM kb_config WHERE kb_id=?", (kb_id,)).fetchone()
        eval_items = conn.execute("SELECT * FROM eval_dataset WHERE kb_id=?", (kb_id,)).fetchall()
        conn.close()

        if not kb or not eval_items:
            return {"error": "No knowledge base or eval data"}

        retrieval_results = []
        generation_results = []

        for item in eval_items:
            relevant_ids = json.loads(item["relevant_doc_ids"] or "[]")
            ctx = RetrievalContext(
                kb_id=kb_id, kb_type=kb["kb_type"],
                access_levels=["public", "internal", "restricted"],
                top_k=config["top_k"] if config else 10,
                retrieval_mode=config["retrieval_mode"] if config else "hybrid",
            )
            chunks = await self.retriever.retrieve(item["question"], ctx)
            retrieved_ids = [c.doc_id for c in chunks]

            retrieval_results.append({
                "question": item["question"],
                "recall@5": recall_at_k(relevant_ids, retrieved_ids, 5),
                "recall@10": recall_at_k(relevant_ids, retrieved_ids, 10),
                "mrr": mrr(relevant_ids, retrieved_ids),
            })

            if item["reference_answer"]:
                result = await self.generator.generate(item["question"], chunks, ctx)
                context_text = "\n".join(c.text for c in chunks[:5])
                faith = await judge_faithfulness(result["answer"], context_text)
                relevance = await judge_relevance(result["answer"], item["question"])
                generation_results.append({
                    "question": item["question"],
                    "faithfulness": round(faith, 3),
                    "relevance": round(relevance, 3),
                })

        avg_recall5 = sum(r["recall@5"] for r in retrieval_results) / len(retrieval_results) if retrieval_results else 0
        avg_mrr = sum(r["mrr"] for r in retrieval_results) / len(retrieval_results) if retrieval_results else 0

        return {
            "kb_id": kb_id,
            "retrieval": {
                "avg_recall@5": round(avg_recall5, 4),
                "avg_mrr": round(avg_mrr, 4),
                "details": retrieval_results,
            },
            "generation": generation_results,
        }
```

---

### 任务 19：组装主应用与中间件

**涉及文件：**
- 修改：`backend/app/main.py`

- [ ] **步骤 1：更新 main.py 集成所有路由**

覆写 `backend/app/main.py`：
```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.db.sqlite import init_db
from app.db.vec_store import init_vec
from app.db.seed import seed_intent_rules
from app.api.auth import router as auth_router
from app.api.knowledge_bases import router as kb_router
from app.api.documents import router as doc_router
from app.api.chat import router as chat_router
from app.api.audit import router as audit_router
from app.api.evaluation import router as eval_router
from app.api.admin import router as admin_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    init_vec()
    seed_intent_rules()
    yield


app = FastAPI(title="Enterprise RAG API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(kb_router)
app.include_router(doc_router)
app.include_router(chat_router)
app.include_router(audit_router)
app.include_router(eval_router)
app.include_router(admin_router)


@app.get("/health")
def health():
    return {"status": "ok"}
```

- [ ] **步骤 2：添加管理员种子用户**

追加到 `backend/app/db/seed.py`：
```python
def seed_admin_user():
    conn = get_connection()
    count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    if count == 0:
        from app.auth.jwt import hash_password
        conn.execute(
            "INSERT INTO users (username, password_hash, role, departments) VALUES (?, ?, ?, ?)",
            ("admin", hash_password("admin123"), "admin", "[]"),
        )
        conn.commit()
    conn.close()
```

更新 main.py 的 lifespan 调用 `seed_admin_user()`。

- [ ] **步骤 3：验证完整启动**

执行：`cd backend && timeout 5 uvicorn app.main:app --port 8000 || true`
预期结果：服务器启动，无导入错误，在 `data/rag.db` 中创建数据库表。

- [ ] **步骤 4：冒烟测试 API**

执行：`cd backend && uvicorn app.main:app --port 8000 &`
然后：`curl -s http://localhost:8000/health`
预期输出：`{"status":"ok"}`

然后：`curl -s -X POST http://localhost:8000/api/v1/auth/login -H "Content-Type: application/json" -d '{"username":"admin","password":"admin123"}'`
预期输出：返回带 token 的 JSON

---

### 任务 20：最终集成测试

**涉及文件：**
- 新建：`backend/tests/test_integration.py`

- [ ] **步骤 1：编写端到端测试**

写入 `backend/tests/test_integration.py`：
```python
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.mark.asyncio
async def test_full_flow_login_create_kb_upload_ask():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/v1/auth/login", json={"username": "admin", "password": "admin123"})
        assert resp.status_code == 200
        token = resp.json()["data"]["token"]
        headers = {"Authorization": f"Bearer {token}"}

        resp = await client.post("/api/v1/knowledge-bases/", json={
            "name": "Test KB", "description": "Integration test", "kb_type": "employee",
            "chunk_strategy": "recursive", "chunk_size": 200, "chunk_overlap": 20,
        }, headers=headers)
        assert resp.status_code == 200
        kb_id = resp.json()["data"]["id"]

        resp = await client.get("/api/v1/knowledge-bases/", headers=headers)
        assert resp.status_code == 200
        assert len(resp.json()["data"]) >= 1

        import io
        resp = await client.post(
            "/api/v1/documents/upload",
            data={"kb_id": str(kb_id)},
            files={"file": ("test.txt", io.BytesIO(b"年假每年15天。\n申请年假需提前3天在OA提交。"))},
            headers=headers,
        )
        assert resp.status_code == 200
        doc_id = resp.json()["data"]["id"]

        import asyncio
        await asyncio.sleep(5)

        resp = await client.get(f"/api/v1/documents/{doc_id}/status", headers=headers)
        assert resp.json()["data"]["status"] == "completed"

        resp = await client.post("/api/v1/chat/ask", json={
            "kb_id": kb_id, "query": "年假有多少天",
        }, headers=headers)
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["answer"] != ""
        assert len(data["sources"]) > 0
        assert data["trace_id"] is not None

        resp = await client.get("/api/v1/admin/stats", headers=headers)
        assert resp.status_code == 200
        stats = resp.json()["data"]
        assert stats["knowledge_bases"] >= 1
        assert stats["documents"] >= 1
        assert stats["total_queries"] >= 1
```

- [ ] **步骤 2：运行集成测试**

执行：`cd backend && python -m pytest tests/test_integration.py::test_full_flow_login_create_kb_upload_ask -v`
预期结果：PASS

---

*计划版本: v1.1 | 日期: 2026-05-24*
