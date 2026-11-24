import os

# 加载 .env
# load_dotenv() 会从「本文件所在目录」(ragkb/config/) 逐级向上找 .env，找到就用。
from dotenv import load_dotenv

load_dotenv()


# Milvus 配置（可用环境变量覆盖）
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
DB_NAME = os.getenv("MILVUS_DB_NAME", "rag_implement")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "docs")
EMBED_DIM = 1024

# 文件配置
ENCODING = "utf-8"

# 切分配置（换长文档后从 200/80 调整为 400/60：技术文档段落更完整，冗余更少）
CHUNK_SIZE = 400
CHUNK_OVERLAP = 60

# Embedding 配置
EMBEDDING_MODEL= "BAAI/bge-m3"

# LLM 配置（密钥只从环境变量读，不写进代码）
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.8-flash")
LLM_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_BASE_URL = os.getenv("OPENAI_BASE_URL")
LLM_TEMPERATURE = 0

# 检索配置
TOP_K = int(os.getenv("TOP_K", "5"))

#拒答标准配置
REFUSE_THRESHOLD = float(os.getenv("REFUSE_THRESHOLD", "0.0"))

