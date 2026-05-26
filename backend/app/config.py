import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY") or ""
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small"
    LLM_MODEL: str = os.getenv("LLM_MODEL") or "gpt-4o-mini"
    DATABASE_URL: str = os.getenv("DATABASE_URL") or f"sqlite:///{BASE_DIR}/data/rag.db"
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR") or str(BASE_DIR / "data" / "uploads")
    CACHE_DIR: str = os.getenv("CACHE_DIR") or str(BASE_DIR / "data" / "cache")
    SECRET_KEY: str = os.getenv("SECRET_KEY") or "change-me-in-production"
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM") or "HS256"
    JWT_EXPIRE_MINUTES: int = int(os.getenv("JWT_EXPIRE_MINUTES") or "1440")

    @property
    def sqlite_path(self) -> str:
        return self.DATABASE_URL.replace("sqlite:///", "")


settings = Settings()
