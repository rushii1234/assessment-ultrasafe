import os
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/task1_db"
    
    # JWT
    SECRET_KEY: str = "your-super-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Third-Party LLM/Embedding API (USF)
    USF_API_KEY: str = ""
    USF_BASE_URL: str = "https://api.us.inc/usf/v1"
    USF_CHAT_MODEL: str = "usf1-mini"
    USF_EMBED_MODEL: str = "usf1-embed"
    USF_RERANK_MODEL: str = "usf1-rerank"
    USF_CHAT_TIMEOUT: int = 30
    USF_EMBED_TIMEOUT: int = 10
    USF_RERANK_TIMEOUT: int = 10
    USE_USF_API: bool = True  # Use USF API instead of local models
    
    # LLM (Legacy - for backwards compatibility)
    OPENAI_API_KEY: str = ""
    LLM_MODEL: str = "gpt-3.5-turbo"
    
    # Vector Database
    VECTOR_DB_PATH: str = "./chroma_db"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Application
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    APP_NAME: str = "Task 1 - FastAPI Chatbot with RAG"
    API_PREFIX: str = "/api/v1"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
