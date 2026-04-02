from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # API Configuration
    api_base_url: str = "https://api.us.inc/usf/v1"
    api_key: str = ""
    
    # Model Configuration
    chat_model: str = "usf1-mini"
    embedding_model: str = "usf1-embed"
    rerank_model: str = "usf1-rerank"
    
    # Chat Parameters
    temperature: float = Field(default=0.7, validation_alias="chat_temperature")
    max_tokens: int = Field(default=1000, validation_alias="chat_max_tokens")
    
    # FastAPI Settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    
    # RAG Settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_retrieval: int = 5
    rerank_top_k: int = 3
    
settings = Settings()
