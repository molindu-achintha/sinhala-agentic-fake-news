"""
config.py - Application Configuration

This module defines all configuration settings for the application.
Settings are loaded from environment variables or .env file.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path

# Calculate .env path at module level (project root / .env)
_ENV_FILE_PATH = Path(__file__).resolve().parent.parent.parent / ".env"


class Settings(BaseSettings):
    """
    Application settings.
    
    All settings can be overridden by environment variables.
    For example, set OPENROUTER_API_KEY in .env file.
    """
    
    # Application Info
    APP_NAME: str = "Sinhala Agentic Fake News Detection"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/v1"
    
    # Model Paths (for local FAISS, not used with Pinecone)
    MODEL_PATH: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    EMBEDDINGS_PATH: str = "data/embeddings"
    FAISS_INDEX_PATH: str = "data/embeddings/faiss_index.bin"
    
    # Database (not used in current implementation)
    DATABASE_URL: str = "sqlite:///./sql_app.db"
    
    # API Keys - Set these in .env file
    HF_API_KEY: str | None = None
    OPENROUTER_API_KEY: str | None = None
    GROQ_API_KEY: str | None = None
    PINECONE_API_KEY: str | None = None
    
    # Pinecone Configuration
    PINECONE_INDEX_NAME: str = "news-store"
    
    # Embedding Configuration
    # EMBEDDING_PROVIDER options: "openrouter", "pinecone", "auto"
    # "auto" = try openrouter first, then pinecone
    EMBEDDING_PROVIDER: str = "auto"
    
    # Embedding Model (via OpenRouter)
    # Using multilingual-e5-large (1024 dimensions) - supports Sinhala
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"
    EMBEDDING_DIMENSION: int = 1024

    class Config:
        """Configuration for settings loading."""
        env_file = str(_ENV_FILE_PATH)
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra fields in .env


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings.
    
    Uses lru_cache to ensure settings are only loaded once.
    
    Returns:
        Settings instance
    """
    print("[config] Loading settings")
    settings = Settings()
    print("[config] App name:", settings.APP_NAME)
    print("[config] Version:", settings.VERSION)
    print("[config] Pinecone index:", settings.PINECONE_INDEX_NAME)
    print("[config] Embedding model:", settings.EMBEDDING_MODEL)
    return settings
