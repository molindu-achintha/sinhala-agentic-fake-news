"""
Configuration settings for the application.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    APP_NAME: str = "Sinhala Agentic Fake News Detection"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/v1"
    
    # Model Paths
    MODEL_PATH: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    EMBEDDINGS_PATH: str = "data/embeddings"
    FAISS_INDEX_PATH: str = "data/embeddings/faiss_index.bin"
    
    # Database
    DATABASE_URL: str = "sqlite:///./sql_app.db"
    
    # API Keys
    HF_API_KEY: str | None = None
    OPENROUTER_API_KEY: str | None = None
    
    # Embedding Model (OpenRouter compatible)
    EMBEDDING_MODEL: str = "openai/text-embedding-3-small"

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
