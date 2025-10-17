"""
Configuration management using Pydantic
All settings loaded from environment variables
"""
from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings from environment"""
    
    # App
    APP_NAME: str = "Perpetual AI"
    VERSION: str = "2.0.0"
    DEBUG: bool = False
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_PREFIX: str = "/v1"
    
    # Model
    MODEL_NAME: str = "mistralai/Mistral-7B-Instruct-v0.2"
    MODEL_QUANTIZATION: str = "gptq"
    GPU_MEMORY_UTILIZATION: float = 0.85
    MAX_MODEL_LEN: int = 4096
    
    # Vector DB
    QDRANT_CLOUD_URL: Optional[str] = None
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION: str = "conversations"
    
    # Supabase Auth
    SUPABASE_URL: str
    SUPABASE_KEY: str
    SUPABASE_JWT_SECRET: str
    
    # Polar.sh Billing
    POLAR_API_KEY: str
    POLAR_ORGANIZATION_ID: str
    
    # Redis Cache
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_DAY: int = 10000
    
    # Retrieval
    RETRIEVAL_TOP_K: int = 5
    CONTEXT_MAX_TOKENS: int = 4096
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
