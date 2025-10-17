"""
Shared dependencies for FastAPI routes
Dependency injection for services and resources
"""
from fastapi import Depends, HTTPException
from typing import Dict, Optional
from services.auth_service import get_auth_service, AuthService
from services.billing_service import get_billing_service, BillingService
from services.token_tracker import TokenTracker
from supabase import create_client, Client
from config.settings import get_settings
import redis
import time

settings = get_settings()

# Globals (initialized on startup)
_supabase_client: Client = None
_vllm_engine = None
_vector_db = None
_memory_manager = None
_redis_client: Optional[redis.Redis] = None
_token_tracker: Optional[TokenTracker] = None


def get_supabase() -> Client:
    """Get Supabase client"""
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_KEY
        )
    return _supabase_client


def get_current_user(
    auth_service: AuthService = Depends(get_auth_service)
) -> Dict:
    """
    Dependency to get current authenticated user
    Use in route: user: Dict = Depends(get_current_user)
    """
    # auth_service.validate_api_key already checks credentials
    # This is a wrapper for cleaner route signatures
    pass  # Actual auth happens in middleware


def get_vllm_engine():
    """Get vLLM engine instance (optional, may be None in proxy mode)"""
    global _vllm_engine
    return _vllm_engine


def get_vector_db():
    """Get vector database instance"""
    global _vector_db
    if _vector_db is None:
        raise HTTPException(500, "Vector DB not initialized")
    return _vector_db


def get_memory_manager():
    """Get memory manager instance"""
    global _memory_manager
    if _memory_manager is None:
        raise HTTPException(500, "Memory manager not initialized")
    return _memory_manager


def get_redis_client() -> Optional[redis.Redis]:
    """Get Redis client for token tracking"""
    global _redis_client
    if _redis_client is None:
        try:
            redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379/0')
            _redis_client = redis.from_url(redis_url, decode_responses=True)
            _redis_client.ping()
            print(f"✓ Connected to Redis at {redis_url}")
        except Exception as e:
            print(f"⚠️  Redis not available: {e}")
            _redis_client = None
    return _redis_client


def get_token_tracker() -> Optional[TokenTracker]:
    """Get token tracker with Redis"""
    global _token_tracker
    if _token_tracker is None:
        redis_client = get_redis_client()
        if redis_client:
            try:
                _token_tracker = TokenTracker(redis_client=redis_client)
                print("✓ Token tracker initialized with Redis")
            except Exception as e:
                print(f"⚠️  Token tracker initialization failed: {e}")
                _token_tracker = None
        else:
            print("⚠️  Token tracker not available (Redis not connected)")
            _token_tracker = None
    return _token_tracker


# Initialization functions (called from main.py startup)
def init_vllm_engine(engine):
    """Initialize vLLM engine"""
    global _vllm_engine
    _vllm_engine = engine


def init_vector_db(vector_db):
    """Initialize vector database"""
    global _vector_db
    _vector_db = vector_db


def init_memory_manager(memory_manager):
    """Initialize memory manager"""
    global _memory_manager
    _memory_manager = memory_manager
