"""
Shared dependencies for FastAPI routes
Dependency injection for services and resources
"""
from fastapi import Depends, HTTPException
from typing import Dict
from services.auth_service import get_auth_service, AuthService
from services.billing_service import get_billing_service, BillingService
from supabase import create_client, Client
from config.settings import get_settings
import time

settings = get_settings()

# Globals (initialized on startup)
_supabase_client: Client = None
_vllm_engine = None
_vector_db = None
_memory_manager = None


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
