"""
Health check and metrics endpoints
"""
from fastapi import APIRouter, Depends
from models.responses import HealthResponse, MetricsResponse
from api.dependencies import get_vllm_engine, get_vector_db, get_supabase
from services.cache_service import get_cache_service
from config.settings import get_settings
import time

router = APIRouter()
settings = get_settings()

# Track uptime
_start_time = time.time()
_total_requests = 0
_total_tokens = 0
_latencies = []


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    Returns service status and connectivity
    """
    try:
        # Check vLLM
        vllm_engine = get_vllm_engine()
        models_loaded = vllm_engine is not None
    except:
        models_loaded = False
    
    try:
        # Check vector DB
        vector_db = get_vector_db()
        # Simple ping
        vector_db_connected = vector_db is not None
    except:
        vector_db_connected = False
    
    uptime = time.time() - _start_time
    
    status = "healthy" if (models_loaded and vector_db_connected) else "degraded"
    
    return HealthResponse(
        status=status,
        version=settings.VERSION,
        uptime_seconds=round(uptime, 2),
        models_loaded=models_loaded,
        vector_db_connected=vector_db_connected
    )


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    cache_service = Depends(get_cache_service)
):
    """
    Metrics endpoint
    Returns usage statistics
    """
    uptime = time.time() - _start_time
    
    avg_latency = sum(_latencies) / len(_latencies) if _latencies else 0
    
    cache_stats = cache_service.get_stats()
    
    return MetricsResponse(
        total_requests=_total_requests,
        total_tokens=_total_tokens,
        avg_latency_ms=round(avg_latency, 2),
        cache_hit_rate=cache_stats['hit_rate_percent'] / 100,
        uptime_seconds=round(uptime, 2)
    )


def record_request_metrics(tokens: int, latency_ms: float):
    """Record metrics for a request"""
    global _total_requests, _total_tokens, _latencies
    
    _total_requests += 1
    _total_tokens += tokens
    _latencies.append(latency_ms)
    
    # Keep only last 1000 latencies
    if len(_latencies) > 1000:
        _latencies = _latencies[-1000:]
