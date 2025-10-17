"""
Main FastAPI application
Entry point for Perpetual AI API
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time

from config.settings import get_settings
from api.routes import chat, memory, health
from api.middleware.auth import AuthMiddleware
from api.middleware.rate_limit import RateLimitMiddleware
from api.middleware.request_logging import RequestLoggingMiddleware
from api.dependencies import (
    init_vllm_engine,
    init_vector_db,
    init_memory_manager
)
from core.llm_wrapper import create_vllm_engine
from core.vector_db import create_vector_db
from core.memory_manager import MemoryManager

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events
    Initialize models and connections
    """
    print("="*80)
    print(f"Starting {settings.APP_NAME} v{settings.VERSION}")
    print("="*80)
    
    # Initialize vLLM
    print("\n1. Initializing vLLM engine...")
    vllm_engine = create_vllm_engine(
        model_name=settings.MODEL_NAME,
        quantization=settings.MODEL_QUANTIZATION,
        gpu_memory_utilization=settings.GPU_MEMORY_UTILIZATION,
        max_model_len=settings.MAX_MODEL_LEN
    )
    init_vllm_engine(vllm_engine)
    
    # Initialize Vector DB
    print("\n2. Initializing vector database...")
    vector_db = create_vector_db(
        backend="qdrant",
        url=settings.QDRANT_CLOUD_URL,
        api_key=settings.QDRANT_API_KEY,
        collection_name=settings.QDRANT_COLLECTION,
        llm_engine=vllm_engine
    )
    init_vector_db(vector_db)
    
    # Initialize Memory Manager
    print("\n3. Initializing memory manager...")
    memory_manager = MemoryManager(
        vector_db=vector_db,
        cache_capacity=1000
    )
    init_memory_manager(memory_manager)
    
    print("\n" + "="*80)
    print(f"✅ {settings.APP_NAME} ready on http://{settings.API_HOST}:{settings.API_PORT}")
    print("="*80)
    
    yield
    
    # Shutdown
    print("\nShutting down...")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware (order matters!)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(AuthMiddleware)

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(chat.router, prefix=settings.API_PREFIX, tags=["chat"])
app.include_router(memory.router, prefix=settings.API_PREFIX, tags=["memory"])


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch all unhandled exceptions"""
    print(f"Unhandled exception: {exc}")
    import traceback
    traceback.print_exc()
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "An error occurred"
        }
    )


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.APP_NAME,
        "version": settings.VERSION,
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
