"""
Infinite Memory Inference API - PRODUCTION VERSION
Ready for deployment with real vLLM + ChromaDB
HARDCODED DEFAULTS - NO .env FILE NEEDED
"""

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal
import time
import os
from dotenv import load_dotenv
import uvicorn
import logging

# Load environment variables (optional now)
load_dotenv()

# Setup logging
os.makedirs("./data/logs", exist_ok=True)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.getenv("LOG_FILE", "./data/logs/api.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Request/Response Models
class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    conversation_id: Optional[str] = Field(default=None)
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False
    user: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict]
    usage: Dict


class InferenceAPI:
    """Production API with infinite memory"""

    def __init__(self, engine, api_keys: Optional[List[str]] = None,
                 rate_limit: int = 60):
        self.engine = engine
        self.api_keys = set(api_keys) if api_keys else None
        self.rate_limit = rate_limit
        self.request_counts = {}
        self.start_time = time.time()
        self.total_requests = 0
        self.total_tokens = 0

        self.app = FastAPI(
            title="Infinite Memory Inference API",
            description="vLLM with infinite conversation memory",
            version="1.0.0"
        )

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self._register_routes()

    def _register_routes(self):
        """Register API routes"""

        @self.app.get("/health")
        async def health():
            """Health check"""
            stats = self.engine.get_stats()
            return {
                "status": "healthy",
                "uptime_seconds": time.time() - self.start_time,
                "model": os.getenv("MODEL_NAME", "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"),
                "memory_stats": stats['memory_stats']
            }

        @self.app.post("/v1/chat/completions")
        async def chat_completions(
            request: ChatCompletionRequest,
            authorization: Optional[str] = Header(None)
        ):
            """OpenAI-compatible chat completions"""
            api_key = self._extract_api_key(authorization)
            if not self._verify_api_key(api_key):
                logger.warning(f"Invalid API key: {api_key}")
                raise HTTPException(status_code=401, detail="Invalid API key")

            if not self._check_rate_limit(api_key):
                logger.warning(f"Rate limit exceeded: {api_key}")
                raise HTTPException(status_code=429, detail="Rate limit exceeded")

            conversation_id = request.conversation_id or f"conv_{int(time.time())}_{api_key[:8] if api_key else 'anon'}"

            logger.info(f"Request: conv={conversation_id}, user={request.user or (api_key[:8] if api_key else 'anon')}")

            from vllm_wrapper_production import GenerationRequest

            gen_request = GenerationRequest(
                conversation_id=conversation_id,
                messages=[{'role': m.role, 'content': m.content} for m in request.messages],
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=request.stream,
                user_id=request.user or api_key
            )

            result = await self.engine.generate(gen_request)

            if not result['success']:
                logger.error(f"Generation failed: {result.get('error')}")
                raise HTTPException(status_code=500, detail=result.get('error'))

            self.total_requests += 1
            tokens = result['metadata']['tokens_generated']
            self.total_tokens += tokens

            logger.info(f"Response: conv={conversation_id}, tokens={tokens}, latency={result['metadata']['latency_ms']:.1f}ms")

            self._log_usage(api_key or 'anon', tokens, conversation_id)

            response = ChatCompletionResponse(
                id=f"chatcmpl-{int(time.time())}",
                created=int(time.time()),
                model=request.model,
                choices=[{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result['response']
                    },
                    "finish_reason": "stop"
                }],
                usage={
                    "prompt_tokens": 0,
                    "completion_tokens": tokens,
                    "total_tokens": tokens
                }
            )

            return response

        @self.app.get("/v1/models")
        async def list_models():
            """List models"""
            model_name = os.getenv("MODEL_NAME", "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ")
            return {
                "object": "list",
                "data": [{
                    "id": model_name,
                    "object": "model",
                    "created": int(self.start_time),
                    "owned_by": "infinite-memory"
                }]
            }

        @self.app.get("/metrics")
        async def metrics():
            """Detailed metrics"""
            stats = self.engine.get_stats()
            return {
                "api_stats": {
                    "uptime_seconds": time.time() - self.start_time,
                    "total_requests": self.total_requests,
                    "total_tokens": self.total_tokens,
                    "avg_tokens_per_request": (
                        self.total_tokens / self.total_requests
                        if self.total_requests > 0 else 0
                    )
                },
                **stats
            }

    def _extract_api_key(self, authorization: Optional[str]) -> Optional[str]:
        if not authorization:
            return None
        return authorization[7:] if authorization.startswith("Bearer ") else authorization

    def _verify_api_key(self, api_key: Optional[str]) -> bool:
        if self.api_keys is None:
            return True
        return api_key in self.api_keys

    def _check_rate_limit(self, api_key: str) -> bool:
        now = time.time()

        if api_key not in self.request_counts:
            self.request_counts[api_key] = (now, 1)
            return True

        last_reset, count = self.request_counts[api_key]

        if now - last_reset >= 60:
            self.request_counts[api_key] = (now, 1)
            return True

        if count >= self.rate_limit:
            return False

        self.request_counts[api_key] = (last_reset, count + 1)
        return True

    def _log_usage(self, api_key: str, tokens: int, conversation_id: str):
        logger.info(f"BILLING: key={api_key[:8] if len(api_key) >= 8 else api_key}, conv={conversation_id}, tokens={tokens}")


def create_app():
    """Create production app with HARDCODED DEFAULTS"""
    from vector_db_adapters import create_vector_db
    from memory_manager import MemoryManager
    from vllm_wrapper_production import InfiniteMemoryEngine, create_vllm_engine

    # HARDCODED DEFAULTS - CORRECT VALUES
    model_name = os.getenv("MODEL_NAME", "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ")
    quantization = os.getenv("MODEL_QUANTIZATION", "gptq")
    gpu_memory = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.85"))
    max_model_len = int(os.getenv("MAX_MODEL_LEN", "8192"))
    vector_db_backend = os.getenv("VECTOR_DB_BACKEND", "chromadb")
    cache_capacity = int(os.getenv("CACHE_CAPACITY", "1000"))
    api_keys_str = os.getenv("API_KEYS", "")
    api_keys = [k.strip() for k in api_keys_str.split(",") if k.strip()] if api_keys_str else None

    logger.info("="*80)
    logger.info("Infinite Memory Inference API")
    logger.info("="*80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Quantization: {quantization}")
    logger.info(f"GPU Memory: {gpu_memory * 100}%")
    logger.info(f"Vector DB: {vector_db_backend}")
    logger.info(f"Cache capacity: {cache_capacity}")
    logger.info(f"API keys: {'Enabled' if api_keys else 'Disabled (development mode)'}")
    logger.info("")

    logger.info("Initializing vector database...")
    vector_db = create_vector_db(backend=vector_db_backend)
    logger.info("✅ Vector DB ready")

    logger.info("Initializing memory manager...")
    memory_manager = MemoryManager(
        vector_db=vector_db,
        cache_capacity=cache_capacity,
        ttl_days=int(os.getenv("TTL_DAYS", "90"))
    )
    logger.info("✅ Memory manager ready")

    logger.info("Loading vLLM engine (this may take 30-60 seconds)...")
    vllm_engine = create_vllm_engine(
        model_name=model_name,
        quantization=quantization,
        gpu_memory_utilization=gpu_memory,
        max_model_len=max_model_len
    )
    logger.info("✅ vLLM engine ready")

    logger.info("Initializing infinite memory engine...")
    infinite_engine = InfiniteMemoryEngine(
        vllm_engine=vllm_engine,
        memory_manager=memory_manager,
        max_context_tokens=int(os.getenv("MAX_CONTEXT_TOKENS", "4096")),
        context_retrieval_k=int(os.getenv("CONTEXT_RETRIEVAL_K", "3"))
    )
    logger.info("✅ Infinite memory engine ready")

    logger.info("Initializing API server...")
    api = InferenceAPI(
        engine=infinite_engine,
        api_keys=api_keys,
        rate_limit=int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    )
    logger.info("✅ API server ready")

    logger.info("="*80)
    logger.info("🚀 Infinite Memory Inference API Started")
    logger.info("="*80)

    return api.app


if __name__ == "__main__":
    app = create_app()

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))

    print()
    print(f"API available at: http://{host}:{port}")
    print(f"Docs: http://{host}:{port}/docs")
    print()

    uvicorn.run(app, host=host, port=port)