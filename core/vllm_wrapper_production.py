"""
vLLM Wrapper with Simple Memory - PRODUCTION
Simple V2: 15-turn sliding window + Qdrant long-term + smart reranking
"""

import asyncio
from typing import List, Dict, Optional
import time
from dataclasses import dataclass


@dataclass
class GenerationRequest:
    """Request for text generation"""
    conversation_id: str
    messages: List[Dict[str, str]]
    model: str
    max_tokens: int = 1024
    temperature: float = 0.7
    stream: bool = False
    user_id: Optional[str] = None


def create_vllm_engine(model_name: str,
                       quantization: str = "gptq",
                       gpu_memory_utilization: float = 0.9,
                       max_model_len: int = 4096):
    """Create vLLM engine"""
    from vllm import LLM

    print(f"Loading vLLM: {model_name}")
    print(f"  Quantization: {quantization}")
    print(f"  Max length: {max_model_len}")
    print(f"  GPU memory: {gpu_memory_utilization * 100}%")

    llm = LLM(
        model=model_name,
        quantization=quantization if quantization else None,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True,
        dtype="float16"
    )

    print("✅ vLLM loaded")
    return llm


class InfiniteMemoryEngine:
    """vLLM with simple short-term + long-term memory"""

    def __init__(self,
                 vllm_engine,
                 memory_manager,
                 max_context_tokens: int = 4096,
                 context_retrieval_k: int = 3,
                 use_simple_memory: bool = True):
        self.engine = vllm_engine
        self.memory = memory_manager
        self.max_context_tokens = max_context_tokens
        self.context_retrieval_k = context_retrieval_k
        
        # Simple context manager
        if use_simple_memory:
            from context_manager import SimpleContextManager
            self.context_manager = SimpleContextManager(short_term_limit=15)
            print("✅ Simple Memory enabled: 15-turn sliding window + Qdrant")
        else:
            self.context_manager = None
            print("⚠️  Simple Memory disabled")
        
        self.generation_count = 0
        self.total_tokens_generated = 0
        self.context_retrievals = 0

    async def generate(self, request: GenerationRequest) -> Dict:
        """Generate with simple memory"""
        start_time = time.time()

        try:
            # Extract user query
            user_query = ""
            if request.messages:
                for msg in reversed(request.messages):
                    if msg['role'] == 'user':
                        user_query = msg['content']
                        break

            if not user_query:
                user_query = "continue"

            # Retrieve from long-term memory (Qdrant with smart reranking)
            context_result = self.memory.retrieve_context(
                conversation_id=request.conversation_id,
                query=user_query,
                top_k=self.context_retrieval_k
            )

            retrieved_long_term = context_result.get('results', [])
            self.context_retrievals += 1

            # Build context: short-term (15 turns) + long-term (retrieved)
            if self.context_manager:
                formatted_context, context_meta = self.context_manager.build_context(
                    conversation_id=request.conversation_id,
                    retrieved_long_term=retrieved_long_term
                )
                
                prompt = self._build_prompt(
                    messages=request.messages,
                    formatted_context=formatted_context,
                    user_query=user_query
                )
            else:
                # Fallback
                formatted_context = ""
                context_meta = {'short_term_turns': 0, 'long_term_retrieved': 0}
                prompt = f"User: {user_query}\nAssistant:"

            # Generate
            response_text = await self._call_vllm(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )

            # Store: short-term (sliding window) + long-term (Qdrant)
            if self.context_manager:
                # Add to short-term (auto-evicts oldest if > 15)
                self.context_manager.add_turn(
                    conversation_id=request.conversation_id,
                    user_query=user_query,
                    assistant_response=response_text,
                    metadata={'user_id': request.user_id, 'model': request.model}
                )
                
                # Also store in long-term for future retrieval
                self.memory.add_turn(
                    conversation_id=request.conversation_id,
                    text=user_query,
                    metadata={
                        'response': response_text, 
                        'timestamp': time.time(),
                        'turn_number': self.generation_count
                    }
                )

            # Metrics
            self.generation_count += 1
            tokens_generated = len(response_text.split())
            self.total_tokens_generated += tokens_generated
            latency = (time.time() - start_time) * 1000

            conv_length = self.context_manager.get_turn_count(request.conversation_id) if self.context_manager else 0

            return {
                'response': response_text,
                'conversation_id': request.conversation_id,
                'metadata': {
                    'latency_ms': latency,
                    'tokens_generated': tokens_generated,
                    'short_term_turns': context_meta.get('short_term_turns', 0),
                    'long_term_retrieved': context_meta.get('long_term_retrieved', 0),
                    'conversation_turn': conv_length
                },
                'success': True
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'conversation_id': request.conversation_id
            }

    async def _call_vllm(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Call vLLM"""
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            stop=["</s>", "User:", "<|eot_id|>", "\n\nUser:", "Human:"]
        )

        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(None, self.engine.generate, [prompt], sampling_params)

        if outputs and len(outputs) > 0:
            return outputs[0].outputs[0].text.strip()
        return ""

    def _build_prompt(self, messages: List[Dict], formatted_context: str, user_query: str) -> str:
        """Build prompt"""
        prompt_parts = []

        # System message
        system_msg = next((m['content'] for m in messages if m['role'] == 'system'), None)
        if system_msg:
            prompt_parts.append(f"System: {system_msg}")
            prompt_parts.append("")

        # Context (short-term + long-term)
        if formatted_context:
            prompt_parts.append(formatted_context)

        # Current query
        prompt_parts.append(f"User: {user_query}")
        prompt_parts.append("Assistant:")

        return "\n".join(prompt_parts)

    def get_stats(self) -> Dict:
        """Get statistics"""
        memory_stats = self.memory.get_metrics()
        
        stats = {
            'engine_stats': {
                'total_generations': self.generation_count,
                'total_tokens_generated': self.total_tokens_generated,
                'context_retrievals': self.context_retrievals,
                'avg_tokens_per_generation': (
                    self.total_tokens_generated / self.generation_count
                    if self.generation_count > 0 else 0
                )
            },
            'memory_stats': memory_stats
        }
        
        if self.context_manager:
            stats['simple_memory_stats'] = self.context_manager.get_stats()
        
        return stats
