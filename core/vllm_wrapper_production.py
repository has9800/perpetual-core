"""
vLLM Wrapper with Infinite Memory - PRODUCTION VERSION
SYNC VERSION: Reliable, simple, 175 tok/s performance
"""

import asyncio
from typing import List, Dict, Optional
import time
from dataclasses import dataclass
from context_manager_v2 import ContextManagerV2
from token_counter import count_tokens

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
                       gpu_memory_utilization: float = 0.9,  # 90% GPU for max performance
                       max_model_len: int = 4096):
    """Create real vLLM engine using high-level API"""
    from vllm import LLM

    print(f"Loading vLLM engine: {model_name}")
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

    print("✅ vLLM engine loaded successfully")
    return llm


class InfiniteMemoryEngine:
    """vLLM engine with infinite conversation memory (SYNC VERSION)"""

    def __init__(self,
                vllm_engine,
                memory_manager,
                max_context_tokens: int = 4096,
                context_retrieval_k: int = 3):
        self.engine = vllm_engine
        self.memory = memory_manager
        self.max_context_tokens = max_context_tokens
        self.context_retrieval_k = context_retrieval_k
        
        # V2: Add context manager
        self.context_manager = ContextManagerV2(
            token_budget=2000,  # 2K tokens for context
            recent_turns_limit=15
        )
        
        self.generation_count = 0
        self.total_tokens_generated = 0
        self.context_retrievals = 0


    async def generate(self, request: GenerationRequest) -> Dict:
        """Generate response with V2 memory management"""
        start_time = time.time()

        try:
            # Extract user query (last message)
            user_query = ""
            if request.messages:
                for msg in reversed(request.messages):
                    if msg['role'] == 'user':
                        user_query = msg['content']
                        break

            if not user_query:
                user_query = "continue"

            # V2: Offload older turns to vector DB (if needed)
            turns_to_offload = self.context_manager.get_turns_for_offload(request.conversation_id)
            if turns_to_offload:
                for turn in turns_to_offload:
                    self.memory.add_turn(
                        conversation_id=request.conversation_id,
                        text=turn['user'],
                        metadata={
                            'response': turn['assistant'],
                            'timestamp': turn['timestamp']
                        }
                    )
                # Prune from memory after successful offload
                self.context_manager.prune_offloaded_turns(
                    request.conversation_id, 
                    len(turns_to_offload)
                )

            # V2: Retrieve relevant context from vector DB (with reranking)
            context_result = self.memory.retrieve_context(
                conversation_id=request.conversation_id,
                query=user_query,
                top_k=20  # Get 20 candidates for reranking
            )

            retrieved_context = context_result.get('results', [])
            self.context_retrievals += 1

            # V2: Build optimized context with token budget
            formatted_context, context_meta = self.context_manager.get_context_for_llm(
                conversation_id=request.conversation_id,
                retrieved_context=retrieved_context
            )

            # Build final prompt
            prompt = self._build_prompt_v2(
                messages=request.messages,
                formatted_context=formatted_context,
                user_query=user_query
            )

            # Generate response
            response_text = await self._call_vllm(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )

            # V2: Store in context manager (in-memory)
            self.context_manager.add_turn(
                conversation_id=request.conversation_id,
                user_query=user_query,
                assistant_response=response_text,
                metadata={
                    'user_id': request.user_id,
                    'model': request.model
                }
            )

            # Update metrics
            self.generation_count += 1
            tokens_generated = len(response_text.split())
            self.total_tokens_generated += tokens_generated
            latency = (time.time() - start_time) * 1000

            return {
                'response': response_text,
                'conversation_id': request.conversation_id,
                'metadata': {
                    'latency_ms': latency,
                    'tokens_generated': tokens_generated,
                    'context_tokens': context_meta['total_tokens'],
                    'context_retrieved': context_meta['retrieved_turns'],
                    'recent_turns_used': context_meta['recent_turns'],
                    'budget_utilization': context_meta['budget_utilization'],
                    'conversation_turn': self.context_manager.get_recent_turns_count(request.conversation_id)
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


    async def _call_vllm(self, prompt: str, max_tokens: int, 
                        temperature: float) -> str:
        """Call vLLM using high-level API"""
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            stop=["</s>", "User:", "<|eot_id|>", "\n\nUser:", "Human:"]
        )

        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(
            None,
            self.engine.generate,
            [prompt],
            sampling_params
        )

        if outputs and len(outputs) > 0:
            return outputs[0].outputs[0].text.strip()
        return ""

    def _build_prompt_v2(self, 
                        messages: List[Dict],
                        formatted_context: str,
                        user_query: str) -> str:
        """Build prompt using V2 formatted context"""
        prompt_parts = []

        # Add system message if present
        system_msg = next((m['content'] for m in messages if m['role'] == 'system'), None)
        if system_msg:
            prompt_parts.append(f"System: {system_msg}")
            prompt_parts.append("")

        # Add V2 formatted context (already optimized with budget)
        if formatted_context:
            prompt_parts.append(formatted_context)
            prompt_parts.append("")

        # Add current query
        prompt_parts.append(f"User: {user_query}")
        prompt_parts.append("Assistant:")

        return "\n".join(prompt_parts)


    def get_stats(self) -> Dict:
        """Get statistics"""
        memory_stats = self.memory.get_metrics()
        context_stats = self.context_manager.get_stats()
        return {
            'engine_stats': {
                'total_generations': self.generation_count,
                'total_tokens_generated': self.total_tokens_generated,
                'context_retrievals': self.context_retrievals,
                'avg_tokens_per_generation': (
                    self.total_tokens_generated / self.generation_count
                    if self.generation_count > 0 else 0
                )
            },
            'memory_stats': memory_stats,
            'context_v2_stats': context_stats
        }
