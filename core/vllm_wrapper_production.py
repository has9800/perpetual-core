"""
vLLM Wrapper with Infinite Memory - V2 PRODUCTION VERSION
FEATURES: Nomic-Embed + BGE Reranker + Sliding Window + Token Budget
"""

import asyncio
from typing import List, Dict, Optional
import time
from dataclasses import dataclass

# V2 imports
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
                       gpu_memory_utilization: float = 0.9,
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
    """vLLM engine with infinite conversation memory (V2)"""

    def __init__(self,
                 vllm_engine,
                 memory_manager,
                 max_context_tokens: int = 4096,
                 context_retrieval_k: int = 3,
                 use_v2: bool = True):
        self.engine = vllm_engine
        self.memory = memory_manager
        self.max_context_tokens = max_context_tokens
        self.context_retrieval_k = context_retrieval_k
        
        # V2: Create context manager
        if use_v2:
            self.context_manager = ContextManagerV2(
                token_budget=2000,
                recent_turns_limit=15
            )
            print("✅ V2 Context Manager enabled (15-turn sliding window, 2000 token budget)")
        else:
            self.context_manager = None
            print("⚠️  V2 Context Manager disabled (using v1)")
        
        self.generation_count = 0
        self.total_tokens_generated = 0
        self.context_retrievals = 0

    async def generate(self, request: GenerationRequest) -> Dict:
        """Generate response with V2 memory management"""
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

            # V2: Check if we need to offload old turns
            if self.context_manager:
                turns_to_offload = self.context_manager.get_turns_for_offload(request.conversation_id)
                if turns_to_offload:
                    for turn in turns_to_offload:
                        self.memory.add_turn(
                            conversation_id=request.conversation_id,
                            text=turn['user'],
                            metadata={
                                'response': turn['assistant'],
                                'timestamp': turn.get('timestamp', time.time())
                            }
                        )
                    self.context_manager.prune_offloaded_turns(
                        request.conversation_id, 
                        len(turns_to_offload)
                    )

            # Retrieve context from vector DB
            context_result = self.memory.retrieve_context(
                conversation_id=request.conversation_id,
                query=user_query,
                top_k=self.context_retrieval_k
            )

            retrieved_context = context_result.get('results', [])
            self.context_retrievals += 1

            # V2: Build context with token budget
            if self.context_manager:
                formatted_context, context_meta = self.context_manager.get_context_for_llm(
                    conversation_id=request.conversation_id,
                    retrieved_context=retrieved_context
                )
                
                prompt = self._build_prompt_v2(
                    messages=request.messages,
                    formatted_context=formatted_context,
                    user_query=user_query
                )
            else:
                # Fallback: Old method
                context_meta = {
                    'total_tokens': 0,
                    'retrieved_turns': len(retrieved_context),
                    'recent_turns': 0,
                    'budget_utilization': 0
                }
                prompt = self._build_prompt(
                    messages=request.messages,
                    recent_turns=self.memory.get_recent_turns(request.conversation_id, limit=3),
                    retrieved_context=retrieved_context,
                    user_query=user_query
                )

            # Generate response
            response_text = await self._call_vllm(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )

            # V2: Store in context manager
            if self.context_manager:
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

            # Get conversation length
            if self.context_manager:
                conv_length = self.context_manager.get_recent_turns_count(request.conversation_id)
            else:
                conv_length = self.memory.get_conversation_length(request.conversation_id)

            return {
                'response': response_text,
                'conversation_id': request.conversation_id,
                'metadata': {
                    'latency_ms': latency,
                    'tokens_generated': tokens_generated,
                    'context_tokens': context_meta.get('total_tokens', 0),
                    'context_retrieved': context_meta.get('retrieved_turns', 0),
                    'recent_turns_used': context_meta.get('recent_turns', 0),
                    'budget_utilization': context_meta.get('budget_utilization', 0),
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

    def _build_prompt(self, 
                     messages: List[Dict],
                     recent_turns: List[str],
                     retrieved_context: List[Dict],
                     user_query: str) -> str:
        """Build prompt with retrieved context (V1 fallback)"""
        prompt_parts = []

        # Add system message if present
        system_msg = next((m['content'] for m in messages if m['role'] == 'system'), None)
        if system_msg:
            prompt_parts.append(f"System: {system_msg}")
            prompt_parts.append("")

        # Add retrieved relevant context (if high similarity)
        if retrieved_context:
            high_quality = [ctx for ctx in retrieved_context if ctx.get('similarity', 0) > 0.5]
            if high_quality:
                prompt_parts.append("# Relevant context from earlier in conversation:")
                for ctx in high_quality[:2]:
                    query = ctx['text']
                    response = ctx['metadata'].get('response', '')
                    if response:
                        prompt_parts.append(f"Previously - User: {query[:100]}...")
                        prompt_parts.append(f"Assistant: {response[:150]}...")
                prompt_parts.append("")

        # Add recent turns (last 3 exchanges)
        if recent_turns:
            prompt_parts.append("# Recent conversation:")
            for turn in recent_turns[-3:]:
                prompt_parts.append(turn)
            prompt_parts.append("")

        # Add current query
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
        
        # Add V2 stats if context manager exists
        if self.context_manager:
            stats['context_v2_stats'] = self.context_manager.get_stats()
        
        return stats
