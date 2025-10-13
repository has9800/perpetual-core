"""
vLLM Wrapper with Infinite Memory - PRODUCTION VERSION
Compatible with vLLM v0.10+
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
                       gpu_memory_utilization: float = 0.6,
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
    """vLLM engine with infinite conversation memory"""

    def __init__(self,
                 vllm_engine,
                 memory_manager,
                 max_context_tokens: int = 4096,
                 context_retrieval_k: int = 3):
        self.engine = vllm_engine
        self.memory = memory_manager
        self.max_context_tokens = max_context_tokens
        self.context_retrieval_k = context_retrieval_k
        self.generation_count = 0
        self.total_tokens_generated = 0
        self.context_retrievals = 0

    async def generate(self, request: GenerationRequest) -> Dict:
        """Generate response with automatic memory"""
        start_time = time.time()

        try:
            recent_turns = self.memory.get_recent_turns(
                request.conversation_id,
                limit=5
            )

            query = request.messages[-1]['content'] if request.messages else ""
            context_result = self.memory.retrieve_context(
                conversation_id=request.conversation_id,
                query=query,
                top_k=self.context_retrieval_k
            )

            retrieved_context = context_result.get('results', [])
            self.context_retrievals += 1

            prompt = self._build_prompt(
                messages=request.messages,
                recent_turns=recent_turns,
                retrieved_context=retrieved_context
            )

            response_text = await self._call_vllm(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )

            exchange_text = f"User: {query}\nAssistant: {response_text}"
            self.memory.add_turn(
                conversation_id=request.conversation_id,
                text=exchange_text,
                metadata={
                    'user_id': request.user_id,
                    'model': request.model,
                    'timestamp': time.time()
                }
            )

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
                    'context_retrieved': len(retrieved_context),
                    'recent_turns_used': len(recent_turns)
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
            stop=["</s>", "User:", "<|eot_id|>", "\n\nUser:"]
        )

        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(
            None,
            self.engine.generate,
            [prompt],
            sampling_params
        )

        if outputs and len(outputs) > 0:
            return outputs[0].outputs[0].text
        return ""

    def _build_prompt(self, 
                     messages: List[Dict],
                     recent_turns: List[str],
                     retrieved_context: List[Dict]) -> str:
        """Build prompt with context"""
        prompt_parts = []

        if retrieved_context:
            prompt_parts.append("# Relevant previous context:")
            for ctx in retrieved_context[:3]:
                prompt_parts.append(f"- {ctx['text'][:200]}")
            prompt_parts.append("")

        if recent_turns:
            prompt_parts.append("# Recent conversation:")
            for turn in recent_turns[-3:]:
                prompt_parts.append(turn)
            prompt_parts.append("")

        prompt_parts.append("# Current query:")
        for msg in messages:
            role = msg['role'].capitalize()
            content = msg['content']
            prompt_parts.append(f"{role}: {content}")

        prompt_parts.append("\nAssistant:")
        return "\n".join(prompt_parts)

    def get_stats(self) -> Dict:
        """Get statistics"""
        memory_stats = self.memory.get_metrics()
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
            'memory_stats': memory_stats
        }