"""
vLLM wrapper with cleaner interface
"""
from vllm import LLM, SamplingParams
from typing import List, Dict, Optional
import time


class VLLMEngine:
    """Wrapper for vLLM with cleaner interface"""
    
    def __init__(
        self,
        model_name: str,
        quantization: Optional[str] = None,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
        tensor_parallel_size: int = 1
    ):
        """
        Initialize vLLM engine
        
        Args:
            model_name: HuggingFace model name
            quantization: Quantization method (gptq, awq, None)
            gpu_memory_utilization: GPU memory to use (0.0-1.0)
            max_model_len: Max context length
            tensor_parallel_size: Number of GPUs for tensor parallelism
        """
        print(f"Loading vLLM model: {model_name}")
        
        self.model_name = model_name
        self.max_model_len = max_model_len
        
        # Initialize vLLM
        self.llm = LLM(
            model=model_name,
            quantization=quantization,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True
        )
        
        print(f"✅ vLLM loaded: {model_name}")
    
    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None
    ) -> List:
        """
        Generate completions
        
        Args:
            prompts: List of prompts
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling
            stop: Stop sequences
            
        Returns:
            List of generated outputs
        """
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop
        )
        
        outputs = self.llm.generate(prompts, sampling_params)
        return outputs
    
    async def generate_async(
        self,
        prompts: List[str],
        **kwargs
    ) -> List:
        """Async generation (uses threading internally)"""
        # vLLM doesn't have true async, but we can wrap it
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate(prompts, **kwargs)
        )


def create_vllm_engine(
    model_name: str,
    quantization: Optional[str] = None,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 4096
) -> VLLMEngine:
    """Factory function to create vLLM engine"""
    return VLLMEngine(
        model_name=model_name,
        quantization=quantization,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len
    )
