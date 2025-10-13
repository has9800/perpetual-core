"""
GPU Benchmark Suite for Infinite Memory Inference API
FIXED: Better error handling, debug output, handles empty results
"""

import asyncio
import time
import numpy as np
from typing import List, Dict
import sys
import os

# Add core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from vector_db_adapters import create_vector_db
from memory_manager import MemoryManager
from vllm_wrapper_production import InfiniteMemoryEngine, create_vllm_engine, GenerationRequest


class ThroughputBenchmark:
    """Benchmark throughput and latency at different conversation lengths"""

    def __init__(self, engine, memory):
        self.engine = engine
        self.memory = memory

    async def run(self) -> Dict:
        """Run throughput benchmark"""
        print("\n" + "="*80)
        print("BENCHMARK 1: THROUGHPUT & LATENCY")
        print("Verify throughput stays constant as conversation grows")
        print("="*80)

        results = {}
        turn_counts = [10, 50, 100]

        for turns in turn_counts:
            print(f"\nLatency for {turns} turns:")
            latencies = []
            throughputs = []

            conv_id = f"throughput_test_{turns}"

            for i in range(turns):
                start = time.time()

                request = GenerationRequest(
                    conversation_id=conv_id,
                    messages=[{"role": "user", "content": f"Message {i+1}: Tell me a short fact."}],
                    model="test",
                    max_tokens=50,
                    temperature=0.7
                )

                result = await self.engine.generate(request)

                if result.get('success'):
                    latency = (time.time() - start) * 1000
                    tokens = result['metadata']['tokens_generated']
                    throughput = tokens / (latency / 1000)

                    latencies.append(latency)
                    throughputs.append(throughput)

            if latencies:
                results[f'{turns}_turns'] = {
                    'avg_latency': np.mean(latencies),
                    'p50_latency': np.percentile(latencies, 50),
                    'p95_latency': np.percentile(latencies, 95),
                    'p99_latency': np.percentile(latencies, 99),
                    'avg_throughput': np.mean(throughputs),
                    'min_throughput': np.min(throughputs),
                    'max_throughput': np.max(throughputs)
                }

                print(f"RESULTS:")
                print(f"  Avg latency: {results[f'{turns}_turns']['avg_latency']:.0f}ms")
                print(f"  P50 latency: {results[f'{turns}_turns']['p50_latency']:.0f}ms")
                print(f"  P95 latency: {results[f'{turns}_turns']['p95_latency']:.0f}ms")
                print(f"  P99 latency: {results[f'{turns}_turns']['p99_latency']:.0f}ms")
                print(f"  Avg throughput: {results[f'{turns}_turns']['avg_throughput']:.1f} tokens/sec")
                print(f"  Range: {results[f'{turns}_turns']['min_throughput']:.1f} - {results[f'{turns}_turns']['max_throughput']:.1f} tok/s")

        return results


class MemoryEfficiencyBenchmark:
    """Benchmark memory efficiency vs traditional approach"""

    def __init__(self, engine, memory):
        self.engine = engine
        self.memory = memory

    async def run(self) -> Dict:
        """Run memory efficiency benchmark"""
        print("\n" + "="*80)
        print("BENCHMARK 2: MEMORY EFFICIENCY")
        print("Verify we send only relevant context, not full history")
        print("="*80)
        print("Building conversation and measuring prompt sizes...")

        conv_id = "memory_test"
        prompt_sizes = []

        # Build 100 turn conversation
        for i in range(100):
            request = GenerationRequest(
                conversation_id=conv_id,
                messages=[{"role": "user", "content": f"Turn {i+1}: Say hello."}],
                model="test",
                max_tokens=30,
                temperature=0.7
            )

            result = await self.engine.generate(request)

            if result.get('success'):
                # Estimate prompt size (recent + retrieved context)
                recent = self.memory.get_recent_turns(conv_id, limit=3)
                context = self.memory.retrieve_context(conv_id, f"Turn {i+1}", top_k=3)

                # Count tokens (rough estimate: ~4 chars per token)
                prompt_size = sum(len(t) for t in recent) // 4
                prompt_size += sum(len(r['text']) for r in context.get('results', [])) // 4

                prompt_sizes.append(prompt_size)

        if not prompt_sizes:
            print("\n❌ ERROR: No prompt sizes collected")
            return {'error': 'No data'}

        # Calculate traditional approach (full history)
        traditional_size = sum(range(1, 101)) * 20  # Assume 20 tokens per turn average
        our_size = prompt_sizes[-1] if prompt_sizes else 0

        savings = ((traditional_size - our_size) / traditional_size * 100) if traditional_size > 0 else 0

        # Check prompt growth
        first_half_avg = np.mean(prompt_sizes[:50]) if len(prompt_sizes) >= 50 else 0
        second_half_avg = np.mean(prompt_sizes[50:]) if len(prompt_sizes) > 50 else 0
        growth = ((second_half_avg - first_half_avg) / first_half_avg * 100) if first_half_avg > 0 else 0

        results = {
            'traditional_tokens': traditional_size,
            'our_tokens': our_size,
            'savings_percent': savings,
            'prompt_growth_percent': growth
        }

        print(f"\nRESULTS:")
        print(f"  At turn 100:")
        print(f"    Traditional: {traditional_size} tokens")
        print(f"    Our system: {our_size} tokens")
        print(f"    Savings: {savings:.1f}%")
        print(f"  Prompt growth: {growth:.1f}% (should be near 0%)")

        return results


class RetrievalAccuracyBenchmark:
    """Benchmark retrieval accuracy"""

    def __init__(self, engine, memory):
        self.engine = engine
        self.memory = memory

    async def run(self) -> Dict:
        """Run retrieval accuracy benchmark"""
        print("\n" + "="*80)
        print("BENCHMARK 3: RETRIEVAL ACCURACY")
        print("Verify semantic search finds relevant past context")
        print("="*80)
        print("Testing retrieval accuracy...")

        conv_id = "accuracy_test"
        test_pairs = [
            ("What is Python?", "Python is a programming language"),
            ("Tell me about dogs", "Dogs are loyal pets"),
            ("Explain machine learning", "ML is about training models"),
            ("What is the weather", "Weather refers to atmospheric conditions"),
            ("How to cook pasta", "Pasta is cooked in boiling water")
        ]

        # Store test data
        print(f"\nStoring {len(test_pairs)} test exchanges...")
        for i, (query, response) in enumerate(test_pairs):
            self.memory.add_turn(
                conversation_id=conv_id,
                text=query,
                metadata={'response': response, 'test_index': i}
            )

        # Give Qdrant time to index (important for local Qdrant)
        await asyncio.sleep(1)

        print(f"Stored data. Conversation has {self.memory.get_conversation_length(conv_id)} turns")

        # Test retrieval
        correct = 0
        similarities = []

        print(f"\nTesting retrieval for each query...")
        for i, (query, expected_response) in enumerate(test_pairs):
            context = self.memory.retrieve_context(conv_id, query, top_k=3)

            print(f"  Query {i+1}: '{query[:30]}...'")
            print(f"    Retrieved: {len(context.get('results', []))} results")

            if context.get('success') and context.get('results'):
                results = context['results']

                # Check if expected response is in top result
                if results:
                    top_result = results[0]
                    top_response = top_result['metadata'].get('response', '')
                    similarity = top_result.get('similarity', 0)
                    similarities.append(similarity)

                    print(f"    Top match: '{top_result['text'][:30]}...'")
                    print(f"    Similarity: {similarity:.3f}")

                    if expected_response in top_response or top_response in expected_response:
                        correct += 1
                        print(f"    ✅ Correct match!")
                    else:
                        print(f"    ❌ Wrong match (expected: '{expected_response[:30]}...')")
                else:
                    print(f"    ❌ No results returned")
            else:
                print(f"    ❌ Retrieval failed: {context.get('error', 'Unknown error')}")

        accuracy = (correct / len(test_pairs) * 100) if test_pairs else 0

        results = {
            'accuracy_percent': accuracy,
            'correct': correct,
            'total': len(test_pairs),
            'avg_similarity': np.mean(similarities) if similarities else 0,
            'min_similarity': np.min(similarities) if similarities else 0,
            'max_similarity': np.max(similarities) if similarities else 0
        }

        print(f"\nRESULTS:")
        print(f"  Accuracy: {accuracy:.1f}% ({correct}/{len(test_pairs)})")
        print(f"  Avg similarity: {results['avg_similarity']:.3f}")

        if similarities:
            print(f"  Min similarity: {results['min_similarity']:.3f}")
            print(f"  Max similarity: {results['max_similarity']:.3f}")
        else:
            print(f"  Min similarity: N/A (no results)")
            print(f"  Max similarity: N/A (no results)")

        return results


async def run_all_benchmarks():
    """Run all benchmarks"""
    print("="*80)
    print("INFINITE MEMORY INFERENCE API - GPU BENCHMARK SUITE")
    print("="*80)
    print("\nStarting comprehensive GPU benchmark suite...")
    print("This will take approximately 10-15 minutes")

    # System info
    try:
        import torch
        import subprocess

        print("\nSYSTEM INFORMATION")
        print("-"*80)

        gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', 
                                           '--format=csv,noheader,nounits']).decode().strip()
        print(f"GPU: {gpu_info}")

        print(f"CPU cores: {os.cpu_count()}")

        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        print(f"RAM: {ram_gb:.1f} GB")

        print(f"Python: {sys.version.split()[0]}")
        print()
    except:
        pass

    # Initialize system
    print("Initializing system...\n")

    # Create vector DB (use env var for backend)
    vector_db_backend = os.getenv("VECTOR_DB_BACKEND", "qdrant")
    print(f"Using vector DB: {vector_db_backend.upper()}")
    vector_db = create_vector_db(backend=vector_db_backend)

    # Create memory manager
    memory_manager = MemoryManager(
        vector_db=vector_db,
        cache_capacity=1000
    )

    # Create vLLM engine
    model_name = os.getenv("MODEL_NAME", "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ")
    quantization = os.getenv("MODEL_QUANTIZATION", "gptq")
    gpu_memory = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9"))

    vllm_engine = create_vllm_engine(
        model_name=model_name,
        quantization=quantization,
        gpu_memory_utilization=gpu_memory,
        max_model_len=4096
    )

    # Create infinite memory engine
    infinite_engine = InfiniteMemoryEngine(
        vllm_engine=vllm_engine,
        memory_manager=memory_manager,
        max_context_tokens=4096,
        context_retrieval_k=3
    )

    print("✅ System initialized\n")

    # Run benchmarks
    all_results = {}

    try:
        # Benchmark 1: Throughput
        throughput_bench = ThroughputBenchmark(infinite_engine, memory_manager)
        all_results['throughput'] = await throughput_bench.run()
    except Exception as e:
        print(f"\n❌ Throughput benchmark failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        # Benchmark 2: Memory Efficiency
        memory_bench = MemoryEfficiencyBenchmark(infinite_engine, memory_manager)
        all_results['memory_efficiency'] = await memory_bench.run()
    except Exception as e:
        print(f"\n❌ Memory efficiency benchmark failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        # Benchmark 3: Retrieval Accuracy
        accuracy_bench = RetrievalAccuracyBenchmark(infinite_engine, memory_manager)
        all_results['accuracy'] = await accuracy_bench.run()
    except Exception as e:
        print(f"\n❌ Retrieval accuracy benchmark failed: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)

    if 'throughput' in all_results and '10_turns' in all_results['throughput']:
        print(f"\nThroughput (10 turns): {all_results['throughput']['10_turns']['avg_throughput']:.1f} tok/s")
        print(f"Latency (P50): {all_results['throughput']['10_turns']['p50_latency']:.0f}ms")

    if 'memory_efficiency' in all_results and 'savings_percent' in all_results['memory_efficiency']:
        print(f"\nMemory savings: {all_results['memory_efficiency']['savings_percent']:.1f}%")
        print(f"Prompt growth: {all_results['memory_efficiency']['prompt_growth_percent']:.1f}%")

    if 'accuracy' in all_results and 'accuracy_percent' in all_results['accuracy']:
        print(f"\nRetrieval accuracy: {all_results['accuracy']['accuracy_percent']:.1f}%")
        print(f"Avg similarity: {all_results['accuracy']['avg_similarity']:.3f}")

    print("\n" + "="*80)
    print("✅ BENCHMARK COMPLETE")
    print("="*80)

    return all_results


if __name__ == "__main__":
    results = asyncio.run(run_all_benchmarks())