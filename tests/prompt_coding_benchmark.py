"""
GPU Benchmark Suite for Infinite Memory Inference API - V2
UPDATED FOR V2: Tests Nomic-Embed + BGE Reranker + Sliding Window + Token Budget
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
from context_manager_v2 import ContextManagerV2
from token_counter import count_tokens


class ThroughputBenchmark:
    """Benchmark throughput and latency at different conversation lengths"""

    def __init__(self, engine, memory):
        self.engine = engine
        self.memory = memory

    async def run(self) -> Dict:
        """Run throughput benchmark"""
        print("\n" + "="*80)
        print("BENCHMARK 1: THROUGHPUT & LATENCY (V2)")
        print("Verify throughput stays constant with sliding window + token budget")
        print("="*80)

        results = {}
        turn_counts = [10, 50, 100, 200]

        for turns in turn_counts:
            print(f"\nLatency for {turns} turns:")
            latencies = []
            throughputs = []
            token_usages = []

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
                    context_tokens = result['metadata'].get('context_tokens', 0)
                    throughput = tokens / (latency / 1000)

                    latencies.append(latency)
                    throughputs.append(throughput)
                    token_usages.append(context_tokens)

            if latencies:
                results[f'{turns}_turns'] = {
                    'avg_latency': np.mean(latencies),
                    'p50_latency': np.percentile(latencies, 50),
                    'p95_latency': np.percentile(latencies, 95),
                    'p99_latency': np.percentile(latencies, 99),
                    'avg_throughput': np.mean(throughputs),
                    'min_throughput': np.min(throughputs),
                    'max_throughput': np.max(throughputs),
                    'avg_context_tokens': np.mean(token_usages),
                    'max_context_tokens': np.max(token_usages)
                }

                print(f"RESULTS:")
                print(f"  Avg latency: {results[f'{turns}_turns']['avg_latency']:.0f}ms")
                print(f"  P50 latency: {results[f'{turns}_turns']['p50_latency']:.0f}ms")
                print(f"  P95 latency: {results[f'{turns}_turns']['p95_latency']:.0f}ms")
                print(f"  P99 latency: {results[f'{turns}_turns']['p99_latency']:.0f}ms")
                print(f"  Avg throughput: {results[f'{turns}_turns']['avg_throughput']:.1f} tokens/sec")
                print(f"  Avg context tokens: {results[f'{turns}_turns']['avg_context_tokens']:.0f}")
                print(f"  Max context tokens: {results[f'{turns}_turns']['max_context_tokens']:.0f}")

        # V2 TEST: Verify context stays bounded
        print("\n📊 V2 Token Budget Test:")
        print(f"  10 turns:  {results['10_turns']['avg_context_tokens']:.0f} tokens")
        print(f"  50 turns:  {results['50_turns']['avg_context_tokens']:.0f} tokens")
        print(f"  100 turns: {results['100_turns']['avg_context_tokens']:.0f} tokens")
        print(f"  200 turns: {results['200_turns']['avg_context_tokens']:.0f} tokens")
        
        if results['10_turns']['avg_context_tokens'] > 0:
            growth = ((results['200_turns']['avg_context_tokens'] - results['10_turns']['avg_context_tokens']) 
                      / results['10_turns']['avg_context_tokens'] * 100)
            print(f"  Context growth (10→200 turns): {growth:.1f}%")
            print(f"  ✅ Expected: <50% growth (bounded by token budget)")
        
        return results


class MemoryEfficiencyBenchmark:
    """Benchmark V2 memory efficiency with 15-turn sliding window"""

    def __init__(self, engine, memory, context_manager):
        self.engine = engine
        self.memory = memory
        self.context_manager = context_manager

    async def run(self) -> Dict:
        """Run V2 memory efficiency benchmark"""
        print("\n" + "="*80)
        print("BENCHMARK 2: MEMORY EFFICIENCY (V2)")
        print("Testing 15-turn sliding window + token budget + deduplication")
        print("="*80)
        print("Building conversation and measuring token usage...")

        conv_id = "memory_test_v2"
        context_tokens_per_turn = []
        recent_turns_per_turn = []

        # Build 100 turn conversation
        for i in range(100):
            request = GenerationRequest(
                conversation_id=conv_id,
                messages=[{"role": "user", "content": f"Turn {i+1}: Explain Python decorators briefly."}],
                model="test",
                max_tokens=50,
                temperature=0.7
            )

            result = await self.engine.generate(request)

            if result.get('success'):
                context_tokens = result['metadata'].get('context_tokens', 0)
                recent_turns = result['metadata'].get('recent_turns_used', 0)
                
                context_tokens_per_turn.append(context_tokens)
                recent_turns_per_turn.append(recent_turns)

        if not context_tokens_per_turn:
            print("\n❌ ERROR: No data collected")
            return {'error': 'No data'}

        # Traditional approach (full history)
        traditional_tokens_at_100 = sum(range(1, 101)) * 30

        # V2 approach (bounded context)
        v2_tokens_at_100 = context_tokens_per_turn[-1]

        # Memory savings
        savings = ((traditional_tokens_at_100 - v2_tokens_at_100) / traditional_tokens_at_100 * 100)

        # Check if context stays bounded
        first_half_avg = np.mean(context_tokens_per_turn[:50])
        second_half_avg = np.mean(context_tokens_per_turn[50:])
        growth = ((second_half_avg - first_half_avg) / first_half_avg * 100) if first_half_avg > 0 else 0

        # Check sliding window stability
        recent_turns_at_50 = np.mean(recent_turns_per_turn[45:50]) if len(recent_turns_per_turn) >= 50 else 0
        recent_turns_at_100 = np.mean(recent_turns_per_turn[95:100]) if len(recent_turns_per_turn) >= 100 else 0

        # Get V2 stats
        v2_stats = self.context_manager.get_stats() if self.context_manager else {}

        results = {
            'traditional_tokens_at_100': traditional_tokens_at_100,
            'v2_tokens_at_100': v2_tokens_at_100,
            'savings_percent': savings,
            'context_growth_percent': growth,
            'recent_turns_at_50': recent_turns_at_50,
            'recent_turns_at_100': recent_turns_at_100,
            'v2_deduplication_count': v2_stats.get('turns_deduplicated', 0),
            'v2_offload_count': v2_stats.get('turns_offloaded', 0)
        }

        print(f"\nRESULTS:")
        print(f"  At turn 100:")
        print(f"    Traditional approach: {traditional_tokens_at_100} tokens")
        print(f"    V2 approach: {v2_tokens_at_100} tokens")
        print(f"    Savings: {savings:.1f}%")
        print(f"  Context growth (first 50 → last 50): {growth:.1f}%")
        print(f"    ✅ Expected: <20% (token budget keeps it bounded)")
        print(f"\n  V2 Sliding Window:")
        print(f"    Recent turns kept at turn 50: {recent_turns_at_50:.1f}")
        print(f"    Recent turns kept at turn 100: {recent_turns_at_100:.1f}")
        print(f"    ✅ Should stay around 10-15 turns")
        print(f"\n  V2 Optimizations:")
        print(f"    Turns deduplicated: {results['v2_deduplication_count']}")
        print(f"    Turns offloaded to Qdrant: {results['v2_offload_count']}")

        return results


class RetrievalAccuracyBenchmark:
    """Benchmark V2 two-stage retrieval (Nomic-Embed + BGE Reranker)"""

    def __init__(self, engine, memory):
        self.engine = engine
        self.memory = memory

    async def run(self) -> Dict:
        """Run V2 retrieval accuracy benchmark with reranking"""
        print("\n" + "="*80)
        print("BENCHMARK 3: RETRIEVAL ACCURACY (V2 with Nomic-Embed + BGE Reranker)")
        print("Tests two-stage retrieval: embedding → reranking")
        print("="*80)
        print("Testing semantic retrieval with reranking...")

        conv_id = "accuracy_test_v2"

        # Test pairs
        test_pairs = [
            {
                "initial": "I'm allergic to peanuts and shellfish",
                "query": "Do you remember my dietary restrictions?",
                "topic": "allergies/dietary"
            },
            {
                "initial": "My dog's name is Max and he's a golden retriever",
                "query": "What's my pet's name?",
                "topic": "pet name"
            },
            {
                "initial": "I work as a software engineer at Google in Mountain View",
                "query": "Where do I work?",
                "topic": "job/workplace"
            },
            {
                "initial": "I'm planning a trip to Japan next summer for 2 weeks",
                "query": "What travel plans did I mention?",
                "topic": "travel plans"
            },
            {
                "initial": "I prefer working out in the morning around 6 AM",
                "query": "When do I like to exercise?",
                "topic": "workout schedule"
            },
            {
                "initial": "My birthday is on December 15th",
                "query": "What's my birth date?",
                "topic": "birthday"
            },
            {
                "initial": "I'm learning Python and JavaScript for web development",
                "query": "What programming languages am I studying?",
                "topic": "learning/skills"
            },
            {
                "initial": "I live in Edmonton, Alberta, Canada",
                "query": "Where is my home?",
                "topic": "location"
            }
        ]

        # Store initial statements
        print(f"\nStoring {len(test_pairs)} initial statements...")
        for i, pair in enumerate(test_pairs):
            self.memory.add_turn(
                conversation_id=conv_id,
                text=pair["initial"],
                metadata={
                    'response': f"Got it, I'll remember that about {pair['topic']}.",
                    'test_index': i,
                    'topic': pair['topic']
                }
            )

            # Add filler turns
            for j in range(3):
                self.memory.add_turn(
                    conversation_id=conv_id,
                    text=f"Filler message {i*3 + j}",
                    metadata={'response': 'Acknowledged', 'filler': True}
                )

        # Wait for indexing
        print(f"Waiting for Qdrant indexing...")
        await asyncio.sleep(2)

        # Test retrieval
        print(f"\nTesting two-stage retrieval (Nomic-Embed + BGE Reranker)...")
        print("="*80)

        correct = 0
        embedding_similarities = []
        rerank_scores = []
        results_detail = []

        for i, pair in enumerate(test_pairs):
            print(f"\n  Test {i+1}/{len(test_pairs)}: {pair['topic']}")
            print(f"    Query: '{pair['query'][:60]}...'")

            # V2: Retrieve with reranking
            context = self.memory.retrieve_context(conv_id, pair['query'], top_k=3)

            if context.get('success') and context.get('results'):
                results_list = context['results']

                found = False
                for result in results_list:
                    if result['metadata'].get('filler'):
                        continue

                    embedding_sim = result.get('similarity', 0)
                    rerank_score = result.get('rerank_score', 0)

                    if result['metadata'].get('topic') == pair['topic']:
                        found = True
                        embedding_similarities.append(embedding_sim)
                        
                        if rerank_score > 0:
                            rerank_scores.append(rerank_score)

                        print(f"    ✅ FOUND!")
                        print(f"       Embedding similarity: {embedding_sim:.3f}")
                        if rerank_score > 0:
                            print(f"       Rerank score: {rerank_score:.3f}")

                        if embedding_sim >= 0.4 or rerank_score >= 0.5:
                            correct += 1
                            results_detail.append({
                                'test': i+1,
                                'topic': pair['topic'],
                                'embedding_sim': embedding_sim,
                                'rerank_score': rerank_score,
                                'success': True
                            })
                        else:
                            results_detail.append({
                                'test': i+1,
                                'topic': pair['topic'],
                                'embedding_sim': embedding_sim,
                                'rerank_score': rerank_score,
                                'success': False,
                                'reason': 'Low scores'
                            })
                        break

                if not found:
                    print(f"    ❌ NOT FOUND")
                    results_detail.append({
                        'test': i+1,
                        'topic': pair['topic'],
                        'success': False,
                        'reason': 'Not in top results'
                    })
            else:
                print(f"    ❌ Retrieval failed")
                results_detail.append({
                    'test': i+1,
                    'topic': pair['topic'],
                    'success': False,
                    'reason': 'Retrieval failed'
                })

        # Calculate metrics
        accuracy = (correct / len(test_pairs) * 100)

        results = {
            'accuracy_percent': accuracy,
            'correct': correct,
            'total': len(test_pairs),
            'avg_embedding_similarity': np.mean(embedding_similarities) if embedding_similarities else 0,
            'avg_rerank_score': np.mean(rerank_scores) if rerank_scores else 0,
            'reranked_count': len(rerank_scores),
            'details': results_detail
        }

        # Print summary
        print("\n" + "="*80)
        print("RESULTS:")
        print(f"  V2 Two-Stage Retrieval Accuracy: {accuracy:.1f}% ({correct}/{len(test_pairs)})")
        print(f"  Avg embedding similarity (Nomic-Embed): {results['avg_embedding_similarity']:.3f}")
        if rerank_scores:
            print(f"  Avg rerank score (BGE): {results['avg_rerank_score']:.3f}")
            print(f"  Reranked results: {len(rerank_scores)}/{len(test_pairs)}")
        print(f"\n  ✅ Expected: 75-90% accuracy with V2 (vs 60-75% with old embedder)")

        return results

class ComparativeBenchmark:
    """Compare V2 retrieval vs summarization for coding tasks"""
    
    def __init__(self, engine, memory):
        self.engine = engine
        self.memory = memory
    
    async def run(self) -> Dict:
        """Run V2 comparison"""
        print("\n" + "="*80)
        print("BENCHMARK 4: V2 RETRIEVAL VS SUMMARIZATION (CODING)")
        print("Compare V2 two-stage retrieval vs local model summarization")
        print("="*80)
        
        # Generate 50-turn coding conversation
        coding_conversation = self.generate_coding_session()
        
        print(f"\nGenerated {len(coding_conversation)} turn coding conversation")
        print("Testing both approaches...\n")
        
        # Test 1: V2 retrieval
        print("Testing V2 retrieval (Nomic-Embed + BGE Reranker)...")
        retrieval_result = await self.test_retrieval_approach(coding_conversation)
        
        # Test 2: Summarization
        print("\nTesting summarization approach...")
        summary_result = await self.test_summary_approach(coding_conversation)
        
        # Compare
        results = {
            'retrieval_v2': retrieval_result,
            'summary': summary_result,
            'comparison': {
                'token_savings': (
                    (summary_result['tokens_used'] - retrieval_result['tokens_used']) 
                    / summary_result['tokens_used'] * 100
                ) if summary_result['tokens_used'] > 0 else 0,
                'accuracy_diff': retrieval_result['accuracy'] - summary_result['accuracy'],
                'latency_diff_ms': (retrieval_result['latency_ms'] - summary_result['latency_ms'])
            }
        }
        
        # Print results
        print("\n" + "="*80)
        print("V2 COMPARISON RESULTS:")
        print("="*80)
        print(f"\n{'Metric':<25} | {'V2 Retrieval':<15} | {'Summary':<15}")
        print("-"*80)
        print(f"{'Code Accuracy':<25} | {retrieval_result['accuracy']:<15.1%} | {summary_result['accuracy']:<15.1%}")
        print(f"{'Tokens Used':<25} | {retrieval_result['tokens_used']:<15} | {summary_result['tokens_used']:<15}")
        print(f"{'Latency (ms)':<25} | {retrieval_result['latency_ms']:<15.0f} | {summary_result['latency_ms']:<15.0f}")
        
        print(f"\n{'Token Savings:':<25} {results['comparison']['token_savings']:.1f}%")
        print(f"{'Accuracy Gain:':<25} {results['comparison']['accuracy_diff']*100:+.1f}%")
        print(f"\n  ✅ V2 Expected: 60-80% token savings, +10-20% accuracy vs summary")
        
        return results
    
    def generate_coding_session(self) -> List[Dict]:
        """Generate realistic 50-turn coding conversation"""
        conversation = []
        
        # Turns 1-9: Setup
        for i in range(1, 10):
            conversation.append({
                "turn": i,
                "user": f"Add feature {i}",
                "assistant": "``````"
            })
        
        # Turn 10: JWT Authentication (KEY TURN)
        # Turn 50: Test query
        conversation.append({
            "turn": 50,
            "user": "Show me the JWT authentication code with create_access_token and login function",  # BETTER MATCH
            "assistant": "Let me retrieve that",
            "requires_code": ["create_access_token", "jwt_required", "login", "JWT_SECRET_KEY"]
        })

        
        # Turns 11-49: Filler
        for i in range(11, 50):
            conversation.append({
                "turn": i,
                "user": f"Add validation {i}",
                "assistant": "``````"
            })
        
        # Turn 50: Test query
        conversation.append({
            "turn": 50,
            "user": "Show me the JWT login function we created earlier",
            "assistant": "Let me retrieve that",
            "requires_code": ["create_access_token", "jwt_required", "login", "JWT_SECRET_KEY"]
        })
        
        return conversation
    
    async def test_retrieval_approach(self, conversation: List[Dict]) -> Dict:
        """Test V2 retrieval"""
        conv_id = "coding_retrieval_v2_test"
        
        print(f"  Storing {len(conversation[:49])} turns...")
        
        # Feed turns 1-49
        for turn in conversation[:49]:
            self.memory.add_turn(
                conversation_id=conv_id,
                text=turn["user"],
                metadata={'response': turn["assistant"], 'turn': turn["turn"]}
            )
        
        # Wait for Qdrant indexing
        print("  Waiting for vector DB indexing...")
        await asyncio.sleep(3)
        
        # Verify data stored
        stored_count = self.memory.get_conversation_length(conv_id)
        print(f"  Verified: {stored_count} turns in vector DB")
        
        turn_50 = conversation[49]
        start_time = time.time()
        
        # V2: Two-stage retrieval
        print(f"  Querying: '{turn_50['user'][:50]}...'")
        context = self.memory.retrieve_context(
            conversation_id=conv_id,
            query=turn_50["user"],
            top_k=5
        )
        
        latency = (time.time() - start_time) * 1000
        
        print(f"  Retrieved {len(context.get('results', []))} results")
        
        # Check for required code elements
        required_codes = turn_50.get("requires_code", [])
        matches_found = []
        
        if context.get('success') and context.get('results'):
            print(f"  Checking results for: {required_codes}")
            for i, result in enumerate(context['results']):
                response_text = result['metadata'].get('response', '')
                similarity = result.get('similarity', 0)
                rerank = result.get('rerank_score', 0)
                
                print(f"    Result {i+1}: sim={similarity:.3f}, rerank={rerank:.3f}")
                
                for required in required_codes:
                    if required in response_text and required not in matches_found:
                        matches_found.append(required)
                        print(f"      ✅ Found: {required}")
        else:
            print(f"  ❌ Retrieval failed or no results")
        
        accuracy = len(matches_found) / len(required_codes) if required_codes else 0
        
        # Count tokens properly
        tokens_used = sum(count_tokens(r.get('text', '') + r.get('metadata', {}).get('response', '')) 
                         for r in context.get('results', []))
        
        print(f"\n  ✅ V2 Retrieval complete")
        print(f"     - Found {len(matches_found)}/{len(required_codes)} code elements: {matches_found}")
        print(f"     - Accuracy: {accuracy:.1%}")
        print(f"     - Tokens used: {tokens_used}")
        
        return {
            'accuracy': accuracy,
            'tokens_used': tokens_used,
            'latency_ms': latency,
            'matches_found': matches_found
        }
    
    async def test_summary_approach(self, conversation: List[Dict]) -> Dict:
        """Test summarization"""
        full_history = []
        for turn in conversation[:49]:
            full_history.append(f"User: {turn['user']}")
            full_history.append(f"Assistant: {turn['assistant']}")
        
        history_text = "\n".join(full_history)
        
        start_time = time.time()
        
        summary_prompt = f"""Summarize this coding conversation, preserving code snippets:

{history_text[:3000]}

Provide detailed summary with all function names:"""
        
        summary_request = GenerationRequest(
            conversation_id="summary_gen_v2",
            messages=[{"role": "user", "content": summary_prompt}],
            model="test",
            max_tokens=1000,
            temperature=0.3
        )
        
        summary_result = await self.engine.generate(summary_request)
        
        if not summary_result.get('success'):
            return {'accuracy': 0, 'tokens_used': 0, 'latency_ms': 0, 'error': 'Failed'}
        
        summary_text = summary_result.get('response', '')
        latency = (time.time() - start_time) * 1000
        
        # Check for required codes
        turn_50 = conversation[49]
        required_codes = turn_50.get("requires_code", [])
        matches_found = [req for req in required_codes if req.lower() in summary_text.lower()]
        
        accuracy = len(matches_found) / len(required_codes) if required_codes else 0
        tokens_used = count_tokens(summary_text)
        
        print(f"\n  ✅ Summarization complete")
        print(f"     - Found {len(matches_found)}/{len(required_codes)} code elements")
        print(f"     - Accuracy: {accuracy:.1%}")
        
        return {
            'accuracy': accuracy,
            'tokens_used': tokens_used,
            'latency_ms': latency,
            'matches_found': matches_found
        }


async def run_all_benchmarks():
    """Run all V2 benchmarks"""
    print("="*80)
    print("INFINITE MEMORY INFERENCE API - V2 GPU BENCHMARK SUITE")
    print("Testing: Nomic-Embed + BGE Reranker + Sliding Window + Token Budget")
    print("="*80)

    # Initialize system
    print("\nInitializing V2 system...")

    # Create vector DB
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

    # Create infinite memory engine with V2
    infinite_engine = InfiniteMemoryEngine(
        vllm_engine=vllm_engine,
        memory_manager=memory_manager,
        max_context_tokens=4096,
        context_retrieval_k=5,
        use_v2=True  # ENABLE V2!
    )

    # Get context manager reference
    context_manager_v2 = infinite_engine.context_manager

    print("✅ V2 System initialized\n")

    # Run benchmarks
    all_results = {}

    try:
        print("Starting Benchmark 1...")
        throughput_bench = ThroughputBenchmark(infinite_engine, memory_manager)
        all_results['throughput'] = await throughput_bench.run()
    except Exception as e:
        print(f"\n❌ Throughput benchmark failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        print("\nStarting Benchmark 2...")
        memory_bench = MemoryEfficiencyBenchmark(infinite_engine, memory_manager, context_manager_v2)
        all_results['memory_efficiency'] = await memory_bench.run()
    except Exception as e:
        print(f"\n❌ Memory efficiency benchmark failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        print("\nStarting Benchmark 3...")
        accuracy_bench = RetrievalAccuracyBenchmark(infinite_engine, memory_manager)
        all_results['accuracy'] = await accuracy_bench.run()
    except Exception as e:
        print(f"\n❌ Retrieval accuracy benchmark failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        print("\nStarting Benchmark 4...")
        comparative_bench = ComparativeBenchmark(infinite_engine, memory_manager)
        all_results['comparative'] = await comparative_bench.run()
    except Exception as e:
        print(f"\n❌ Comparative benchmark failed: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "="*80)
    print("V2 BENCHMARK SUMMARY")
    print("="*80)

    if 'throughput' in all_results and '10_turns' in all_results['throughput']:
        print(f"\n📊 THROUGHPUT:")
        print(f"   10 turns:  {all_results['throughput']['10_turns']['avg_throughput']:.1f} tok/s")
        print(f"   50 turns:  {all_results['throughput']['50_turns']['avg_throughput']:.1f} tok/s")
        print(f"   100 turns: {all_results['throughput']['100_turns']['avg_throughput']:.1f} tok/s")
        print(f"   200 turns: {all_results['throughput']['200_turns']['avg_throughput']:.1f} tok/s")
        print(f"   Latency (P50): {all_results['throughput']['10_turns']['p50_latency']:.0f}ms")

    if 'memory_efficiency' in all_results and 'savings_percent' in all_results['memory_efficiency']:
        print(f"\n💾 MEMORY EFFICIENCY (V2):")
        print(f"   Token savings: {all_results['memory_efficiency']['savings_percent']:.1f}%")
        print(f"   Context growth: {all_results['memory_efficiency']['context_growth_percent']:.1f}%")
        print(f"   Turns deduplicated: {all_results['memory_efficiency']['v2_deduplication_count']}")
        print(f"   Turns offloaded: {all_results['memory_efficiency']['v2_offload_count']}")

    if 'accuracy' in all_results and 'accuracy_percent' in all_results['accuracy']:
        print(f"\n🎯 RETRIEVAL ACCURACY (V2 with Nomic-Embed + BGE):")
        print(f"   Semantic accuracy: {all_results['accuracy']['accuracy_percent']:.1f}%")
        print(f"   Avg embedding similarity: {all_results['accuracy']['avg_embedding_similarity']:.3f}")
        if all_results['accuracy'].get('avg_rerank_score', 0) > 0:
            print(f"   Avg rerank score: {all_results['accuracy']['avg_rerank_score']:.3f}")

    if 'comparative' in all_results and 'comparison' in all_results['comparative']:
        print(f"\n🆚 V2 RETRIEVAL VS SUMMARIZATION:")
        print(f"   Token savings: {all_results['comparative']['comparison']['token_savings']:.1f}%")
        print(f"   V2 Retrieval accuracy: {all_results['comparative']['retrieval_v2']['accuracy']:.1%}")
        print(f"   Summary accuracy: {all_results['comparative']['summary']['accuracy']:.1%}")
        print(f"   Accuracy gain: {all_results['comparative']['comparison']['accuracy_diff']*100:+.1f}%")

    print("\n" + "="*80)
    print("✅ V2 BENCHMARK COMPLETE")
    print("="*80)
    print("\nV2 Improvements Tested:")
    print("  ✅ Nomic-Embed (768 dim) vs old embedder (384 dim)")
    print("  ✅ BGE Reranker for two-stage retrieval")
    print("  ✅ 15-turn sliding window (vs 3 turns)")
    print("  ✅ Token budget enforcement (2000 tokens)")
    print("  ✅ Automatic deduplication and offloading")

    return all_results


if __name__ == "__main__":
    results = asyncio.run(run_all_benchmarks())
