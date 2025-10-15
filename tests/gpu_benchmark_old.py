"""
GPU Benchmark Suite for Infinite Memory Inference API
UPDATED: Realistic semantic retrieval testing + Retrieval vs Summarization comparison
Uses local model for summarization (no OpenAI API required)
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
    """Benchmark retrieval accuracy with REALISTIC semantic queries"""

    def __init__(self, engine, memory):
        self.engine = engine
        self.memory = memory

    async def run(self) -> Dict:
        """Run retrieval accuracy benchmark with semantic understanding test"""
        print("\n" + "="*80)
        print("BENCHMARK 3: RETRIEVAL ACCURACY (SEMANTIC UNDERSTANDING)")
        print("Tests if system can find relevant context with different wording")
        print("="*80)
        print("Testing semantic retrieval accuracy...")

        conv_id = "accuracy_test"

        # REALISTIC TEST PAIRS: Store with one wording, query with different wording
        # This tests actual semantic understanding, not just exact matching
        test_pairs = [
            # (what user says initially, what they ask later, expected topic match)
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

        # Store initial statements (simulating early conversation)
        print(f"\nStoring {len(test_pairs)} initial statements (turns 1-{len(test_pairs)})...")
        for i, pair in enumerate(test_pairs):
            # Store the initial statement
            self.memory.add_turn(
                conversation_id=conv_id,
                text=pair["initial"],
                metadata={
                    'response': f"Got it, I'll remember that about {pair['topic']}.",
                    'test_index': i,
                    'topic': pair['topic']
                }
            )

            # Add some filler turns to simulate real conversation distance
            for j in range(3):
                filler_text = f"Filler message {i*3 + j}"
                self.memory.add_turn(
                    conversation_id=conv_id,
                    text=filler_text,
                    metadata={'response': 'Acknowledged', 'filler': True}
                )

        # Give Qdrant time to index (important!)
        print(f"Waiting for Qdrant to index {self.memory.get_conversation_length(conv_id)} turns...")
        await asyncio.sleep(2)  # 2 seconds for larger dataset

        print(f"Stored data. Conversation has {self.memory.get_conversation_length(conv_id)} turns")

        # Test retrieval with DIFFERENT wording (semantic understanding)
        print(f"\nTesting semantic retrieval (different wording than storage)...")
        print("="*80)

        correct = 0
        similarities = []
        results_detail = []

        for i, pair in enumerate(test_pairs):
            print(f"\n  Test {i+1}/{len(test_pairs)}: {pair['topic']}")
            print(f"    Stored: '{pair['initial'][:60]}...'")
            print(f"    Query:  '{pair['query'][:60]}...'")

            # Retrieve using different wording
            context = self.memory.retrieve_context(conv_id, pair['query'], top_k=5)

            if context.get('success') and context.get('results'):
                results = context['results']

                # Look for the original statement in results
                found = False
                top_similarity = 0

                for result in results:
                    # Skip filler messages
                    if result['metadata'].get('filler'):
                        continue

                    similarity = result.get('similarity', 0)

                    # Check if this is the matching topic
                    if result['metadata'].get('topic') == pair['topic']:
                        found = True
                        similarities.append(similarity)
                        top_similarity = similarity

                        print(f"    ✅ FOUND! Similarity: {similarity:.3f}")
                        print(f"    Retrieved: '{result['text'][:60]}...'")

                        if similarity >= 0.4:  # Good semantic match threshold
                            correct += 1
                            results_detail.append({
                                'test': i+1,
                                'topic': pair['topic'],
                                'similarity': similarity,
                                'success': True
                            })
                        else:
                            print(f"    ⚠️  Low similarity (< 0.4), may not be useful")
                            results_detail.append({
                                'test': i+1,
                                'topic': pair['topic'],
                                'similarity': similarity,
                                'success': False,
                                'reason': 'Low similarity'
                            })
                        break

                if not found:
                    print(f"    ❌ NOT FOUND in top results")
                    print(f"    Top result was: '{results[0]['text'][:60]}...' ({results[0].get('similarity', 0):.3f})")
                    results_detail.append({
                        'test': i+1,
                        'topic': pair['topic'],
                        'success': False,
                        'reason': 'Not in top results'
                    })
            else:
                print(f"    ❌ Retrieval failed: {context.get('error', 'No results')}")
                results_detail.append({
                    'test': i+1,
                    'topic': pair['topic'],
                    'success': False,
                    'reason': 'Retrieval failed'
                })

        # Calculate metrics
        accuracy = (correct / len(test_pairs) * 100) if test_pairs else 0

        results = {
            'accuracy_percent': accuracy,
            'correct': correct,
            'total': len(test_pairs),
            'avg_similarity': np.mean(similarities) if similarities else 0,
            'min_similarity': np.min(similarities) if similarities else 0,
            'max_similarity': np.max(similarities) if similarities else 0,
            'details': results_detail
        }

        # Print summary
        print("\n" + "="*80)
        print("RESULTS:")
        print(f"  Accuracy: {accuracy:.1f}% ({correct}/{len(test_pairs)} with similarity >= 0.4)")
        print(f"  Avg similarity: {results['avg_similarity']:.3f}")

        if similarities:
            print(f"  Min similarity: {results['min_similarity']:.3f}")
            print(f"  Max similarity: {results['max_similarity']:.3f}")
        else:
            print(f"  Min similarity: N/A (no matches found)")
            print(f"  Max similarity: N/A (no matches found)")

        # Interpretation guide
        print("\n  Similarity Score Interpretation:")
        print("    0.7-1.0: Excellent semantic match")
        print("    0.5-0.7: Good semantic match")
        print("    0.4-0.5: Acceptable match")
        print("    <0.4:   Poor match (not counted as correct)")

        return results


class ComparativeBenchmark:
    """Compare retrieval vs summarization for coding tasks - uses local model only"""
    
    def __init__(self, engine, memory):
        self.engine = engine
        self.memory = memory
    
    async def run(self) -> Dict:
        """Run comparison: Your retrieval vs local model summarization"""
        print("\n" + "="*80)
        print("BENCHMARK 4: RETRIEVAL VS SUMMARIZATION (CODING)")
        print("Compare semantic retrieval vs summary-based memory")
        print("Using local model for both approaches (no external APIs)")
        print("="*80)
        
        # Generate 50-turn coding conversation
        coding_conversation = self.generate_coding_session()
        
        print(f"\nGenerated {len(coding_conversation)} turn coding conversation")
        print("Testing both approaches...\n")
        
        # Test 1: Your retrieval system
        print("Testing retrieval approach...")
        retrieval_result = await self.test_retrieval_approach(coding_conversation)
        
        # Test 2: Local model summarization
        print("\nTesting summarization approach (using local model)...")
        summary_result = await self.test_summary_approach(coding_conversation)
        
        # Compare results
        results = {
            'retrieval': retrieval_result,
            'summary': summary_result,
            'comparison': {
                'token_savings': (
                    (summary_result['tokens_used'] - retrieval_result['tokens_used']) 
                    / summary_result['tokens_used'] * 100
                ) if summary_result['tokens_used'] > 0 else 0,
                'accuracy_diff': retrieval_result['accuracy'] - summary_result['accuracy'],
                'latency_diff_ms': (retrieval_result['latency'] - summary_result['latency']) * 1000
            }
        }
        
        # Print results
        print("\n" + "="*80)
        print("COMPARISON RESULTS:")
        print("="*80)
        print(f"\n{'Metric':<25} | {'Retrieval':<15} | {'Summary':<15}")
        print("-"*80)
        print(f"{'Code Accuracy':<25} | {retrieval_result['accuracy']:<15.1%} | {summary_result['accuracy']:<15.1%}")
        print(f"{'Tokens Used':<25} | {retrieval_result['tokens_used']:<15} | {summary_result['tokens_used']:<15}")
        print(f"{'Latency (ms)':<25} | {retrieval_result['latency']*1000:<15.0f} | {summary_result['latency']*1000:<15.0f}")
        
        print(f"\n{'Token Savings:':<25} {results['comparison']['token_savings']:.1f}%")
        print(f"{'Accuracy Gain:':<25} {results['comparison']['accuracy_diff']*100:+.1f}%")
        
        if results['comparison']['latency_diff_ms'] < 0:
            print(f"{'Latency:':<25} {abs(results['comparison']['latency_diff_ms']):.0f}ms FASTER")
        else:
            print(f"{'Latency:':<25} {results['comparison']['latency_diff_ms']:.0f}ms slower")
        
        return results
    
    def generate_coding_session(self) -> List[Dict]:
        """Generate realistic 50-turn coding conversation"""
        
        conversation = []
        
        # Turns 1-10: Initial setup + authentication
        conversation.append({
            "turn": 1, 
            "user": "Create a Flask REST API for a todo app", 
            "assistant": "``````"
        })
        
        conversation.append({
            "turn": 2,
            "user": "Add a POST endpoint to create todos",
            "assistant": "``````"
        })
        
        conversation.append({
            "turn": 3,
            "user": "Add SQLAlchemy for database",
            "assistant": "``````"
        })
        
        # Turns 4-9: More basic features
        for i in range(4, 10):
            conversation.append({
                "turn": i,
                "user": f"Add more endpoints turn {i}",
                "assistant": f"``````"
            })
        
        # Turn 10: JWT Authentication (THE KEY TURN TO RETRIEVE)
        conversation.append({
            "turn": 10,
            "user": "Add JWT authentication to protect routes",
            "assistant": "``````"
        })
        
        # Turns 11-49: Filler conversations
        for i in range(11, 50):
            conversation.append({
                "turn": i,
                "user": f"Add validation for turn {i}",
                "assistant": f"``````"
            })
        
        # Turn 50: The test query
        conversation.append({
            "turn": 50,
            "user": "Show me the JWT login function we created earlier",
            "assistant": "Let me retrieve that for you",
            "requires_code": ["create_access_token", "jwt_required", "login", "JWT_SECRET_KEY"]
        })
        
        print(f"  Generated {len(conversation)} turns")  # DEBUG
        
        return conversation


    
    async def test_retrieval_approach(self, conversation: List[Dict]) -> Dict:
        """Test your retrieval system"""
        conv_id = "coding_retrieval_test"
        
        # SAFETY CHECK
        if len(conversation) < 50:
            print(f"  ❌ ERROR: Conversation only has {len(conversation)} turns, need 50")
            return {
                'accuracy': 0,
                'tokens_used': 0,
                'latency': 0,
                'error': f'Not enough turns ({len(conversation)})'
            }
        
        # Feed turns 1-49 (indices 0-48)
        for turn in conversation[:49]:
            if not isinstance(turn, dict):
                print(f"  ❌ ERROR: Turn is not a dict: {type(turn)}")
                continue
                
            self.memory.add_turn(
                conversation_id=conv_id,
                text=turn["user"],
                metadata={'response': turn["assistant"], 'turn': turn["turn"]}
            )
        
        # Wait for indexing
        print("  Waiting for vector DB indexing...")
        await asyncio.sleep(2)
        
        # Turn 50 (index 49)
        turn_50 = conversation[49]
        
        # SAFETY CHECK
        if not isinstance(turn_50, dict):
            print(f"  ❌ ERROR: Turn 50 is not a dict: {type(turn_50)}")
            return {
                'accuracy': 0,
                'tokens_used': 0,
                'latency': 0,
                'error': 'Turn 50 wrong type'
            }
        
        start_time = time.time()
        
        # Retrieve context
        context = self.memory.retrieve_context(
            conversation_id=conv_id,
            query=turn_50["user"],
            top_k=3
        )
        
        latency = time.time() - start_time
        
        # Rest of the method stays the same...
        # (Use the version from my previous message)

    
    async def test_summary_approach(self, conversation: List[Dict]) -> Dict:
        """Test summarization using YOUR local model (no OpenAI needed)"""
        
        print("  Using local model for summarization...")
        
        # Build full conversation history (traditional approach)
        full_history = []
        for turn in conversation[:49]:
            full_history.append(f"User: {turn['user']}")
            full_history.append(f"Assistant: {turn['assistant']}")
        
        # Create summary prompt
        history_text = "\n".join(full_history)
        
        start_time = time.time()
        
        # Use YOUR model to summarize
        summary_prompt = f"""Summarize this coding conversation in detail, preserving all important code snippets, function names, and technical details:

    {history_text[:3000]}

    Provide a detailed summary including all function names and code structures:"""
        
        # Generate summary using your vLLM engine
        summary_request = GenerationRequest(
            conversation_id="summary_generation",
            messages=[{"role": "user", "content": summary_prompt}],
            model="test",
            max_tokens=1000,
            temperature=0.3
        )
        
        summary_result = await self.engine.generate(summary_request)
        
        if not summary_result.get('success'):
            print(f"  ❌ Summary generation failed")
            return {
                'accuracy': 0,
                'tokens_used': 0,
                'latency': 0,
                'error': 'Summary generation failed'
            }
        
        # Get summary text from response
        summary_text = None
        for key in ['text', 'output', 'response', 'generated_text', 'content']:
            if key in summary_result:
                summary_text = summary_result[key]
                break
        
        if summary_text is None:
            # Try getting from metadata
            if 'metadata' in summary_result and 'generated_text' in summary_result['metadata']:
                summary_text = summary_result['metadata']['generated_text']
            else:
                print(f"  ❌ Could not find text. Keys: {summary_result.keys()}")
                return {
                    'accuracy': 0,
                    'tokens_used': 0,
                    'latency': 0,
                    'error': 'Text key not found'
                }
        
        latency = time.time() - start_time
        
        # DEBUG: Print summary preview
        print(f"\n  📋 DEBUG - Summary preview:")
        print(f"     {summary_text[:300]}...")
        
        # Turn 50 - test if summary contains JWT code
        turn_50 = conversation
        required_codes = turn_50.get("requires_code", [])
        matches_found = []
        
        for required in required_codes:
            # Flexible matching
            if required.lower() in summary_text.lower():
                matches_found.append(required)
        
        accuracy = len(matches_found) / len(required_codes) if required_codes else 0
        
        # Count tokens in summary
        tokens_used = len(summary_text) // 4
        
        print(f"\n  ✅ Summarization complete")
        print(f"     - Found {len(matches_found)}/{len(required_codes)} required code elements:")
        for match in matches_found:
            print(f"       ✓ {match}")
        for required in required_codes:
            if required not in matches_found:
                print(f"       ✗ {required} (NOT FOUND)")
        print(f"     - Code accuracy: {accuracy:.1%}")
        print(f"     - Tokens used: {tokens_used}")
        print(f"     - Summary length: {len(summary_text)} chars")
        
        return {
            'accuracy': accuracy,
            'tokens_used': tokens_used,
            'latency': latency,
            'summary_length': len(summary_text),
            'matches_found': matches_found
        }

async def run_all_benchmarks():
    """Run all benchmarks"""
    print("="*80)
    print("INFINITE MEMORY INFERENCE API - GPU BENCHMARK SUITE")
    print("="*80)
    print("\nStarting comprehensive GPU benchmark suite...")
    print("This will take approximately 15-20 minutes")

    # Initialize system
    print("\nInitializing system...\n")

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

    try:
        # Benchmark 4: Comparative (Retrieval vs Summarization)
        comparative_bench = ComparativeBenchmark(infinite_engine, memory_manager)
        all_results['comparative'] = await comparative_bench.run()
    except Exception as e:
        print(f"\n❌ Comparative benchmark failed: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)

    if 'throughput' in all_results and '10_turns' in all_results['throughput']:
        print(f"\n📊 THROUGHPUT:")
        print(f"   10 turns: {all_results['throughput']['10_turns']['avg_throughput']:.1f} tok/s")
        print(f"   Latency (P50): {all_results['throughput']['10_turns']['p50_latency']:.0f}ms")

    if 'memory_efficiency' in all_results and 'savings_percent' in all_results['memory_efficiency']:
        print(f"\n💾 MEMORY EFFICIENCY:")
        print(f"   Token savings: {all_results['memory_efficiency']['savings_percent']:.1f}%")
        print(f"   Prompt growth: {all_results['memory_efficiency']['prompt_growth_percent']:.1f}%")

    if 'accuracy' in all_results and 'accuracy_percent' in all_results['accuracy']:
        print(f"\n🎯 RETRIEVAL ACCURACY:")
        print(f"   Semantic accuracy: {all_results['accuracy']['accuracy_percent']:.1f}%")
        print(f"   Avg similarity: {all_results['accuracy']['avg_similarity']:.3f}")

    if 'comparative' in all_results and 'comparison' in all_results['comparative']:
        print(f"\n🆚 RETRIEVAL VS SUMMARIZATION:")
        print(f"   Token savings: {all_results['comparative']['comparison']['token_savings']:.1f}%")
        print(f"   Retrieval accuracy: {all_results['comparative']['retrieval']['accuracy']:.1%}")
        print(f"   Summary accuracy: {all_results['comparative']['summary']['accuracy']:.1%}")
        print(f"   Accuracy gain: {all_results['comparative']['comparison']['accuracy_diff']*100:+.1f}%")

    print("\n" + "="*80)
    print("✅ BENCHMARK COMPLETE")
    print("="*80)

    return all_results


if __name__ == "__main__":
    results = asyncio.run(run_all_benchmarks())
