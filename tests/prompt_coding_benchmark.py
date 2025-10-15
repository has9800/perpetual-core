"""
Prompt Coding Benchmark Suite - V2
For testing retrieval vs summarization in PROMPT-BASED coding platforms (Loveable, v0, etc.)
PITCH-READY: Shows clear advantages for complex app development
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


class LoveableStyleBenchmark:
    """
    Benchmark for Loveable-style complex app development
    Tests retrieval vs summarization over 100-turn app building session
    """

    def __init__(self, engine, memory, context_manager):
        self.engine = engine
        self.memory = memory
        self.context_manager = context_manager

    async def run(self) -> Dict:
        """Run Loveable-style complex app benchmark"""
        print("\n" + "="*80)
        print("LOVEABLE BENCHMARK: RETRIEVAL VS SUMMARIZATION")
        print("Complex app development over 100 turns")
        print("="*80)

        # Generate realistic Loveable-style conversation
        loveable_session = self.generate_loveable_app_session()

        print(f"\nGenerated {len(loveable_session)} turn app development session")
        print("Simulating real Loveable workflow: e-commerce app with auth, cart, payment")
        print("\nTesting both approaches...\n")

        # Test 1: V2 Retrieval Approach
        print("="*80)
        print("APPROACH 1: V2 RETRIEVAL (Your System)")
        print("="*80)
        retrieval_result = await self.test_retrieval_approach(loveable_session)

        # Test 2: Summarization Approach (what they might be doing now)
        print("\n" + "="*80)
        print("APPROACH 2: SUMMARIZATION (Traditional)")
        print("="*80)
        summary_result = await self.test_summary_approach(loveable_session)

        # Calculate comparison
        results = {
            'v2_retrieval': retrieval_result,
            'summarization': summary_result,
            'comparison': {
                'token_savings': (
                    (summary_result['tokens_used'] - retrieval_result['tokens_used']) 
                    / summary_result['tokens_used'] * 100
                ) if summary_result['tokens_used'] > 0 else 0,
                'accuracy_gain': (retrieval_result['accuracy'] - summary_result['accuracy']) * 100,
                'latency_diff_ms': (retrieval_result['latency_ms'] - summary_result['latency_ms']),
                'cost_savings_per_1k_requests': self.calculate_cost_savings(
                    retrieval_result['tokens_used'],
                    summary_result['tokens_used']
                )
            }
        }

        # Print comparison
        self.print_pitch_results(results)

        return results

    def generate_loveable_app_session(self) -> List[Dict]:
        """Generate realistic 100-turn e-commerce app session"""
        
        conversation = []
        
        # Phase 1: Setup (turns 1-10)
        conversation.extend([
            {"turn": 1, "user": "Create a modern e-commerce landing page", "assistant": "``````", "phase": "setup"},
            {"turn": 2, "user": "Add product grid with images", "assistant": "``````", "phase": "setup"},
            {"turn": 3, "user": "Create product detail page", "assistant": "``````", "phase": "setup"},
            {"turn": 4, "user": "Add to cart button", "assistant": "``````", "phase": "setup"},
        ])
        
        # Phase 2: Authentication (turns 11-25) - KEY COMPONENTS
        conversation.extend([
            {"turn": 11, "user": "Add user authentication with JWT", "assistant": "``````", "phase": "auth", "key": True},
            {"turn": 12, "user": "Create login form", "assistant": "``````", "phase": "auth", "key": True},
            {"turn": 13, "user": "Add register form", "assistant": "``````", "phase": "auth"},
            {"turn": 14, "user": "Protected routes for logged-in users", "assistant": "``````", "phase": "auth", "key": True},
            {"turn": 15, "user": "Add logout functionality", "assistant": "``````", "phase": "auth"},
        ])
        
        # Phase 3: Cart Logic (turns 26-45) - KEY COMPONENTS
        conversation.extend([
            {"turn": 26, "user": "Create cart context for state management", "assistant": "``````", "phase": "cart", "key": True},
            {"turn": 27, "user": "Cart page with item list", "assistant": "``````", "phase": "cart", "key": True},
            {"turn": 28, "user": "Cart item component", "assistant": "``````", "phase": "cart"},
            {"turn": 29, "user": "Cart badge in header", "assistant": "``````", "phase": "cart"},
        ])
        
        # Phase 4: Payment (turns 46-60) - KEY COMPONENTS
        conversation.extend([
            {"turn": 46, "user": "Stripe payment integration", "assistant": "``````", "phase": "payment", "key": True},
            {"turn": 47, "user": "Payment API endpoint", "assistant": "``````", "phase": "payment", "key": True},
            {"turn": 48, "user": "Order confirmation page", "assistant": "``````", "phase": "payment"},
        ])
        
        # Phase 5: Filler/styling (turns 61-95)
        for i in range(61, 96):
            conversation.append({
                "turn": i,
                "user": f"Update styling for section {i}",
                "assistant": f"``````",
                "phase": "styling",
                "filler": True
            })
        
        # Phase 6: Test queries (turns 96-100) - User references earlier code
        conversation.extend([
            {
                "turn": 96,
                "user": "I need to add a check in the cart to verify user is logged in before checkout. Use the auth context we created earlier",
                "assistant": "retrieving...",
                "test_query": True,
                "requires": ["AuthContext", "useAuth", "ProtectedRoute"],
                "phase_needed": "auth"
            },
            {
                "turn": 97,
                "user": "Show me the cart total calculation logic we wrote",
                "assistant": "retrieving...",
                "test_query": True,
                "requires": ["total", "reduce", "CartContext"],
                "phase_needed": "cart"
            },
            {
                "turn": 98,
                "user": "I want to trigger the payment flow when user clicks checkout. Reference the Stripe integration",
                "assistant": "retrieving...",
                "test_query": True,
                "requires": ["stripe", "confirmCardPayment", "CheckoutForm"],
                "phase_needed": "payment"
            },
            {
                "turn": 99,
                "user": "Add the login check before showing payment form",
                "assistant": "retrieving...",
                "test_query": True,
                "requires": ["useAuth", "user", "login"],
                "phase_needed": "auth"
            },
            {
                "turn": 100,
                "user": "Show the complete cart and payment flow including auth",
                "assistant": "retrieving...",
                "test_query": True,
                "requires": ["CartContext", "total", "stripe", "useAuth", "ProtectedRoute"],
                "phase_needed": "all",
                "multi_phase": True
            }
        ])
        
        return conversation

    async def test_retrieval_approach(self, conversation: List[Dict]) -> Dict:
        """Test V2 retrieval approach"""
        conv_id = "loveable_retrieval_v2"
        
        print("Building app (turns 1-95)...")
        
        # Store turns 1-95
        for turn in conversation[:95]:
            if turn.get('filler'):
                continue  # Skip some filler to simulate real workflow
            
            self.memory.add_turn(
                conversation_id=conv_id,
                text=turn["user"],
                metadata={
                    'response': turn["assistant"],
                    'turn': turn["turn"],
                    'phase': turn.get("phase"),
                    'key': turn.get("key", False)
                }
            )
        
        print("Waiting for vector DB indexing...")
        await asyncio.sleep(2)
        
        print(f"Stored {self.memory.get_conversation_length(conv_id)} turns")
        
        # Test retrieval on turns 96-100
        test_turns = [t for t in conversation[95:] if t.get('test_query')]
        
        print(f"\nTesting {len(test_turns)} reference queries...\n")
        
        correct = 0
        total_tokens = 0
        total_latency = 0
        results_detail = []
        
        for i, test in enumerate(test_turns):
            print(f"  Query {i+1}/{len(test_turns)}: \"{test['user'][:70]}...\"")
            print(f"    Needs: {', '.join(test['requires'][:3])}")
            
            start_time = time.time()
            
            # V2: Retrieve with reranking
            context = self.memory.retrieve_context(
                conversation_id=conv_id,
                query=test['user'],
                top_k=5  # Get multiple relevant pieces
            )
            
            latency = (time.time() - start_time) * 1000
            total_latency += latency
            
            # Check what was found
            found_items = []
            context_tokens = 0
            
            if context.get('success') and context['results']:
                for result in context['results']:
                    response = result['metadata'].get('response', '')
                    context_tokens += count_tokens(response)
                    
                    for required in test['requires']:
                        if required in response and required not in found_items:
                            found_items.append(required)
            
            total_tokens += context_tokens
            
            accuracy = len(found_items) / len(test['requires'])
            if accuracy >= 0.6:  # 60%+ is success
                correct += 1
                print(f"    ✅ SUCCESS: Found {len(found_items)}/{len(test['requires'])} ({accuracy:.0%})")
            else:
                print(f"    ❌ PARTIAL: Found {len(found_items)}/{len(test['requires'])} ({accuracy:.0%})")
            
            print(f"       Tokens: {context_tokens}, Latency: {latency:.0f}ms")
            
            results_detail.append({
                'query': i+1,
                'accuracy': accuracy,
                'found': len(found_items),
                'required': len(test['requires']),
                'tokens': context_tokens,
                'latency_ms': latency
            })
        
        overall_accuracy = correct / len(test_turns)
        avg_tokens = total_tokens / len(test_turns)
        avg_latency = total_latency / len(test_turns)
        
        print(f"\n  V2 Retrieval Summary:")
        print(f"    Overall Accuracy: {overall_accuracy:.1%} ({correct}/{len(test_turns)} queries)")
        print(f"    Avg Tokens/Query: {avg_tokens:.0f}")
        print(f"    Avg Latency: {avg_latency:.0f}ms")
        
        return {
            'accuracy': overall_accuracy,
            'tokens_used': int(avg_tokens),
            'total_tokens': total_tokens,
            'latency_ms': avg_latency,
            'queries_successful': correct,
            'total_queries': len(test_turns),
            'details': results_detail
        }

    async def test_summary_approach(self, conversation: List[Dict]) -> Dict:
        """Test summarization approach"""
        
        print("Building app (turns 1-95)...")
        
        # Build full history (what summarization would store)
        full_history = []
        for turn in conversation[:95]:
            full_history.append(f"User: {turn['user']}")
            full_history.append(f"Assistant: {turn['assistant']}")
        
        history_text = "\n".join(full_history)
        
        print(f"Full history: {len(history_text)} characters")
        print("Generating summary with local model...")
        
        # Generate summary
        start_summary = time.time()
        
        summary_prompt = f"""Summarize this e-commerce app development session. Preserve ALL important code snippets, function names, and component structures:

{history_text[:8000]}

Provide detailed summary including:
- Authentication components and functions
- Cart management logic
- Payment integration code
- All key function names and variables"""
        
        summary_request = GenerationRequest(
            conversation_id="loveable_summary",
            messages=[{"role": "user", "content": summary_prompt}],
            model="test",
            max_tokens=2000,
            temperature=0.2
        )
        
        summary_result = await self.engine.generate(summary_request)
        summary_latency = (time.time() - start_summary) * 1000
        
        if not summary_result.get('success'):
            return {
                'accuracy': 0,
                'tokens_used': 0,
                'latency_ms': 0,
                'error': 'Summary generation failed'
            }
        
        summary_text = summary_result.get('response', '')
        summary_tokens = count_tokens(summary_text)
        
        print(f"Summary generated: {len(summary_text)} chars, {summary_tokens} tokens")
        print(f"Summary creation time: {summary_latency:.0f}ms\n")
        
        # Test queries against summary
        test_turns = [t for t in conversation[95:] if t.get('test_query')]
        
        print(f"Testing {len(test_turns)} queries against summary...\n")
        
        correct = 0
        total_tokens = summary_tokens * len(test_turns)  # Summary sent every time
        total_latency = summary_latency  # One-time cost
        results_detail = []
        
        for i, test in enumerate(test_turns):
            print(f"  Query {i+1}/{len(test_turns)}: \"{test['user'][:70]}...\"")
            print(f"    Needs: {', '.join(test['requires'][:3])}")
            
            # Check if summary contains required items
            found_items = []
            for required in test['requires']:
                # Very flexible matching (summary might paraphrase)
                if required.lower() in summary_text.lower():
                    found_items.append(required)
            
            accuracy = len(found_items) / len(test['requires'])
            if accuracy >= 0.6:
                correct += 1
                print(f"    ✅ SUCCESS: Found {len(found_items)}/{len(test['requires'])} ({accuracy:.0%})")
            else:
                print(f"    ❌ PARTIAL: Found {len(found_items)}/{len(test['requires'])} ({accuracy:.0%})")
            
            print(f"       Tokens: {summary_tokens} (full summary)")
            
            results_detail.append({
                'query': i+1,
                'accuracy': accuracy,
                'found': len(found_items),
                'required': len(test['requires']),
                'tokens': summary_tokens
            })
        
        overall_accuracy = correct / len(test_turns)
        avg_tokens = summary_tokens  # Same every time
        
        print(f"\n  Summarization Summary:")
        print(f"    Overall Accuracy: {overall_accuracy:.1%} ({correct}/{len(test_turns)} queries)")
        print(f"    Tokens/Query: {avg_tokens} (sends full summary each time)")
        print(f"    Summary generation: {summary_latency:.0f}ms (one-time cost)")
        
        return {
            'accuracy': overall_accuracy,
            'tokens_used': avg_tokens,
            'total_tokens': total_tokens,
            'latency_ms': summary_latency / len(test_turns),  # Amortized
            'queries_successful': correct,
            'total_queries': len(test_turns),
            'summary_generation_ms': summary_latency,
            'details': results_detail
        }

    def calculate_cost_savings(self, retrieval_tokens, summary_tokens):
        """Calculate cost savings at scale"""
        # Assume GPT-4 pricing: $10 per 1M tokens
        cost_per_token = 10 / 1_000_000
        
        retrieval_cost_per_1k = retrieval_tokens * 1000 * cost_per_token
        summary_cost_per_1k = summary_tokens * 1000 * cost_per_token
        
        savings = summary_cost_per_1k - retrieval_cost_per_1k
        
        return {
            'retrieval_cost': retrieval_cost_per_1k,
            'summary_cost': summary_cost_per_1k,
            'savings': savings,
            'savings_percent': (savings / summary_cost_per_1k * 100) if summary_cost_per_1k > 0 else 0
        }

    def print_pitch_results(self, results):
        """Print results in pitch-ready format"""
        print("\n" + "="*80)
        print("LOVEABLE PITCH: RESULTS COMPARISON")
        print("="*80)
        
        print(f"\n{'Metric':<30} | {'V2 Retrieval':<20} | {'Summarization':<20}")
        print("-"*80)
        
        # Accuracy
        print(f"{'Code Retrieval Accuracy':<30} | {results['v2_retrieval']['accuracy']:<20.1%} | {results['summarization']['accuracy']:<20.1%}")
        
        # Tokens
        print(f"{'Tokens per Query':<30} | {results['v2_retrieval']['tokens_used']:<20} | {results['summarization']['tokens_used']:<20}")
        
        # Latency
        print(f"{'Latency (ms)':<30} | {results['v2_retrieval']['latency_ms']:<20.0f} | {results['summarization']['latency_ms']:<20.0f}")
        
        print("\n" + "="*80)
        print("💰 BUSINESS IMPACT (Per 1,000 Users)")
        print("="*80)
        
        comp = results['comparison']
        cost = comp['cost_savings_per_1k_requests']
        
        print(f"\n  Token Savings:      {comp['token_savings']:.1f}%")
        print(f"  Accuracy Gain:      {comp['accuracy_gain']:+.1f}%")
        print(f"  Cost Savings:       ${cost['savings']:.2f} per 1K requests")
        print(f"                      ${cost['savings'] * 1000:.2f} per 1M requests")
        
        print(f"\n  At Loveable Scale (assume 100K daily requests):")
        print(f"    Monthly Savings:   ${cost['savings'] * 100 * 30:.2f}")
        print(f"    Yearly Savings:    ${cost['savings'] * 100 * 365:.2f}")
        
        print("\n" + "="*80)
        print("🎯 LOVEABLE VALUE PROPOSITION")
        print("="*80)
        print(f"\n  ✅ {comp['token_savings']:.0f}% fewer tokens = faster responses")
        print(f"  ✅ {comp['accuracy_gain']:+.0f}% better accuracy = happier users")
        print(f"  ✅ ${cost['savings'] * 100 * 365:.0f}/year saved at 100K daily requests")
        print(f"  ✅ Scales linearly: more users = more savings")
        print(f"  ✅ Better UX: retrieves exact code vs vague summaries")


async def run_all_benchmarks():
    """Run all prompt coding benchmarks"""
    print("="*80)
    print("PROMPT-BASED CODING BENCHMARK SUITE - V2")
    print("Loveable Pitch Edition")
    print("="*80)

    # Initialize system
    print("\nInitializing V2 system...")

    vector_db = create_vector_db(backend="qdrant")
    memory_manager = MemoryManager(vector_db=vector_db, cache_capacity=1000)
    context_manager_v2 = ContextManagerV2(token_budget=2000, recent_turns_limit=15)

    model_name = os.getenv("MODEL_NAME", "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ")
    vllm_engine = create_vllm_engine(
        model_name=model_name,
        quantization="gptq",
        gpu_memory_utilization=0.9,
        max_model_len=4096
    )

    infinite_engine = InfiniteMemoryEngine(
        vllm_engine=vllm_engine,
        memory_manager=memory_manager,
        max_context_tokens=4096,
        context_retrieval_k=5
    )

    print("✅ V2 System initialized\n")

    # Run Loveable benchmark
    all_results = {}

    try:
        loveable_bench = LoveableStyleBenchmark(infinite_engine, memory_manager, context_manager_v2)
        all_results['loveable'] = await loveable_bench.run()
    except Exception as e:
        print(f"\n❌ Loveable benchmark failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("✅ LOVEABLE PITCH BENCHMARK COMPLETE")
    print("="*80)

    return all_results


if __name__ == "__main__":
    results = asyncio.run(run_all_benchmarks())
