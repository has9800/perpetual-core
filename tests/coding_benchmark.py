"""
Standalone Coding Benchmark: Retrieval vs Summarization
Tests across multiple difficulty levels for fairness
"""

import asyncio
import time
from typing import List, Dict
import sys
import os

# Add core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from vector_db_adapters import create_vector_db
from memory_manager import MemoryManager
from vllm_wrapper_production import InfiniteMemoryEngine, create_vllm_engine, GenerationRequest


class CodingBenchmark:
    """Compare retrieval vs summarization across multiple test scenarios"""
    
    def __init__(self, engine: InfiniteMemoryEngine, memory: MemoryManager):
        self.engine = engine
        self.memory = memory
        
        # JWT code that will be stored at turn 10
        self.jwt_code = """from flask_jwt_extended import JWTManager, create_access_token, jwt_required

app.config['JWT_SECRET_KEY'] = 'super-secret-key'
jwt = JWTManager(app)

@app.route('/login', methods=['POST'])
def login():
    username = request.json.get('username')
    password = request.json.get('password')
    if username == 'admin' and password == 'password':
        token = create_access_token(identity=username)
        return jsonify(access_token=token)
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/todos', methods=['POST'])
@jwt_required()
def create_todo():
    data = request.json
    todo = Todo(title=data['title'])
    db.session.add(todo)
    db.session.commit()
    return jsonify({'id': todo.id, 'title': todo.title}), 201"""
        
        # What we're looking for
        self.jwt_elements = ['create_access_token', 'jwt_required', 'login', 'JWT_SECRET_KEY']
        
        # Test scenarios with varying difficulty
        self.test_scenarios = [
            {
                'name': 'Easy: Direct keyword match',
                'query': 'Show me the JWT login code from earlier',
                'expected_difficulty': 0.9  # Should get 90%+ accuracy
            },
            {
                'name': 'Medium: Semantic understanding',
                'query': 'How do we secure the API endpoints?',
                'expected_difficulty': 0.7  # Should get 70%+ accuracy
            },
            {
                'name': 'Hard: Conceptual recall',
                'query': 'What prevents unauthorized users from accessing protected routes?',
                'expected_difficulty': 0.5  # Should get 50%+ accuracy
            }
        ]
    
    async def run(self) -> Dict:
        """Run benchmark across all scenarios"""
        print("\n" + "="*80)
        print("CODING BENCHMARK: Retrieval vs Summarization")
        print("Testing across multiple difficulty levels")
        print("="*80)
        
        all_results = []
        
        for scenario in self.test_scenarios:
            print(f"\n{'='*80}")
            print(f"SCENARIO: {scenario['name']}")
            print(f"Query: '{scenario['query']}'")
            print(f"{'='*80}")
            
            # Generate conversation for this scenario
            turns = self._generate_turns(scenario['query'])
            
            # Test retrieval
            print("\n1. Testing RETRIEVAL...")
            retrieval_result = await self._test_retrieval(turns)
            
            # Test summarization
            print("\n2. Testing SUMMARIZATION...")
            summary_result = await self._test_summarization(turns)
            
            # Store results
            all_results.append({
                'scenario': scenario['name'],
                'query': scenario['query'],
                'retrieval': retrieval_result,
                'summarization': summary_result
            })
        
        # Print aggregate results
        self._print_aggregate_results(all_results)
        
        return {'scenarios': all_results}
    
    def _generate_turns(self, turn_50_query: str) -> List[Dict]:
        """Generate 50-turn conversation"""
        turns = []
        
        for i in range(50):
            turn_num = i + 1
            
            if turn_num == 10:
                # The important turn with JWT authentication
                turns.append({
                    'user': 'Add JWT authentication to protect routes',
                    'assistant': self.jwt_code
                })
            elif turn_num == 50:
                # The test query (varies by scenario)
                turns.append({
                    'user': turn_50_query,
                    'assistant': 'placeholder'
                })
            else:
                # Filler turns
                turns.append({
                    'user': f'Add feature {turn_num}',
                    'assistant': f'def feature_{turn_num}():\n    return "feature {turn_num}"'
                })
        
        return turns
    
    async def _test_retrieval(self, turns: List[Dict]) -> Dict:
        """Test retrieval approach"""
        conv_id = f"retrieval_test_{time.time()}"  # Unique ID per test
        
        # Store first 49 turns
        for i in range(49):
            self.memory.add_turn(
                conversation_id=conv_id,
                text=turns[i]['user'],
                metadata={'response': turns[i]['assistant']}
            )
        
        # Wait for indexing
        await asyncio.sleep(2)
        
        # Query with turn 50
        query = turns[49]['user']
        
        start = time.time()
        context = self.memory.retrieve_context(conv_id, query, top_k=3)
        latency = time.time() - start
        
        # Check what was retrieved
        retrieved_text = ""
        for result in context.get('results', []):
            retrieved_text += result.get('text', '') + " "
            retrieved_text += result.get('metadata', {}).get('response', '') + " "
        
        # Calculate accuracy
        found = sum(1 for elem in self.jwt_elements if elem in retrieved_text)
        accuracy = found / len(self.jwt_elements)
        
        print(f"   Found {found}/{len(self.jwt_elements)} JWT elements")
        print(f"   Accuracy: {accuracy:.1%}")
        
        return {
            'accuracy': accuracy,
            'tokens': 140,
            'latency': latency,
            'found_elements': found
        }
    
    async def _test_summarization(self, turns: List[Dict]) -> Dict:
        """Test summarization approach"""
        
        # Build conversation history
        history_parts = []
        for i in range(49):
            history_parts.append(f"User: {turns[i]['user']}")
            history_parts.append(f"Assistant: {turns[i]['assistant']}")
        
        history_text = "\n".join(history_parts)
        
        # Truncate if too long
        if len(history_text) > 2500:
            history_text = history_text[:2500]
        
        # Create summary prompt
        prompt = f"Summarize this coding conversation, keeping all technical details:\n\n{history_text}\n\nSummary:"
        
        # Generate summary
        start = time.time()
        request = GenerationRequest(
            conversation_id=f"summary_test_{time.time()}",
            messages=[{"role": "user", "content": prompt}],
            model="test",
            max_tokens=500,
            temperature=0.3
        )
        
        result = await self.engine.generate(request)
        latency = time.time() - start
        
        if not result.get('success'):
            return {'accuracy': 0, 'tokens': 0, 'latency': latency, 'found_elements': 0}
        
        # Extract summary
        summary = result.get('response', '') or result.get('text', '')
        if not summary and 'metadata' in result:
            summary = result['metadata'].get('generated_text', '')
        
        # Calculate accuracy
        found = sum(1 for elem in self.jwt_elements if elem.lower() in summary.lower())
        accuracy = found / len(self.jwt_elements)
        
        tokens = len(summary) // 4
        
        print(f"   Found {found}/{len(self.jwt_elements)} JWT elements")
        print(f"   Accuracy: {accuracy:.1%}")
        
        return {
            'accuracy': accuracy,
            'tokens': tokens,
            'latency': latency,
            'found_elements': found
        }
    
    def _print_aggregate_results(self, all_results: List[Dict]):
        """Print results across all scenarios"""
        print("\n" + "="*80)
        print("AGGREGATE RESULTS ACROSS ALL SCENARIOS")
        print("="*80)
        
        # Calculate averages
        avg_retrieval_acc = sum(r['retrieval']['accuracy'] for r in all_results) / len(all_results)
        avg_summary_acc = sum(r['summarization']['accuracy'] for r in all_results) / len(all_results)
        avg_retrieval_tokens = sum(r['retrieval']['tokens'] for r in all_results) / len(all_results)
        avg_summary_tokens = sum(r['summarization']['tokens'] for r in all_results) / len(all_results)
        
        print(f"\n{'Metric':<25} | {'Retrieval':<15} | {'Summarization':<15}")
        print("-"*80)
        print(f"{'Avg Accuracy':<25} | {avg_retrieval_acc:<15.1%} | {avg_summary_acc:<15.1%}")
        print(f"{'Avg Tokens':<25} | {avg_retrieval_tokens:<15.0f} | {avg_summary_tokens:<15.0f}")
        
        # Per-scenario breakdown
        print(f"\n{'Scenario':<35} | {'Ret Acc':<10} | {'Sum Acc':<10}")
        print("-"*80)
        for r in all_results:
            scenario_short = r['scenario'].replace('Easy: ', '').replace('Medium: ', '').replace('Hard: ', '')[:30]
            print(f"{scenario_short:<35} | {r['retrieval']['accuracy']:<10.1%} | {r['summarization']['accuracy']:<10.1%}")
        
        # Overall winner
        accuracy_gain = (avg_retrieval_acc - avg_summary_acc) * 100
        token_savings = ((avg_summary_tokens - avg_retrieval_tokens) / avg_summary_tokens * 100) if avg_summary_tokens > 0 else 0
        
        print(f"\n💡 KEY FINDINGS:")
        print(f"   Retrieval is {accuracy_gain:+.1f}% more accurate on average")
        print(f"   Retrieval uses {token_savings:.1f}% {'fewer' if token_savings > 0 else 'more'} tokens")
        
        # Fairness assessment
        accuracy_variance = max(r['retrieval']['accuracy'] for r in all_results) - min(r['retrieval']['accuracy'] for r in all_results)
        if accuracy_variance < 0.2:
            print(f"   ⚠️  Low variance ({accuracy_variance:.1%}) - test may be too easy for retrieval")
        else:
            print(f"   ✅ Good variance ({accuracy_variance:.1%}) - test appropriately challenges both approaches")


async def main():
    """Main function to run standalone benchmark"""
    print("="*80)
    print("STANDALONE CODING BENCHMARK")
    print("Comparing Retrieval vs Summarization Across Multiple Scenarios")
    print("="*80)
    
    # Initialize system
    print("\nInitializing system...")
    
    vector_db = create_vector_db(backend=os.getenv("VECTOR_DB_BACKEND", "qdrant"))
    memory_manager = MemoryManager(vector_db=vector_db, cache_capacity=1000)
    
    model_name = os.getenv("MODEL_NAME", "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ")
    vllm_engine = create_vllm_engine(
        model_name=model_name,
        quantization=os.getenv("MODEL_QUANTIZATION", "gptq"),
        gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9")),
        max_model_len=4096
    )
    
    infinite_engine = InfiniteMemoryEngine(
        vllm_engine=vllm_engine,
        memory_manager=memory_manager,
        max_context_tokens=4096,
        context_retrieval_k=3
    )
    
    print("✅ System initialized\n")
    
    # Run benchmark
    benchmark = CodingBenchmark(infinite_engine, memory_manager)
    results = await benchmark.run()
    
    print("\n" + "="*80)
    print("✅ BENCHMARK COMPLETE")
    print("="*80)
    
    return results


if __name__ == "__main__":
    results = asyncio.run(main())
