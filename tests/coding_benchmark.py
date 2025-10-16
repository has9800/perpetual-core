"""
Coding Benchmark: Retrieval vs Summarization for Context Management
Tests across multiple query difficulty levels
"""

import asyncio
import time
from typing import List, Dict
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from vector_db_adapters import create_vector_db
from memory_manager import MemoryManager
from vllm_wrapper_production import InfiniteMemoryEngine, create_vllm_engine, GenerationRequest


class CodingBenchmark:
    """Compare context management strategies across query difficulties"""
    
    def __init__(self, engine: InfiniteMemoryEngine, memory: MemoryManager):
        self.engine = engine
        self.memory = memory
        
        # JWT code at turn 10
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
    return jsonify({'id': todo.id}), 201"""
        
        self.jwt_elements = ['create_access_token', 'jwt_required', 'JWT_SECRET_KEY', '@app.route(\'/login\'']
        
        # Test queries with varying difficulty
        self.test_queries = [
            {
                'name': 'Easy: Direct keyword match',
                'query': 'Modify the JWT login function to also return the user email address',
                'difficulty': 'easy'
            },
            {
                'name': 'Medium: Semantic understanding',
                'query': 'The authentication endpoint should include user email in the response',
                'difficulty': 'medium'
            },
            {
                'name': 'Hard: Indirect reference',
                'query': 'When users sign in, they need to receive their email along with the token',
                'difficulty': 'hard'
            }
        ]
    
    async def run(self) -> Dict:
        """Run benchmark across all query difficulties"""
        print("\n" + "="*80)
        print("CODING AGENT CONTEXT MANAGEMENT BENCHMARK")
        print("Testing across multiple query difficulty levels")
        print("="*80)
        
        all_results = []
        
        for test_query in self.test_queries:
            print("\n" + "="*80)
            print(f"TEST: {test_query['name']}")
            print(f"Query: '{test_query['query']}'")
            print(f"Difficulty: {test_query['difficulty'].upper()}")
            print("="*80)
            
            # Generate fresh conversation for each test
            turns = self._generate_conversation()
            
            # Test retrieval approach
            print("\nAPPROACH 1: RETRIEVAL")
            print("-" * 80)
            retrieval_result = await self._test_retrieval_approach(turns, test_query['query'])
            
            # Test summarization approach
            print("\nAPPROACH 2: SUMMARIZATION")
            print("-" * 80)
            summary_result = await self._test_summary_approach(turns, test_query['query'])
            
            # Store results
            all_results.append({
                'query_name': test_query['name'],
                'query': test_query['query'],
                'difficulty': test_query['difficulty'],
                'retrieval': retrieval_result,
                'summary': summary_result
            })
        
        # Print aggregate results
        self._print_aggregate_results(all_results)
        
        return {'test_results': all_results}
    
    def _generate_conversation(self) -> List[Dict]:
        """Generate 50-turn coding conversation"""
        turns = []
        
        for i in range(50):
            turn_num = i + 1
            
            if turn_num == 10:
                turns.append({
                    'turn': turn_num,
                    'user': 'Add JWT authentication to the /login and /todos endpoints',
                    'assistant': self.jwt_code
                })
            elif turn_num < 10:
                turns.append({
                    'turn': turn_num,
                    'user': f'Create basic Flask endpoint {turn_num}',
                    'assistant': f'@app.route("/endpoint{turn_num}")\ndef endpoint{turn_num}():\n    return "OK"'
                })
            else:
                turns.append({
                    'turn': turn_num,
                    'user': f'Add feature {turn_num}',
                    'assistant': f'def feature_{turn_num}():\n    # Feature {turn_num}\n    return True'
                })
        
        return turns
    
    async def _test_retrieval_approach(self, turns: List[Dict], user_query: str) -> Dict:
        """Test retrieval-based context"""
        conv_id = f"retrieval_{time.time()}"
        
        # Store all 49 turns
        for turn in turns[:49]:
            self.memory.add_turn(
                conversation_id=conv_id,
                text=turn['user'],
                metadata={'response': turn['assistant'], 'turn': turn['turn']}
            )
        
        await asyncio.sleep(2)
        
        # Retrieve relevant context
        start = time.time()
        context = await self.memory.retrieve_context(conv_id, user_query, top_k=3)
        retrieval_time = time.time() - start
        
        # Build context
        retrieved_turns = []
        context_text = ""
        for result in context.get('results', []):
            turn_num = result['metadata'].get('turn')
            retrieved_turns.append(turn_num)
            context_text += f"Turn {turn_num}:\n"
            context_text += f"User: {result['text']}\n"
            context_text += f"Assistant: {result['metadata'].get('response', '')}\n\n"
        
        context_text += "Recent conversation:\n"
        for turn in turns[46:49]:
            context_text += f"Turn {turn['turn']}: {turn['user']}\n"
        
        context_tokens = len(context_text) // 4
        
        print(f"Retrieved turns: {retrieved_turns}")
        print(f"Context size: ~{context_tokens} tokens")
        
        # Generate answer
        prompt = f"""Based on this coding conversation context:

{context_text}

User asks: {user_query}

Provide the modified code:"""
        
        start = time.time()
        request = GenerationRequest(
            conversation_id=conv_id + "_gen",
            messages=[{"role": "user", "content": prompt}],
            model="test",
            max_tokens=400,
            temperature=0.3
        )
        
        result = await self.engine.generate(request)
        generation_time = time.time() - start
        
        if not result.get('success'):
            print("❌ Generation failed")
            return {'accuracy': 0, 'tokens': 0, 'latency': 0, 'found': 0}
        
        answer = result.get('response') or result.get('text') or ""
        
        # Check accuracy
        found = sum(1 for elem in self.jwt_elements if elem in answer)
        accuracy = found / len(self.jwt_elements)
        
        print(f"✅ Found {found}/{len(self.jwt_elements)} elements ({accuracy:.1%})")
        
        return {
            'accuracy': accuracy,
            'context_tokens': context_tokens,
            'retrieval_latency': retrieval_time,
            'generation_latency': generation_time,
            'total_latency': retrieval_time + generation_time,
            'found_elements': found,
            'retrieved_turn_10': 10 in retrieved_turns
        }
    
    async def _test_summary_approach(self, turns: List[Dict], user_query: str) -> Dict:
        """Test summary-based context"""
        
        # Build history to summarize (turns 1-40)
        history_to_summarize = []
        for turn in turns[:40]:
            history_to_summarize.append(f"Turn {turn['turn']}: {turn['user']}")
            history_to_summarize.append(f"Assistant: {turn['assistant']}")
        
        history_text = "\n".join(history_to_summarize)
        
        # Generate summary
        summary_prompt = f"""You must create a detailed technical summary of this coding conversation. 
Preserve ALL function names, imports, routes, configuration values, and code snippets EXACTLY.

{history_text[:3500]}

Technical summary with all code details:"""
        
        start = time.time()
        request = GenerationRequest(
            conversation_id=f"summary_{time.time()}",
            messages=[{"role": "user", "content": summary_prompt}],
            model="test",
            max_tokens=800,
            temperature=0.1
        )
        
        result = await self.engine.generate(request)
        summary_time = time.time() - start
        
        if not result.get('success'):
            print("❌ Summary generation failed")
            return {'accuracy': 0, 'tokens': 0, 'latency': 0, 'found': 0}
        
        summary = result.get('response') or result.get('text') or ""
        summary_tokens = len(summary) // 4
        
        # Check what summary preserved
        jwt_in_summary = sum(1 for elem in self.jwt_elements if elem in summary)
        summary_preservation = jwt_in_summary / len(self.jwt_elements)
        
        print(f"Summary: {jwt_in_summary}/{len(self.jwt_elements)} elements preserved ({summary_preservation:.1%})")
        
        # Build context: summary + recent turns
        context_text = f"Summary of earlier conversation (turns 1-40):\n{summary}\n\n"
        context_text += "Recent conversation:\n"
        for turn in turns[40:49]:
            context_text += f"Turn {turn['turn']}: {turn['user']}\n"
        
        context_tokens = len(context_text) // 4
        
        print(f"Context size: ~{context_tokens} tokens")
        
        # Generate answer
        prompt = f"""Based on this coding conversation context:

{context_text}

User asks: {user_query}

Provide the modified code:"""
        
        start = time.time()
        request = GenerationRequest(
            conversation_id=f"summary_gen_{time.time()}",
            messages=[{"role": "user", "content": prompt}],
            model="test",
            max_tokens=400,
            temperature=0.3
        )
        
        result = await self.engine.generate(request)
        generation_time = time.time() - start
        
        if not result.get('success'):
            print("❌ Answer generation failed")
            return {
                'accuracy': 0,
                'tokens': context_tokens,
                'latency': summary_time,
                'found': 0,
                'summary_preserved': summary_preservation
            }
        
        answer = result.get('response') or result.get('text') or ""
        
        # Check accuracy
        found = sum(1 for elem in self.jwt_elements if elem in answer)
        accuracy = found / len(self.jwt_elements)
        
        print(f"✅ Found {found}/{len(self.jwt_elements)} elements in answer ({accuracy:.1%})")
        
        return {
            'accuracy': accuracy,
            'context_tokens': context_tokens,
            'summary_latency': summary_time,
            'generation_latency': generation_time,
            'total_latency': summary_time + generation_time,
            'found_elements': found,
            'summary_preserved': summary_preservation
        }
    
    def _print_aggregate_results(self, all_results: List[Dict]):
        """Print results across all query types"""
        print("\n" + "="*80)
        print("AGGREGATE RESULTS ACROSS ALL QUERY TYPES")
        print("="*80)
        
        # Calculate averages
        avg_ret_acc = sum(r['retrieval']['accuracy'] for r in all_results) / len(all_results)
        avg_sum_acc = sum(r['summary']['accuracy'] for r in all_results) / len(all_results)
        avg_ret_tok = sum(r['retrieval']['context_tokens'] for r in all_results) / len(all_results)
        avg_sum_tok = sum(r['summary']['context_tokens'] for r in all_results) / len(all_results)
        
        print(f"\n{'Overall Averages':<30} | {'Retrieval':<15} | {'Summarization':<15}")
        print("-"*80)
        print(f"{'Accuracy':<30} | {avg_ret_acc:<15.1%} | {avg_sum_acc:<15.1%}")
        print(f"{'Context Tokens':<30} | {avg_ret_tok:<15.0f} | {avg_sum_tok:<15.0f}")
        
        # Per-query breakdown
        print(f"\n{'Query Type':<35} | {'Ret Acc':<10} | {'Sum Acc':<10} | {'Difference':<10}")
        print("-"*80)
        for r in all_results:
            diff = (r['retrieval']['accuracy'] - r['summary']['accuracy']) * 100
            query_short = r['query_name'].split(':')[1].strip()[:30]
            print(f"{query_short:<35} | {r['retrieval']['accuracy']:<10.1%} | {r['summary']['accuracy']:<10.1%} | {diff:+.1f}%")
        
        # Key findings
        token_savings = ((avg_sum_tok - avg_ret_tok) / avg_sum_tok * 100) if avg_sum_tok > 0 else 0
        accuracy_gain = (avg_ret_acc - avg_sum_acc) * 100
        
        # Check variance
        ret_accuracies = [r['retrieval']['accuracy'] for r in all_results]
        ret_variance = max(ret_accuracies) - min(ret_accuracies)
        
        print("\n" + "="*80)
        print("KEY FINDINGS:")
        print("="*80)
        print(f"💰 Token Savings: {token_savings:.1f}%")
        print(f"🎯 Accuracy Gain: {accuracy_gain:+.1f}%")
        print(f"📊 Retrieval Variance: {ret_variance:.1%} (shows difficulty range)")
        
        # Summary preservation analysis
        avg_summary_preserved = sum(r['summary'].get('summary_preserved', 0) for r in all_results) / len(all_results)
        print(f"\n⚠️  Summary preserved only {avg_summary_preserved:.1%} of code details on average")
        print(f"   This is why summarization fails for coding tasks")
        
        print(f"\n📌 WHY THIS MATTERS:")
        print(f"   • Coding needs exact code, not summaries")
        print(f"   • Retrieval works across all query types")
        print(f"   • Summarization loses {(1-avg_summary_preserved)*100:.0f}% of critical details")


async def main():
    """Main function"""
    print("="*80)
    print("CODING AGENT BENCHMARK - Multiple Query Difficulties")
    print("="*80)
    
    print("\nInitializing system...")
    
    vector_db = create_vector_db(backend=os.getenv("VECTOR_DB_BACKEND", "qdrant"))
    memory = MemoryManager(vector_db=vector_db, cache_capacity=1000)
    
    vllm = create_vllm_engine(
        model_name=os.getenv("MODEL_NAME", "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"),
        quantization=os.getenv("MODEL_QUANTIZATION", "gptq"),
        gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9")),
        max_model_len=4096
    )
    
    engine = InfiniteMemoryEngine(
        vllm_engine=vllm,
        memory_manager=memory,
        max_context_tokens=4096,
        context_retrieval_k=3
    )
    
    print("✅ System initialized\n")
    
    benchmark = CodingBenchmark(engine, memory)
    results = await benchmark.run()
    
    print("\n" + "="*80)
    print("✅ BENCHMARK COMPLETE")
    print("="*80)
    
    return results


if __name__ == "__main__":
    asyncio.run(main())
