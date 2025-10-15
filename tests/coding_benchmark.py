"""
Coding Benchmark: Retrieval vs Summarization for Context Management
Simulates what coding agents do when context window fills up

Traditional approach: Summarize old context when window fills
Our approach: Retrieve exact code from vector DB
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
    """Compare context management strategies for coding agents"""
    
    def __init__(self, engine: InfiniteMemoryEngine, memory: MemoryManager):
        self.engine = engine
        self.memory = memory
        
        # JWT code at turn 10 (what we need to recall later)
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
    
    async def run(self) -> Dict:
        """Run benchmark comparing both approaches"""
        print("\n" + "="*80)
        print("CODING AGENT CONTEXT MANAGEMENT BENCHMARK")
        print("="*80)
        print("\nSimulating 50-turn coding session where user needs to recall")
        print("JWT code from turn 10 at turn 50")
        print("\nTraditional: Summarize turns 1-40, keep recent 41-49")
        print("Our approach: Retrieve exact turn 10 from vector DB")
        
        # Generate conversation
        turns = self._generate_conversation()
        user_query = "Modify the JWT login function to also return the user's email address"
        
        print(f"\nTurn 50 query: '{user_query}'")
        print("(Requires exact JWT code from turn 10 to answer correctly)")
        
        # Test 1: Your retrieval approach
        print("\n" + "="*80)
        print("APPROACH 1: RETRIEVAL (Your System)")
        print("="*80)
        retrieval_result = await self._test_retrieval_approach(turns, user_query)
        
        # Test 2: Traditional summarization approach  
        print("\n" + "="*80)
        print("APPROACH 2: SUMMARIZATION (Traditional Coding Agents)")
        print("="*80)
        summary_result = await self._test_summary_approach(turns, user_query)
        
        # Compare
        self._print_comparison(retrieval_result, summary_result)
        
        return {
            'retrieval': retrieval_result,
            'summarization': summary_result
        }
    
    def _generate_conversation(self) -> List[Dict]:
        """Generate 50-turn coding conversation"""
        turns = []
        
        for i in range(50):
            turn_num = i + 1
            
            if turn_num == 10:
                # Important: JWT authentication code
                turns.append({
                    'turn': turn_num,
                    'user': 'Add JWT authentication to the /login and /todos endpoints',
                    'assistant': self.jwt_code
                })
            elif turn_num < 10:
                # Early turns: basic setup
                turns.append({
                    'turn': turn_num,
                    'user': f'Create basic Flask endpoint {turn_num}',
                    'assistant': f'@app.route("/endpoint{turn_num}")\ndef endpoint{turn_num}():\n    return "OK"'
                })
            else:
                # Later turns: various features (filler)
                turns.append({
                    'turn': turn_num,
                    'user': f'Add feature {turn_num}',
                    'assistant': f'def feature_{turn_num}():\n    # Feature {turn_num}\n    return True'
                })
        
        return turns
    
    async def _test_retrieval_approach(self, turns: List[Dict], user_query: str) -> Dict:
        """Test retrieval-based context (your approach)"""
        conv_id = f"retrieval_{time.time()}"
        
        # Store all 49 turns in vector DB
        print("Storing 49 turns in vector DB...")
        for turn in turns[:49]:
            self.memory.add_turn(
                conversation_id=conv_id,
                text=turn['user'],
                metadata={'response': turn['assistant'], 'turn': turn['turn']}
            )
        
        await asyncio.sleep(2)  # Wait for indexing
        
        # Retrieve relevant context
        print("Retrieving relevant context...")
        start = time.time()
        context = self.memory.retrieve_context(conv_id, user_query, top_k=3)
        retrieval_time = time.time() - start
        
        # Build context for model
        retrieved_turns = []
        context_text = ""
        for result in context.get('results', []):
            turn_num = result['metadata'].get('turn')
            retrieved_turns.append(turn_num)
            context_text += f"Turn {turn_num}:\n"
            context_text += f"User: {result['text']}\n"
            context_text += f"Assistant: {result['metadata'].get('response', '')}\n\n"
        
        # Add recent turns (last 3)
        context_text += "Recent conversation:\n"
        for turn in turns[46:49]:
            context_text += f"Turn {turn['turn']}: {turn['user']}\n"
        
        context_tokens = len(context_text) // 4
        
        print(f"  Retrieved turns: {retrieved_turns}")
        print(f"  Context size: ~{context_tokens} tokens")
        
        # Ask model to answer with this context
        print("Generating answer with retrieved context...")
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
            print("  ❌ Generation failed")
            return {'accuracy': 0, 'tokens': 0, 'latency': 0, 'found': 0}
        
        answer = result.get('response') or result.get('text') or ""
        
        # Check if answer contains JWT code elements
        found = sum(1 for elem in self.jwt_elements if elem in answer)
        accuracy = found / len(self.jwt_elements)
        
        print(f"\n  ✅ Answer generated")
        print(f"  Found {found}/{len(self.jwt_elements)} JWT elements in answer")
        print(f"  Accuracy: {accuracy:.1%}")
        print(f"  Answer preview: {answer[:150]}...")
        
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
        """Test summary-based context (traditional approach)"""
        
        # Simulate hitting context limit at turn 40
        # Summarize turns 1-40, keep recent turns 41-49
        print("Simulating context window filling up...")
        print("Summarizing turns 1-40 (traditional approach when hitting token limit)...")
        
        # Build history to summarize
        history_to_summarize = []
        for turn in turns[:40]:
            history_to_summarize.append(f"Turn {turn['turn']}: {turn['user']}")
            history_to_summarize.append(f"Response: {turn['assistant']}")
        
        history_text = "\n".join(history_to_summarize)
        
        # Generate summary
        summary_prompt = f"""Summarize this coding conversation, preserving key technical details:

{history_text[:3000]}

Summary:"""
        
        start = time.time()
        request = GenerationRequest(
            conversation_id=f"summary_{time.time()}",
            messages=[{"role": "user", "content": summary_prompt}],
            model="test",
            max_tokens=600,
            temperature=0.3
        )
        
        result = await self.engine.generate(request)
        summary_time = time.time() - start
        
        if not result.get('success'):
            print("  ❌ Summary generation failed")
            return {'accuracy': 0, 'tokens': 0, 'latency': 0, 'found': 0}
        
        summary = result.get('response') or result.get('text') or ""
        summary_tokens = len(summary) // 4
        
        print(f"  Summary generated ({summary_tokens} tokens)")
        print(f"  Summary preview: {summary[:150]}...")
        
        # Build context: summary + recent turns
        context_text = f"Summary of earlier conversation (turns 1-40):\n{summary}\n\n"
        context_text += "Recent conversation:\n"
        for turn in turns[40:49]:
            context_text += f"Turn {turn['turn']}: {turn['user']}\n"
        
        context_tokens = len(context_text) // 4
        
        print(f"  Total context: ~{context_tokens} tokens (summary + recent turns)")
        
        # Ask model to answer with this context
        print("Generating answer with summarized context...")
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
            print("  ❌ Answer generation failed")
            return {'accuracy': 0, 'tokens': context_tokens, 'latency': summary_time, 'found': 0}
        
        answer = result.get('response') or result.get('text') or ""
        
        # Check if answer contains JWT code elements
        found = sum(1 for elem in self.jwt_elements if elem in answer)
        accuracy = found / len(self.jwt_elements)
        
        print(f"\n  ✅ Answer generated")
        print(f"  Found {found}/{len(self.jwt_elements)} JWT elements in answer")
        print(f"  Accuracy: {accuracy:.1%}")
        print(f"  Answer preview: {answer[:150]}...")
        
        return {
            'accuracy': accuracy,
            'context_tokens': context_tokens,
            'summary_latency': summary_time,
            'generation_latency': generation_time,
            'total_latency': summary_time + generation_time,
            'found_elements': found
        }
    
    def _print_comparison(self, retrieval: Dict, summary: Dict):
        """Print comparison results"""
        print("\n" + "="*80)
        print("RESULTS: Context Management Comparison")
        print("="*80)
        
        print(f"\n{'Metric':<30} | {'Retrieval':<15} | {'Summarization':<15}")
        print("-"*80)
        print(f"{'Code Accuracy':<30} | {retrieval['accuracy']:<15.1%} | {summary['accuracy']:<15.1%}")
        print(f"{'Context Tokens':<30} | {retrieval['context_tokens']:<15} | {summary['context_tokens']:<15}")
        print(f"{'Total Latency (ms)':<30} | {retrieval['total_latency']*1000:<15.0f} | {summary['total_latency']*1000:<15.0f}")
        print(f"{'Elements Found':<30} | {retrieval['found_elements']}/{len(self.jwt_elements):<14} | {summary['found_elements']}/{len(self.jwt_elements):<14}")
        
        # Calculate improvements
        token_savings = ((summary['context_tokens'] - retrieval['context_tokens']) 
                        / summary['context_tokens'] * 100) if summary['context_tokens'] > 0 else 0
        accuracy_gain = (retrieval['accuracy'] - summary['accuracy']) * 100
        
        print("\n" + "="*80)
        print("KEY FINDINGS:")
        print("="*80)
        print(f"💰 Token Savings: {token_savings:.1f}%")
        print(f"🎯 Accuracy Gain: {accuracy_gain:+.1f}%")
        
        if retrieval.get('retrieved_turn_10'):
            print(f"✓  Retrieval found exact JWT code from turn 10")
        else:
            print(f"✗  Retrieval did NOT retrieve turn 10")
        
        print(f"\n📊 Why this matters for coding agents:")
        print(f"   - Coding needs EXACT code, not summaries")
        print(f"   - Summarization loses variable names, function signatures")
        print(f"   - Retrieval maintains code fidelity across long sessions")


async def main():
    """Main function"""
    print("="*80)
    print("CODING AGENT CONTEXT MANAGEMENT BENCHMARK")
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
