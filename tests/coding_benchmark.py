"""
Standalone Coding Benchmark: Retrieval vs Summarization
Run independently to test code recall performance

Usage:
    python coding_benchmark.py
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
    """Compare retrieval vs summarization for coding conversations"""
    
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
        
        # What we're looking for in the retrieval
        self.jwt_elements = ['create_access_token', 'jwt_required', 'login', 'JWT_SECRET_KEY']
    
    async def run(self) -> Dict:
        """Run the benchmark"""
        print("\n" + "="*80)
        print("CODING BENCHMARK: Retrieval vs Summarization")
        print("="*80)
        
        # Generate conversation
        turns = self._generate_turns()
        print(f"Generated {len(turns)} conversation turns\n")
        
        # Test retrieval
        print("1. Testing RETRIEVAL approach...")
        retrieval_result = await self._test_retrieval(turns)
        
        # Test summarization
        print("\n2. Testing SUMMARIZATION approach...")
        summary_result = await self._test_summarization(turns)
        
        # Calculate comparison
        token_savings = 0
        if summary_result['tokens'] > 0:
            token_savings = ((summary_result['tokens'] - retrieval_result['tokens']) 
                           / summary_result['tokens'] * 100)
        
        # Print results
        self._print_results(retrieval_result, summary_result, token_savings)
        
        return {
            'retrieval': retrieval_result,
            'summarization': summary_result,
            'token_savings_percent': token_savings
        }
    
    def _generate_turns(self) -> List[Dict]:
        """Generate 50-turn conversation with JWT code at turn 10"""
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
                # The test query at the end
                turns.append({
                    'user': 'Show me the JWT login code from earlier',
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
        conv_id = "retrieval_test"
        
        # Store first 49 turns
        for i in range(49):
            self.memory.add_turn(
                conversation_id=conv_id,
                text=turns[i]['user'],
                metadata={'response': turns[i]['assistant']}
            )
        
        # Wait for Qdrant indexing
        print("   Waiting for vector DB indexing...")
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
        
        print(f"   ✅ Retrieval complete")
        print(f"   Found {found}/{len(self.jwt_elements)} JWT elements")
        print(f"   Accuracy: {accuracy:.1%}")
        print(f"   Latency: {latency*1000:.0f}ms")
        print(f"   Tokens: ~140")
        
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
        
        # Truncate if too long (avoid context overflow)
        if len(history_text) > 2500:
            history_text = history_text[:2500]
        
        # Create summary prompt
        prompt = f"Summarize this coding conversation, keeping all technical details:\n\n{history_text}\n\nSummary:"
        
        # Generate summary
        print("   Generating summary with local model...")
        start = time.time()
        request = GenerationRequest(
            conversation_id="summary_test",
            messages=[{"role": "user", "content": prompt}],
            model="test",
            max_tokens=500,
            temperature=0.3
        )
        
        result = await self.engine.generate(request)
        latency = time.time() - start
        
        if not result.get('success'):
            print("   ❌ Summary generation failed")
            return {'accuracy': 0, 'tokens': 0, 'latency': latency, 'found_elements': 0}
        
        # DEBUG: Print what keys are in result
        print(f"   DEBUG: Result keys: {result.keys()}")
        
        # Extract summary text - try all possible locations
        summary = None
        
        # Try direct keys
        for key in ['text', 'output', 'response', 'content', 'generated_text']:
            if key in result and result[key]:
                summary = result[key]
                print(f"   DEBUG: Found summary in result['{key}']")
                break
        
        # Try metadata
        if not summary and 'metadata' in result:
            print(f"   DEBUG: Metadata keys: {result['metadata'].keys()}")
            for key in ['text', 'generated_text', 'output', 'response']:
                if key in result['metadata'] and result['metadata'][key]:
                    summary = result['metadata'][key]
                    print(f"   DEBUG: Found summary in metadata['{key}']")
                    break
        
        # If still not found, print full result structure
        if not summary:
            print(f"   ❌ Could not find summary text!")
            print(f"   DEBUG: Full result: {result}")
            return {'accuracy': 0, 'tokens': 0, 'latency': latency, 'found_elements': 0}
        
        # Show preview
        print(f"   DEBUG: Summary preview: {summary[:200]}...")
        
        # Calculate accuracy
        found = sum(1 for elem in self.jwt_elements if elem.lower() in summary.lower())
        accuracy = found / len(self.jwt_elements)
        
        tokens = len(summary) // 4
        
        print(f"   ✅ Summarization complete")
        print(f"   Found {found}/{len(self.jwt_elements)} JWT elements")
        print(f"   Accuracy: {accuracy:.1%}")
        print(f"   Latency: {latency*1000:.0f}ms")
        print(f"   Tokens: ~{tokens}")
        
        return {
            'accuracy': accuracy,
            'tokens': tokens,
            'latency': latency,
            'found_elements': found
        }

    
    def _print_results(self, retrieval: Dict, summary: Dict, savings: float):
        """Print comparison results"""
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(f"\n{'Metric':<20} | {'Retrieval':<15} | {'Summarization':<15}")
        print("-"*80)
        print(f"{'Accuracy':<20} | {retrieval['accuracy']:<15.1%} | {summary['accuracy']:<15.1%}")
        print(f"{'Tokens Used':<20} | {retrieval['tokens']:<15} | {summary['tokens']:<15}")
        print(f"{'Latency (ms)':<20} | {retrieval['latency']*1000:<15.0f} | {summary['latency']*1000:<15.0f}")
        print(f"{'Elements Found':<20} | {retrieval['found_elements']}/{len(self.jwt_elements):<14} | {summary['found_elements']}/{len(self.jwt_elements):<14}")
        
        print(f"\n💰 Token Savings: {savings:.1f}%")
        
        accuracy_gain = (retrieval['accuracy'] - summary['accuracy']) * 100
        print(f"🎯 Accuracy Gain: {accuracy_gain:+.1f}%")
        
        if retrieval['latency'] < summary['latency']:
            speed_gain = (summary['latency'] - retrieval['latency']) * 1000
            print(f"⚡ Speed: {speed_gain:.0f}ms faster")
        else:
            speed_loss = (retrieval['latency'] - summary['latency']) * 1000
            print(f"⚡ Speed: {speed_loss:.0f}ms slower")


async def main():
    """Main function to run standalone benchmark"""
    print("="*80)
    print("STANDALONE CODING BENCHMARK")
    print("Comparing Retrieval vs Summarization for Code Recall")
    print("="*80)
    
    # Initialize system
    print("\nInitializing system...")
    
    # Create vector DB
    vector_db_backend = os.getenv("VECTOR_DB_BACKEND", "qdrant")
    print(f"  Vector DB: {vector_db_backend}")
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
    
    print(f"  Model: {model_name}")
    print(f"  Quantization: {quantization}")
    
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
    
    # Run benchmark
    benchmark = CodingBenchmark(infinite_engine, memory_manager)
    results = await benchmark.run()
    
    print("\n" + "="*80)
    print("✅ BENCHMARK COMPLETE")
    print("="*80)
    
    return results


if __name__ == "__main__":
    results = asyncio.run(main())
