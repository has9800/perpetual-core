"""
Prompt-Based Coding Platform Benchmark
Tests memory systems for iterative AI-driven app development

Simulates 50-turn app building session where early architectural
decisions need to be recalled in later prompts
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


class PromptCodingBenchmark:
    """Compare memory approaches for multi-turn app building"""
    
    def __init__(self, engine: InfiniteMemoryEngine, memory: MemoryManager):
        self.engine = engine
        self.memory = memory
        
        # Authentication code from early in the session (turn 10)
        self.auth_code = """// Email verification system with JWT
import { sendEmail } from '@/lib/email';
import { generateToken } from '@/lib/jwt';

export async function verifyEmail(email: string, userId: string) {
  const token = generateToken({ userId, email, type: 'verification' });
  const verificationUrl = `${process.env.APP_URL}/verify?token=${token}`;
  
  await sendEmail({
    to: email,
    subject: 'Verify your email',
    template: 'verification',
    data: { verificationUrl, userId }
  });
  
  return { success: true, token };
}"""
        
        self.auth_elements = ['generateToken', 'verifyEmail', 'sendEmail', 'verification']
    
    async def run(self) -> Dict:
        """Run benchmark comparing memory approaches"""
        print("\n" + "="*80)
        print("PROMPT-BASED CODING PLATFORM BENCHMARK")
        print("Testing Memory Systems for Multi-Turn App Development")
        print("="*80)
        
        print("\nScenario: Building a SaaS app over 50 prompts")
        print("Turn 10: User adds email verification system")
        print("Turn 50: User asks to modify the verification system")
        print("\nComparing 3 approaches:")
        print("  1. No Memory (traditional - loses context)")
        print("  2. Manual Docs (current workaround)")
        print("  3. Automatic Retrieval (your system)")
        
        # Generate app building session
        prompts = self._generate_app_session()
        test_query = "Update the email verification to include the company name in the email"
        
        print(f"\n{len(prompts)} prompts generated")
        print(f"Test query (turn 50): '{test_query}'")
        
        # Test 1: No memory (just recent context)
        print("\n" + "="*80)
        print("APPROACH 1: NO MEMORY (Traditional)")
        print("="*80)
        no_memory_result = await self._test_no_memory(prompts, test_query)
        
        # Test 2: Manual documentation system
        print("\n" + "="*80)
        print("APPROACH 2: MANUAL DOCS (Current Workaround)")
        print("="*80)
        manual_docs_result = await self._test_manual_docs(prompts, test_query)
        
        # Test 3: Automatic retrieval
        print("\n" + "="*80)
        print("APPROACH 3: AUTOMATIC RETRIEVAL (Your System)")
        print("="*80)
        retrieval_result = await self._test_retrieval(prompts, test_query)
        
        # Compare
        self._print_comparison(no_memory_result, manual_docs_result, retrieval_result)
        
        return {
            'no_memory': no_memory_result,
            'manual_docs': manual_docs_result,
            'retrieval': retrieval_result
        }
    
    def _generate_app_session(self) -> List[Dict]:
        """Generate 50-turn app building session"""
        prompts = []
        
        for i in range(50):
            turn_num = i + 1
            
            if turn_num == 10:
                # Critical: Email verification system
                prompts.append({
                    'turn': turn_num,
                    'user': 'Add email verification with JWT tokens for new user signups',
                    'assistant': self.auth_code
                })
            elif turn_num < 10:
                # Early setup prompts
                prompts.append({
                    'turn': turn_num,
                    'user': f'Create basic {["landing page", "signup form", "database schema", "API routes", "user model", "auth middleware", "email config", "env setup", "error handling"][turn_num-1]}',
                    'assistant': f'// {turn_num}. Basic implementation\nexport default function Component() {{\n  return <div>Feature {turn_num}</div>\n}}'
                })
            else:
                # Later feature prompts
                features = [
                    'dashboard', 'billing', 'analytics', 'notifications', 'settings',
                    'profile', 'teams', 'invitations', 'permissions', 'audit log',
                    'webhooks', 'API keys', 'integrations', 'export', 'import',
                    'search', 'filters', 'sorting', 'pagination', 'caching',
                    'rate limiting', 'logging', 'monitoring', 'alerts', 'backups',
                    'migrations', 'seeds', 'tests', 'docs', 'deployment',
                    'CI/CD', 'staging', 'production', 'rollback', 'feature flags',
                    'A/B testing', 'analytics events', 'error tracking', 'performance', 'security'
                ]
                feature = features[(turn_num - 11) % len(features)]
                prompts.append({
                    'turn': turn_num,
                    'user': f'Add {feature} feature',
                    'assistant': f'// {turn_num}. {feature.title()} implementation\nconst {feature.replace(" ", "")} = () => {{\n  return "implemented"\n}}'
                })
        
        return prompts
    
    async def _test_no_memory(self, prompts: List[Dict], query: str) -> Dict:
        """Test with no memory - only recent context"""
        print("Using only last 5 prompts as context (no memory system)...")
        
        # Build context from only recent prompts
        recent_context = "Recent prompts:\n"
        for prompt in prompts[44:49]:
            recent_context += f"Turn {prompt['turn']}: {prompt['user']}\n"
        
        context_tokens = len(recent_context) // 4
        print(f"Context size: ~{context_tokens} tokens (recent prompts only)")
        
        # Generate answer
        prompt_text = f"""You are helping build an app. Here's the recent context:

{recent_context}

User asks: {query}

Provide the code:"""
        
        start = time.time()
        request = GenerationRequest(
            conversation_id=f"no_memory_{time.time()}",
            messages=[{"role": "user", "content": prompt_text}],
            model="test",
            max_tokens=400,
            temperature=0.3
        )
        
        result = await self.engine.generate(request)
        latency = time.time() - start
        
        if not result.get('success'):
            print("❌ Generation failed")
            return {'accuracy': 0, 'tokens': context_tokens, 'latency': latency, 'found': 0}
        
        answer = result.get('response') or result.get('text') or ""
        
        # Check if answer has auth system details
        found = sum(1 for elem in self.auth_elements if elem in answer)
        accuracy = found / len(self.auth_elements)
        
        print(f"✅ Answer generated")
        print(f"Found {found}/{len(self.auth_elements)} auth elements ({accuracy:.1%})")
        print(f"Expected behavior: Low accuracy (no access to turn 10)")
        
        return {
            'accuracy': accuracy,
            'context_tokens': context_tokens,
            'latency': latency,
            'found_elements': found
        }
    
    async def _test_manual_docs(self, prompts: List[Dict], query: str) -> Dict:
        """Test with manual documentation (current workaround)"""
        print("Simulating manual docs/memory.md file...")
        
        # Simulate what a user would manually write in docs
        # (Generic summary, missing specifics)
        manual_memory = """# Project Memory

## Authentication
- Using JWT for auth
- Email verification for signups
- Tokens expire after 24h

## Features
- Dashboard, billing, teams implemented
- Using React + TypeScript
- API routes in /api folder
"""
        
        # Build context
        context = f"""Documentation (docs/memory.md):
{manual_memory}

Recent prompts:
"""
        for prompt in prompts[44:49]:
            context += f"Turn {prompt['turn']}: {prompt['user']}\n"
        
        context_tokens = len(context) // 4
        print(f"Context size: ~{context_tokens} tokens (manual docs + recent)")
        
        # Generate answer
        prompt_text = f"""You are helping build an app. Here's the context:

{context}

User asks: {query}

Provide the code:"""
        
        start = time.time()
        request = GenerationRequest(
            conversation_id=f"manual_{time.time()}",
            messages=[{"role": "user", "content": prompt_text}],
            model="test",
            max_tokens=400,
            temperature=0.3
        )
        
        result = await self.engine.generate(request)
        latency = time.time() - start
        
        if not result.get('success'):
            print("❌ Generation failed")
            return {'accuracy': 0, 'tokens': context_tokens, 'latency': latency, 'found': 0}
        
        answer = result.get('response') or result.get('text') or ""
        
        # Check accuracy
        found = sum(1 for elem in self.auth_elements if elem in answer)
        accuracy = found / len(self.auth_elements)
        
        print(f"✅ Answer generated")
        print(f"Found {found}/{len(self.auth_elements)} auth elements ({accuracy:.1%})")
        print(f"Expected behavior: Medium accuracy (has overview, missing specifics)")
        
        return {
            'accuracy': accuracy,
            'context_tokens': context_tokens,
            'latency': latency,
            'found_elements': found
        }
    
    async def _test_retrieval(self, prompts: List[Dict], query: str) -> Dict:
        """Test with automatic retrieval"""
        conv_id = f"retrieval_{time.time()}"
        
        print("Storing all 49 prompts in vector DB...")
        
        # Store all prompts
        for prompt in prompts[:49]:
            self.memory.add_turn(
                conversation_id=conv_id,
                text=prompt['user'],
                metadata={'response': prompt['assistant'], 'turn': prompt['turn']}
            )
        
        await asyncio.sleep(2)
        
        # Retrieve relevant context
        print("Retrieving relevant context...")
        start = time.time()
        context_result = self.memory.retrieve_context(conv_id, query, top_k=3)
        retrieval_time = time.time() - start
        
        # Build context
        retrieved_turns = []
        context_text = "Relevant past prompts:\n"
        for result in context_result.get('results', []):
            turn_num = result['metadata'].get('turn')
            retrieved_turns.append(turn_num)
            context_text += f"\nTurn {turn_num}:\n"
            context_text += f"User: {result['text']}\n"
            context_text += f"Code: {result['metadata'].get('response', '')}\n"
        
        context_text += "\nRecent prompts:\n"
        for prompt in prompts[46:49]:
            context_text += f"Turn {prompt['turn']}: {prompt['user']}\n"
        
        context_tokens = len(context_text) // 4
        
        print(f"Retrieved turns: {retrieved_turns}")
        print(f"Context size: ~{context_tokens} tokens")
        
        # Generate answer
        prompt_text = f"""You are helping build an app. Here's the context:

{context_text}

User asks: {query}

Provide the code:"""
        
        start = time.time()
        request = GenerationRequest(
            conversation_id=conv_id + "_gen",
            messages=[{"role": "user", "content": prompt_text}],
            model="test",
            max_tokens=400,
            temperature=0.3
        )
        
        result = await self.engine.generate(request)
        generation_time = time.time() - start
        
        if not result.get('success'):
            print("❌ Generation failed")
            return {'accuracy': 0, 'tokens': context_tokens, 'latency': retrieval_time, 'found': 0}
        
        answer = result.get('response') or result.get('text') or ""
        
        # Check accuracy
        found = sum(1 for elem in self.auth_elements if elem in answer)
        accuracy = found / len(self.auth_elements)
        
        print(f"✅ Answer generated")
        print(f"Found {found}/{len(self.auth_elements)} auth elements ({accuracy:.1%})")
        print(f"Expected behavior: High accuracy (has exact code from turn 10)")
        
        return {
            'accuracy': accuracy,
            'context_tokens': context_tokens,
            'retrieval_latency': retrieval_time,
            'generation_latency': generation_time,
            'total_latency': retrieval_time + generation_time,
            'found_elements': found,
            'retrieved_turn_10': 10 in retrieved_turns
        }
    
    def _print_comparison(self, no_memory: Dict, manual: Dict, retrieval: Dict):
        """Print comparison results"""
        print("\n" + "="*80)
        print("RESULTS: Memory System Comparison")
        print("="*80)
        
        print(f"\n{'Metric':<25} | {'No Memory':<12} | {'Manual Docs':<12} | {'Retrieval':<12}")
        print("-"*80)
        print(f"{'Code Accuracy':<25} | {no_memory['accuracy']:<12.1%} | {manual['accuracy']:<12.1%} | {retrieval['accuracy']:<12.1%}")
        print(f"{'Context Tokens':<25} | {no_memory['context_tokens']:<12} | {manual['context_tokens']:<12} | {retrieval['context_tokens']:<12}")
        print(f"{'Elements Found':<25} | {no_memory['found_elements']}/{len(self.auth_elements):<11} | {manual['found_elements']}/{len(self.auth_elements):<11} | {retrieval['found_elements']}/{len(self.auth_elements):<11}")
        
        # Calculate improvements
        if manual['context_tokens'] > 0:
            token_savings_vs_manual = ((manual['context_tokens'] - retrieval['context_tokens']) 
                                      / manual['context_tokens'] * 100)
        else:
            token_savings_vs_manual = 0
        
        accuracy_vs_no_memory = (retrieval['accuracy'] - no_memory['accuracy']) * 100
        accuracy_vs_manual = (retrieval['accuracy'] - manual['accuracy']) * 100
        
        print("\n" + "="*80)
        print("KEY FINDINGS:")
        print("="*80)
        print(f"🎯 {accuracy_vs_no_memory:+.1f}% more accurate than no memory")
        print(f"🎯 {accuracy_vs_manual:+.1f}% more accurate than manual docs")
        print(f"💰 {token_savings_vs_manual:.1f}% fewer tokens vs manual system")
        
        if retrieval.get('retrieved_turn_10'):
            print(f"✓  Successfully retrieved turn 10 (email verification code)")
        
        print(f"\n📌 USE CASE:")
        print(f"   Problem: AI coding platforms lose context after 30-40 prompts")
        print(f"   Current fix: Users manually maintain docs/memory.md files")
        print(f"   Your solution: Automatic retrieval of relevant past prompts")
        print(f"\n📌 IMPACT:")
        print(f"   • No manual documentation needed")
        print(f"   • {accuracy_vs_no_memory:.0f}% better code accuracy")
        print(f"   • Works automatically in background")


async def main():
    """Main function"""
    print("="*80)
    print("PROMPT-BASED CODING PLATFORM BENCHMARK")
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
    
    benchmark = PromptCodingBenchmark(engine, memory)
    results = await benchmark.run()
    
    print("\n" + "="*80)
    print("✅ BENCHMARK COMPLETE")
    print("="*80)
    
    return results


if __name__ == "__main__":
    asyncio.run(main())
