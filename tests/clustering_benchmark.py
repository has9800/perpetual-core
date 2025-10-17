"""
Clustering A/B Benchmark

Tests 3 approaches:
1. No clustering (current retrieval baseline)
2. K-means clustering (free but dumb)
3. GPT-4o-mini clustering (smart semantic grouping)

Metrics:
- Semantic similarity (how close to full context response)
- Multi-hop reasoning accuracy (finds all dependencies)
- Response differentiation (how different responses are)
- Latency (retrieval + clustering time)
- Cost (clustering + generation)
"""
import asyncio
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.llm_wrapper import VLLMEngine
from core.vector_db import QdrantAdapter
from core.semantic_clustering import SemanticClusterer, KMeansClusterer
from difflib import SequenceMatcher


class Colors:
    """ANSI color codes"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


class ClusteringBenchmark:
    """A/B/C test: No clustering vs K-means vs GPT-4o clustering"""

    def __init__(
        self,
        model_name: str = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
        qdrant_url: str = None,
        qdrant_api_key: str = None,
        openai_api_key: str = None,
        openai_model: str = "gpt-4o-mini",
        skip_gpt4o: bool = False
    ):
        """Initialize benchmark with all clustering methods"""
        print(f"{Colors.CYAN}Initializing Clustering Benchmark...{Colors.END}")

        # Initialize vLLM
        print(f"Loading model: {model_name}")
        self.llm = VLLMEngine(
            model_name=model_name,
            quantization="gptq",
            gpu_memory_utilization=0.9,
            max_model_len=4096
        )

        # Initialize Qdrant
        import os
        qdrant_url = qdrant_url or os.getenv("QDRANT_URL") or os.getenv("QDRANT_CLOUD_URL")
        qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")

        if qdrant_url:
            print(f"Connecting to Qdrant at: {qdrant_url}")
            self.vector_db = QdrantAdapter(
                url=qdrant_url,
                api_key=qdrant_api_key,
                collection_name="clustering_benchmark"
            )
        else:
            # Use in-memory Qdrant (no server needed)
            print(f"Using in-memory Qdrant (no server required)")
            self.vector_db = QdrantAdapter(
                persist_dir=":memory:",
                collection_name="clustering_benchmark"
            )

        # Initialize clusterers
        self.kmeans_clusterer = KMeansClusterer()

        # Semantic clustering (OpenAI or local vLLM)
        if not skip_gpt4o:
            # Prefer OpenAI if API key provided
            if openai_api_key or os.getenv("OPENAI_API_KEY"):
                try:
                    self.semantic_clusterer = SemanticClusterer(api_key=openai_api_key, model=openai_model)
                    self.has_semantic = True
                    self.semantic_method = openai_model
                    print(f"{Colors.GREEN}✓ Using {openai_model} for semantic clustering{Colors.END}")
                except ValueError as e:
                    print(f"{Colors.YELLOW}⚠️  OpenAI clustering failed: {e}{Colors.END}")
                    print(f"{Colors.YELLOW}   Falling back to local vLLM...{Colors.END}")
                    try:
                        self.semantic_clusterer = SemanticClusterer(vllm_engine=self.llm)
                        self.has_semantic = True
                        self.semantic_method = "Local Mistral"
                        print(f"{Colors.GREEN}✓ Using local vLLM for semantic clustering (FREE!){Colors.END}")
                    except Exception as e2:
                        print(f"{Colors.YELLOW}⚠️  Semantic clustering disabled: {e2}{Colors.END}")
                        self.semantic_clusterer = None
                        self.has_semantic = False
                        self.semantic_method = None
            else:
                # No API key, try local vLLM
                try:
                    self.semantic_clusterer = SemanticClusterer(vllm_engine=self.llm)
                    self.has_semantic = True
                    self.semantic_method = "Local Mistral"
                    print(f"{Colors.GREEN}✓ Using local vLLM for semantic clustering (FREE!){Colors.END}")
                except Exception as e:
                    print(f"{Colors.YELLOW}⚠️  Semantic clustering disabled: {e}{Colors.END}")
                    print(f"{Colors.YELLOW}   Will run FREE comparison: No Clustering vs K-means{Colors.END}")
                    self.semantic_clusterer = None
                    self.has_semantic = False
                    self.semantic_method = None
        else:
            print(f"{Colors.YELLOW}⚠️  Semantic clustering skipped (--skip-gpt4o){Colors.END}")
            self.semantic_clusterer = None
            self.has_semantic = False
            self.semantic_method = None

        print(f"{Colors.GREEN}✓ Initialization complete{Colors.END}\n")

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using embeddings"""
        emb1 = self.vector_db.model.encode([text1])[0]
        emb2 = self.vector_db.model.encode([text2])[0]

        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        return float(dot_product / (norm1 * norm2))

    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate exact text similarity"""
        return SequenceMatcher(None, text1, text2).ratio()

    def calculate_multi_hop_accuracy(
        self,
        response: str,
        required_context: List[str]
    ) -> float:
        """
        Calculate how many required context pieces appear in response

        Args:
            response: Generated response
            required_context: List of required terms (e.g., ["Tokyo", "Kyoto", "student budget"])

        Returns:
            Accuracy score (0-1)
        """
        response_lower = response.lower()
        found = sum(1 for term in required_context if term.lower() in response_lower)
        return found / len(required_context) if required_context else 1.0

    async def run_with_no_clustering(
        self,
        conversation: List[Dict[str, str]],
        test_query: Dict[str, str],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Baseline: Current retrieval without clustering

        Args:
            conversation: Conversation history
            test_query: Test query with question and required context
            top_k: Number of semantic matches to retrieve

        Returns:
            Result dict with metrics
        """
        conversation_id = f"no_cluster_{int(time.time())}"

        # Store conversation in vector DB
        for i, turn in enumerate(conversation):
            text = f"{turn['role'].capitalize()}: {turn['content']}"
            self.vector_db.add(
                conversation_id=conversation_id,
                text=text,
                metadata={'role': turn['role'], 'turn_number': i + 1, 'timestamp': time.time()}
            )

        # Retrieve
        query = test_query['question']
        retrieval_start = time.time()

        results = await self.vector_db.query(conversation_id, query, top_k=top_k)

        retrieval_latency = (time.time() - retrieval_start) * 1000

        # Build context
        context_text = "\n\n".join([
            f"[Context {i+1}]: {r['text']}"
            for i, r in enumerate(results)
        ])

        # Add recent turns
        recent_text = "\n".join([
            f"{turn['role'].capitalize()}: {turn['content']}"
            for turn in conversation[-15:]
        ])

        full_prompt = f"""Relevant context:
{context_text}

Recent conversation:
{recent_text}

User: {query}
Assistant:"""

        # Generate
        gen_start = time.time()
        outputs = self.llm.generate(
            prompts=[full_prompt],
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            stop=["User:", "\n\n\n"]
        )
        response = outputs[0].outputs[0].text.strip()
        gen_latency = (time.time() - gen_start) * 1000

        return {
            'method': 'no_clustering',
            'response': response,
            'retrieval_latency_ms': retrieval_latency,
            'generation_latency_ms': gen_latency,
            'total_latency_ms': retrieval_latency + gen_latency,
            'clustering_cost': 0.0,
            'prompt_tokens': len(full_prompt.split()),
            'retrieved_turns': [r['metadata'].get('turn_number') for r in results]
        }

    async def run_with_kmeans(
        self,
        conversation: List[Dict[str, str]],
        test_query: Dict[str, str],
        num_clusters: int = 5
    ) -> Dict[str, Any]:
        """
        K-means clustering approach

        Args:
            conversation: Conversation history
            test_query: Test query
            num_clusters: Number of clusters

        Returns:
            Result dict with metrics
        """
        conversation_id = f"kmeans_{int(time.time())}"

        # Get embeddings for all turns
        turns_text = [f"{t['role'].capitalize()}: {t['content']}" for t in conversation]
        embeddings = self.vector_db.model.encode(turns_text)

        # Cluster
        cluster_start = time.time()
        clusters, cost = self.kmeans_clusterer.cluster_conversation(
            conversation,
            embeddings,
            num_clusters=num_clusters
        )
        clustering_latency = (time.time() - cluster_start) * 1000

        # Find relevant clusters
        query = test_query['question']
        query_embedding = self.vector_db.model.encode([query])[0]

        relevant_clusters = self.kmeans_clusterer.find_relevant_clusters(
            clusters,
            query_embedding,
            top_k=2
        )

        # Retrieve turns from relevant clusters
        retrieval_start = time.time()
        retrieved_turns = []
        for cluster in relevant_clusters:
            retrieved_turns.extend(cluster.turns)
        retrieval_latency = (time.time() - retrieval_start) * 1000

        # Build context
        context_text = "\n\n".join([
            f"{t['role'].capitalize()}: {t['content']}"
            for t in retrieved_turns[:10]  # Limit to 10 turns
        ])

        recent_text = "\n".join([
            f"{t['role'].capitalize()}: {t['content']}"
            for t in conversation[-15:]
        ])

        full_prompt = f"""Relevant context from conversation:
{context_text}

Recent conversation:
{recent_text}

User: {query}
Assistant:"""

        # Generate
        gen_start = time.time()
        outputs = self.llm.generate(
            prompts=[full_prompt],
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            stop=["User:", "\n\n\n"]
        )
        response = outputs[0].outputs[0].text.strip()
        gen_latency = (time.time() - gen_start) * 1000

        return {
            'method': 'kmeans',
            'response': response,
            'clustering_latency_ms': clustering_latency,
            'retrieval_latency_ms': retrieval_latency,
            'generation_latency_ms': gen_latency,
            'total_latency_ms': clustering_latency + retrieval_latency + gen_latency,
            'clustering_cost': cost,
            'prompt_tokens': len(full_prompt.split()),
            'num_clusters': len(clusters),
            'retrieved_clusters': [c.cluster_id for c in relevant_clusters]
        }

    async def run_with_gpt4o(
        self,
        conversation: List[Dict[str, str]],
        test_query: Dict[str, str],
        num_clusters: int = 5
    ) -> Dict[str, Any]:
        """
        GPT-4o-mini semantic clustering approach

        Args:
            conversation: Conversation history
            test_query: Test query
            num_clusters: Number of clusters

        Returns:
            Result dict with metrics
        """
        conversation_id = f"gpt4o_{int(time.time())}"

        # Cluster with GPT-4o-mini
        cluster_start = time.time()
        clusters, clustering_cost = await self.semantic_clusterer.cluster_conversation(
            conversation,
            conversation_id,
            num_clusters=num_clusters
        )
        clustering_latency = (time.time() - cluster_start) * 1000

        # Find relevant clusters
        query = test_query['question']
        retrieval_start = time.time()

        relevant_clusters = await self.semantic_clusterer.find_relevant_clusters(
            clusters,
            query,
            top_k=2
        )
        retrieval_latency = (time.time() - retrieval_start) * 1000

        # Build context from relevant clusters
        retrieved_turns = []
        for cluster in relevant_clusters:
            retrieved_turns.extend(cluster.turns)

        context_text = "\n\n".join([
            f"[{cluster.label}] {t['role'].capitalize()}: {t['content']}"
            for cluster in relevant_clusters
            for t in cluster.turns[:5]  # Limit turns per cluster
        ])

        recent_text = "\n".join([
            f"{t['role'].capitalize()}: {t['content']}"
            for t in conversation[-15:]
        ])

        full_prompt = f"""Relevant context from conversation:
{context_text}

Recent conversation:
{recent_text}

User: {query}
Assistant:"""

        # Generate
        gen_start = time.time()
        outputs = self.llm.generate(
            prompts=[full_prompt],
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            stop=["User:", "\n\n\n"]
        )
        response = outputs[0].outputs[0].text.strip()
        gen_latency = (time.time() - gen_start) * 1000

        return {
            'method': 'gpt4o_clustering',
            'response': response,
            'clustering_latency_ms': clustering_latency,
            'retrieval_latency_ms': retrieval_latency,
            'generation_latency_ms': gen_latency,
            'total_latency_ms': clustering_latency + retrieval_latency + gen_latency,
            'clustering_cost': clustering_cost,
            'prompt_tokens': len(full_prompt.split()),
            'num_clusters': len(clusters),
            'cluster_labels': [c.label for c in clusters],
            'retrieved_clusters': [(c.cluster_id, c.label) for c in relevant_clusters]
        }

    async def run_multi_hop_test(self) -> Dict[str, Any]:
        """
        Test multi-hop reasoning: Japan trip planning

        This tests if clustering preserves logical dependencies across turns
        """
        print(f"\n{Colors.BOLD}{Colors.BLUE}Multi-Hop Reasoning Test{Colors.END}")
        print("Scenario: Japan trip with budget, cities, duration dependencies\n")

        conversation = [
            {"role": "user", "content": "I'm planning a trip to Japan in March."},
            {"role": "assistant", "content": "Great! March is cherry blossom season. Which cities?"},
            {"role": "user", "content": "I'm thinking Tokyo and Kyoto."},
            {"role": "assistant", "content": "Excellent choices! Tokyo is modern, Kyoto has temples."},
            {"role": "user", "content": "I love traditional architecture. How many days in Kyoto?"},
            {"role": "assistant", "content": "For architecture, 3-4 days to see Kinkaku-ji and Fushimi Inari."},
            {"role": "user", "content": "What about budget? I'm a student."},
            {"role": "assistant", "content": "Budget $80-100/day. Consider hostels."},
            {"role": "user", "content": "Are there student discounts?"},
            {"role": "assistant", "content": "Yes! Many temples offer discounts. Bring student ID."},
            {"role": "user", "content": "What should I pack for March?"},
            {"role": "assistant", "content": "March is mild (10-15°C). Pack layers and walking shoes."},
        ]

        test_query = {
            "question": "Based on everything we discussed, create a 5-day itinerary that fits my budget and interests.",
            "context_needed": ["Tokyo", "Kyoto", "traditional architecture", "student budget", "3-4 days", "March", "temples"]
        }

        # Run methods based on availability
        total_steps = 4 if self.has_semantic else 3
        step = 1

        print(f"{Colors.CYAN}[{step}/{total_steps}] Running without clustering...{Colors.END}")
        result_no_cluster = await self.run_with_no_clustering(conversation, test_query)
        step += 1

        print(f"{Colors.CYAN}[{step}/{total_steps}] Running with k-means...{Colors.END}")
        result_kmeans = await self.run_with_kmeans(conversation, test_query)
        step += 1

        if self.has_semantic:
            print(f"{Colors.CYAN}[{step}/{total_steps}] Running with {self.semantic_method} clustering...{Colors.END}")
            result_semantic = await self.run_with_gpt4o(conversation, test_query)
            step += 1
        else:
            result_semantic = None

        # Full context baseline
        print(f"{Colors.CYAN}[{step}/{total_steps}] Running with full context (baseline)...{Colors.END}")
        full_prompt = "\n".join([
            f"{t['role'].capitalize()}: {t['content']}"
            for t in conversation
        ])
        full_prompt += f"\nUser: {test_query['question']}\nAssistant:"

        outputs = self.llm.generate([full_prompt], max_tokens=512, temperature=0.7, top_p=0.9)
        baseline_response = outputs[0].outputs[0].text.strip()

        # Compare results
        print(f"\n{Colors.GREEN}Results:{Colors.END}")

        results = [
            ("No Clustering", result_no_cluster),
            ("K-means", result_kmeans),
        ]

        if result_semantic:
            results.append((f"{self.semantic_method} Clustering", result_semantic))

        comparison = {}
        for name, result in results:
            sem_sim = self.calculate_semantic_similarity(result['response'], baseline_response)
            text_sim = self.calculate_text_similarity(result['response'], baseline_response)
            multi_hop_acc = self.calculate_multi_hop_accuracy(result['response'], test_query['context_needed'])

            comparison[name] = {
                'semantic_similarity': sem_sim,
                'text_similarity': text_sim,
                'multi_hop_accuracy': multi_hop_acc,
                'latency_ms': result['total_latency_ms'],
                'cost': result.get('clustering_cost', 0),
                'response': result['response']
            }

            print(f"\n{Colors.BOLD}{name}:{Colors.END}")
            print(f"  Semantic similarity: {sem_sim:.2%}")
            print(f"  Multi-hop accuracy: {multi_hop_acc:.2%} ({int(multi_hop_acc * len(test_query['context_needed']))}/{len(test_query['context_needed'])} terms found)")
            print(f"  Total latency: {result['total_latency_ms']:.0f}ms")
            print(f"  Cost: ${result.get('clustering_cost', 0):.4f}")

        return {
            'test_name': 'multi_hop_reasoning',
            'conversation_length': len(conversation),
            'baseline_response': baseline_response,
            'comparison': comparison
        }

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all clustering benchmark tests"""
        print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.HEADER}  CLUSTERING A/B/C BENCHMARK{Colors.END}")
        print(f"{Colors.BOLD}{Colors.HEADER}  Testing: No Clustering | K-means | GPT-4o Semantic{Colors.END}")
        print(f"{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.END}")

        start_time = time.time()

        results = {
            'timestamp': datetime.now().isoformat(),
            'tests': []
        }

        # Run multi-hop test
        test_result = await self.run_multi_hop_test()
        results['tests'].append(test_result)

        # Summary
        print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.HEADER}  SUMMARY{Colors.END}")
        print(f"{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.END}\n")

        comparison = test_result['comparison']

        print(f"{Colors.BOLD}Semantic Similarity (vs Full Context):{Colors.END}")
        for name, data in comparison.items():
            print(f"  {name:20s}: {data['semantic_similarity']:6.2%}")

        print(f"\n{Colors.BOLD}Multi-Hop Accuracy:{Colors.END}")
        for name, data in comparison.items():
            print(f"  {name:20s}: {data['multi_hop_accuracy']:6.2%}")

        print(f"\n{Colors.BOLD}Latency:{Colors.END}")
        for name, data in comparison.items():
            print(f"  {name:20s}: {data['latency_ms']:6.0f}ms")

        print(f"\n{Colors.BOLD}Cost:{Colors.END}")
        for name, data in comparison.items():
            cost_str = f"${data['cost']:.4f}" if data['cost'] > 0 else "FREE"
            print(f"  {name:20s}: {cost_str}")

        # Winner determination
        print(f"\n{Colors.BOLD}Recommendation:{Colors.END}")

        semantic_key = f"{self.semantic_method} Clustering" if self.has_semantic else None

        if semantic_key and semantic_key in comparison:
            semantic_data = comparison[semantic_key]
            kmeans_data = comparison['K-means']

            if semantic_data['semantic_similarity'] > kmeans_data['semantic_similarity'] + 0.03:
                cost_msg = f"Worth the ${semantic_data['cost']:.4f} cost" if semantic_data['cost'] > 0 else "(and it's FREE!)"
                print(f"  {Colors.GREEN}✓ {self.semantic_method} clustering is SIGNIFICANTLY BETTER{Colors.END}")
                print(f"    +{(semantic_data['semantic_similarity'] - kmeans_data['semantic_similarity'])*100:.1f}% accuracy improvement")
                print(f"    {cost_msg}")
            else:
                print(f"  {Colors.YELLOW}○ K-means is sufficient for this scenario{Colors.END}")
        else:
            # Compare no-clustering vs k-means only
            no_cluster_data = comparison['No Clustering']
            kmeans_data = comparison['K-means']

            if kmeans_data['semantic_similarity'] > no_cluster_data['semantic_similarity']:
                print(f"  {Colors.GREEN}✓ K-means clustering improves accuracy (FREE!){Colors.END}")
                print(f"    +{(kmeans_data['semantic_similarity'] - no_cluster_data['semantic_similarity'])*100:.1f}% improvement")
            elif kmeans_data['latency_ms'] < no_cluster_data['latency_ms'] * 0.8:
                print(f"  {Colors.GREEN}✓ K-means is faster with similar accuracy{Colors.END}")
                print(f"    {((no_cluster_data['latency_ms'] - kmeans_data['latency_ms']) / no_cluster_data['latency_ms'] * 100):.0f}% faster")
            else:
                print(f"  {Colors.YELLOW}○ Current approach (no clustering) is sufficient{Colors.END}")
                print(f"  {Colors.CYAN}💡 Semantic clustering might still help - run without --skip-gpt4o{Colors.END}")

        # Save results
        output_file = f"clustering_benchmark_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{Colors.CYAN}Results saved to: {output_file}{Colors.END}")
        print(f"{Colors.CYAN}Total duration: {time.time() - start_time:.1f}s{Colors.END}\n")

        return results


async def main():
    """Main entry point"""
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Clustering A/B/C Benchmark')
    parser.add_argument('--model', type=str, default='TheBloke/Mistral-7B-Instruct-v0.2-GPTQ')
    parser.add_argument('--qdrant-url', type=str, default=None)
    parser.add_argument('--qdrant-api-key', type=str, default=None)
    parser.add_argument('--openai-api-key', type=str, default=None)
    parser.add_argument('--openai-model', type=str, default='gpt-4o-mini',
                      help='OpenAI model for clustering (gpt-5-nano, gpt-4o-mini, etc.)')
    parser.add_argument('--skip-gpt4o', action='store_true',
                      help='Skip semantic clustering (FREE test: no-clustering vs k-means only)')

    args = parser.parse_args()

    benchmark = ClusteringBenchmark(
        model_name=args.model,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        openai_api_key=args.openai_api_key,
        openai_model=args.openai_model,
        skip_gpt4o=args.skip_gpt4o
    )

    results = await benchmark.run_all_tests()

    # Exit code based on semantic clustering advantage (if available)
    comparison = results['tests'][0]['comparison']

    # Find the semantic clustering result (could be GPT-4o or Local Mistral)
    semantic_key = None
    for key in comparison.keys():
        if 'Clustering' in key and key != 'No Clustering' and key != 'K-means':
            semantic_key = key
            break

    if semantic_key:
        semantic_acc = comparison[semantic_key]['multi_hop_accuracy']
        sys.exit(0 if semantic_acc >= 0.85 else 1)
    else:
        # No semantic clustering, check if k-means is good
        kmeans_acc = comparison['K-means']['multi_hop_accuracy']
        sys.exit(0 if kmeans_acc >= 0.75 else 1)


if __name__ == "__main__":
    asyncio.run(main())
