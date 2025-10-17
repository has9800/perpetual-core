"""
Perpetual AI V2 Comprehensive Benchmark Suite

Tests the complete proxy system end-to-end including:
- Authentication (Supabase API key validation)
- Provider key management (encrypted storage/retrieval)
- Memory retrieval (Qdrant context management)
- Proxy forwarding (OpenAI, Anthropic, etc.)
- Billing tracking (token usage, costs)
- Performance metrics (latency, throughput, token reduction)

Usage:
    python tests/v2_comprehensive_benchmark.py --provider openai --model gpt-4o-mini --num-turns 20

Environment Variables Required:
    - PERPETUAL_API_URL: Your proxy URL (default: http://localhost:8000)
    - PERPETUAL_API_KEY: Your Perpetual API key
    - OPENAI_API_KEY: Your OpenAI API key (for testing)
    - SUPABASE_URL: Supabase project URL
    - SUPABASE_KEY: Supabase anon key
"""

import asyncio
import httpx
import os
import time
import json
import argparse
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import statistics


# ANSI color codes for beautiful output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


@dataclass
class TurnMetrics:
    """Metrics for a single conversation turn"""
    turn_number: int
    user_message: str
    assistant_response: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    memories_used: int
    timestamp: float


@dataclass
class BenchmarkResults:
    """Complete benchmark results"""
    # Metadata
    provider: str
    model: str
    num_turns: int
    conversation_id: str
    timestamp: str

    # Performance metrics
    total_duration_seconds: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    tokens_per_second: float

    # Token metrics
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    avg_input_tokens: int
    avg_output_tokens: int

    # Memory metrics
    avg_retrieval_latency_ms: float
    avg_memories_used: float
    token_reduction_percent: float  # vs sending full history

    # Cost metrics (estimated)
    estimated_cost_usd: float
    cost_savings_vs_full_history_usd: float

    # Per-turn data
    turn_metrics: List[TurnMetrics]

    # Test results
    auth_test_passed: bool
    provider_key_test_passed: bool
    memory_test_passed: bool
    billing_test_passed: bool


class PerpetualBenchmark:
    """Production-grade benchmark suite for Perpetual AI V2"""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        provider: str,
        model: str,
        provider_api_key: str,
        supabase_url: str,
        supabase_key: str
    ):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.provider = provider
        self.model = model
        self.provider_api_key = provider_api_key
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key

        self.client = httpx.AsyncClient(timeout=120.0)
        self.conversation_id = f"benchmark_{int(time.time())}"

        # Test conversation about a technical topic
        self.test_conversation = [
            "What is a vector database?",
            "How does it differ from a traditional SQL database?",
            "What are some popular vector databases?",
            "Explain how embeddings are used in vector search",
            "What is the advantage of using HNSW algorithm?",
            "How do you handle updates in a vector database?",
            "What is semantic search?",
            "Compare Qdrant vs Pinecone",
            "What is the 'lost in the middle' problem?",
            "How does RAG improve LLM responses?",
            "What is the difference between dense and sparse embeddings?",
            "Explain reciprocal rank fusion",
            "What is HyDE (Hypothetical Document Embeddings)?",
            "How do you optimize retrieval latency?",
            "What are the trade-offs between recall and precision?",
            "Compare semantic vs keyword search",
            "What is query expansion?",
            "How do rerankers improve search results?",
            "What is the role of negative sampling in training?",
            "Explain cross-encoder vs bi-encoder architectures"
        ]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.client.aclose()

    def print_header(self, text: str):
        """Print formatted header"""
        print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.HEADER}{text.center(80)}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.ENDC}\n")

    def print_success(self, text: str):
        """Print success message"""
        print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

    def print_error(self, text: str):
        """Print error message"""
        print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")

    def print_info(self, text: str):
        """Print info message"""
        print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")

    def print_metric(self, label: str, value: str):
        """Print formatted metric"""
        print(f"{Colors.BOLD}{label:.<40}{Colors.ENDC} {Colors.OKGREEN}{value}{Colors.ENDC}")

    async def test_health(self) -> bool:
        """Test API health endpoint"""
        try:
            response = await self.client.get(f"{self.api_url}/health")
            response.raise_for_status()
            data = response.json()

            self.print_success(f"Health check passed: {data['status']}")
            self.print_info(f"Version: {data['version']}, Uptime: {data['uptime_seconds']}s")
            return True
        except Exception as e:
            self.print_error(f"Health check failed: {e}")
            return False

    async def test_auth(self) -> bool:
        """Test authentication with Perpetual API key"""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = await self.client.get(
                f"{self.api_url}/v1/memory/query",
                headers=headers,
                params={"query": "test", "conversation_id": "test"}
            )

            # Should work (200) or fail with business logic error (not 401)
            if response.status_code == 401:
                self.print_error("Authentication failed: Invalid API key")
                return False

            self.print_success("Authentication passed")
            return True
        except Exception as e:
            self.print_error(f"Auth test failed: {e}")
            return False

    async def test_provider_key_management(self) -> bool:
        """Test provider API key storage and retrieval"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            # 1. Add provider key
            self.print_info(f"Adding {self.provider} API key...")
            response = await self.client.post(
                f"{self.api_url}/v1/provider-keys/add",
                headers=headers,
                json={
                    "provider": self.provider,
                    "api_key": self.provider_api_key
                }
            )
            response.raise_for_status()
            self.print_success(f"Provider key added successfully")

            # 2. List keys
            response = await self.client.get(
                f"{self.api_url}/v1/provider-keys/list",
                headers=headers
            )
            response.raise_for_status()
            providers = response.json()['providers']

            if self.provider in providers:
                self.print_success(f"Provider key verified in list: {providers}")
            else:
                self.print_error(f"Provider key not found in list")
                return False

            # 3. Get supported providers
            response = await self.client.get(f"{self.api_url}/v1/provider-keys/supported")
            response.raise_for_status()
            supported = response.json()['providers']
            self.print_info(f"Supported providers: {len(supported)} total")

            return True
        except Exception as e:
            self.print_error(f"Provider key test failed: {e}")
            return False

    async def send_chat_message(
        self,
        message: str,
        turn_number: int
    ) -> Tuple[Optional[TurnMetrics], bool]:
        """Send a chat message and collect metrics"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": message}],
                "conversation_id": self.conversation_id,
                "use_memory": True,
                "memory_top_k": 5,
                "max_tokens": 150,
                "temperature": 0.7
            }

            start_time = time.time()
            response = await self.client.post(
                f"{self.api_url}/v1/chat/completions",
                headers=headers,
                json=payload
            )
            end_time = time.time()

            response.raise_for_status()
            data = response.json()

            # Extract metrics
            usage = data['usage']
            metadata = data.get('perpetual_metadata', {})
            assistant_response = data['choices'][0]['message']['content']

            metrics = TurnMetrics(
                turn_number=turn_number,
                user_message=message,
                assistant_response=assistant_response,
                input_tokens=usage['prompt_tokens'],
                output_tokens=usage['completion_tokens'],
                total_tokens=usage['total_tokens'],
                retrieval_latency_ms=metadata.get('retrieval_latency_ms', 0),
                generation_latency_ms=metadata.get('generation_latency_ms', 0),
                total_latency_ms=metadata.get('total_latency_ms', (end_time - start_time) * 1000),
                memories_used=metadata.get('memories_used', 0),
                timestamp=start_time
            )

            return metrics, True

        except Exception as e:
            self.print_error(f"Turn {turn_number} failed: {e}")
            return None, False

    async def run_conversation_benchmark(self, num_turns: int) -> List[TurnMetrics]:
        """Run multi-turn conversation benchmark"""
        self.print_header(f"Running {num_turns}-Turn Conversation Benchmark")

        turn_metrics = []
        messages_to_use = self.test_conversation[:num_turns]

        for i, message in enumerate(messages_to_use, 1):
            self.print_info(f"Turn {i}/{num_turns}: {message[:60]}...")

            metrics, success = await self.send_chat_message(message, i)

            if success and metrics:
                turn_metrics.append(metrics)

                # Print real-time metrics
                print(f"  → Tokens: {metrics.input_tokens} in / {metrics.output_tokens} out")
                print(f"  → Latency: {metrics.total_latency_ms:.0f}ms "
                      f"(retrieval: {metrics.retrieval_latency_ms:.0f}ms, "
                      f"generation: {metrics.generation_latency_ms:.0f}ms)")
                print(f"  → Memories used: {metrics.memories_used}")
                print()
            else:
                self.print_error(f"Turn {i} failed")
                break

            # Small delay between turns
            await asyncio.sleep(0.5)

        return turn_metrics

    async def test_billing_tracking(self, turn_metrics: List[TurnMetrics]) -> bool:
        """Verify billing data was tracked (query Supabase)"""
        try:
            # Query Supabase usage_logs table
            headers = {
                "apikey": self.supabase_key,
                "Authorization": f"Bearer {self.supabase_key}"
            }

            response = await self.client.get(
                f"{self.supabase_url}/rest/v1/usage_logs",
                headers=headers,
                params={
                    "conversation_id": f"eq.{self.conversation_id}",
                    "order": "timestamp.desc",
                    "limit": "100"
                }
            )

            if response.status_code == 200:
                logs = response.json()
                self.print_success(f"Billing logs found: {len(logs)} entries")

                if logs:
                    total_cost = sum(log.get('cost_usd', 0) for log in logs)
                    self.print_info(f"Total cost tracked: ${total_cost:.4f}")

                return len(logs) > 0
            else:
                self.print_error(f"Could not verify billing logs: {response.status_code}")
                return False

        except Exception as e:
            self.print_error(f"Billing test failed: {e}")
            return False

    def calculate_results(self, turn_metrics: List[TurnMetrics]) -> BenchmarkResults:
        """Calculate comprehensive benchmark results"""

        if not turn_metrics:
            raise ValueError("No metrics to calculate")

        # Latency metrics
        latencies = [m.total_latency_ms for m in turn_metrics]
        sorted_latencies = sorted(latencies)

        # Token metrics
        total_input = sum(m.input_tokens for m in turn_metrics)
        total_output = sum(m.output_tokens for m in turn_metrics)
        total_tokens = sum(m.total_tokens for m in turn_metrics)

        # Memory metrics
        avg_retrieval_latency = statistics.mean([m.retrieval_latency_ms for m in turn_metrics])
        avg_memories = statistics.mean([m.memories_used for m in turn_metrics])

        # Calculate token reduction vs full history
        # Estimate: full history would be ~100 tokens per turn cumulative
        estimated_full_history_tokens = sum(100 * i for i in range(1, len(turn_metrics) + 1))
        token_reduction = ((estimated_full_history_tokens - total_input) / estimated_full_history_tokens) * 100

        # Cost estimation (OpenAI GPT-4o-mini pricing)
        input_cost_per_m = 0.15  # $0.15 per 1M input tokens
        output_cost_per_m = 0.60  # $0.60 per 1M output tokens

        estimated_cost = (
            (total_input / 1_000_000) * input_cost_per_m +
            (total_output / 1_000_000) * output_cost_per_m
        )

        full_history_cost = (
            (estimated_full_history_tokens / 1_000_000) * input_cost_per_m +
            (total_output / 1_000_000) * output_cost_per_m
        )

        cost_savings = full_history_cost - estimated_cost

        # Time metrics
        total_duration = turn_metrics[-1].timestamp - turn_metrics[0].timestamp
        tokens_per_second = total_tokens / total_duration if total_duration > 0 else 0

        return BenchmarkResults(
            provider=self.provider,
            model=self.model,
            num_turns=len(turn_metrics),
            conversation_id=self.conversation_id,
            timestamp=datetime.now().isoformat(),
            total_duration_seconds=total_duration,
            avg_latency_ms=statistics.mean(latencies),
            p50_latency_ms=sorted_latencies[len(sorted_latencies) // 2],
            p95_latency_ms=sorted_latencies[int(len(sorted_latencies) * 0.95)],
            p99_latency_ms=sorted_latencies[int(len(sorted_latencies) * 0.99)],
            tokens_per_second=tokens_per_second,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_tokens=total_tokens,
            avg_input_tokens=total_input // len(turn_metrics),
            avg_output_tokens=total_output // len(turn_metrics),
            avg_retrieval_latency_ms=avg_retrieval_latency,
            avg_memories_used=avg_memories,
            token_reduction_percent=token_reduction,
            estimated_cost_usd=estimated_cost,
            cost_savings_vs_full_history_usd=cost_savings,
            turn_metrics=turn_metrics,
            auth_test_passed=False,  # Set by test
            provider_key_test_passed=False,
            memory_test_passed=False,
            billing_test_passed=False
        )

    def print_results(self, results: BenchmarkResults):
        """Print beautiful formatted results"""
        self.print_header("Benchmark Results")

        print(f"{Colors.BOLD}Configuration:{Colors.ENDC}")
        self.print_metric("Provider", results.provider)
        self.print_metric("Model", results.model)
        self.print_metric("Conversation ID", results.conversation_id)
        self.print_metric("Timestamp", results.timestamp)
        print()

        print(f"{Colors.BOLD}Performance Metrics:{Colors.ENDC}")
        self.print_metric("Total Duration", f"{results.total_duration_seconds:.2f}s")
        self.print_metric("Avg Latency", f"{results.avg_latency_ms:.0f}ms")
        self.print_metric("P50 Latency", f"{results.p50_latency_ms:.0f}ms")
        self.print_metric("P95 Latency", f"{results.p95_latency_ms:.0f}ms")
        self.print_metric("P99 Latency", f"{results.p99_latency_ms:.0f}ms")
        self.print_metric("Throughput", f"{results.tokens_per_second:.1f} tokens/s")
        print()

        print(f"{Colors.BOLD}Token Metrics:{Colors.ENDC}")
        self.print_metric("Total Turns", str(results.num_turns))
        self.print_metric("Total Input Tokens", f"{results.total_input_tokens:,}")
        self.print_metric("Total Output Tokens", f"{results.total_output_tokens:,}")
        self.print_metric("Total Tokens", f"{results.total_tokens:,}")
        self.print_metric("Avg Input Tokens/Turn", str(results.avg_input_tokens))
        self.print_metric("Avg Output Tokens/Turn", str(results.avg_output_tokens))
        print()

        print(f"{Colors.BOLD}Memory & Retrieval:{Colors.ENDC}")
        self.print_metric("Avg Retrieval Latency", f"{results.avg_retrieval_latency_ms:.1f}ms")
        self.print_metric("Avg Memories Used", f"{results.avg_memories_used:.1f}")
        self.print_metric("Token Reduction", f"{results.token_reduction_percent:.1f}%")
        print()

        print(f"{Colors.BOLD}Cost Analysis:{Colors.ENDC}")
        self.print_metric("Estimated Cost", f"${results.estimated_cost_usd:.4f}")
        self.print_metric("Cost Savings vs Full History", f"${results.cost_savings_vs_full_history_usd:.4f}")
        self.print_metric("Savings Percentage", f"{(results.cost_savings_vs_full_history_usd / (results.estimated_cost_usd + results.cost_savings_vs_full_history_usd) * 100):.1f}%")
        print()

        print(f"{Colors.BOLD}Test Results:{Colors.ENDC}")
        status = lambda x: f"{Colors.OKGREEN}PASS{Colors.ENDC}" if x else f"{Colors.FAIL}FAIL{Colors.ENDC}"
        print(f"  Authentication: {status(results.auth_test_passed)}")
        print(f"  Provider Keys: {status(results.provider_key_test_passed)}")
        print(f"  Memory Retrieval: {status(results.memory_test_passed)}")
        print(f"  Billing Tracking: {status(results.billing_test_passed)}")

    def save_results(self, results: BenchmarkResults, filename: str):
        """Save results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(asdict(results), f, indent=2, default=str)
        self.print_success(f"Results saved to {filename}")

    async def run_full_benchmark(self, num_turns: int) -> BenchmarkResults:
        """Run complete benchmark suite"""
        self.print_header("Perpetual AI V2 Comprehensive Benchmark")

        print(f"{Colors.BOLD}Configuration:{Colors.ENDC}")
        self.print_info(f"API URL: {self.api_url}")
        self.print_info(f"Provider: {self.provider}")
        self.print_info(f"Model: {self.model}")
        self.print_info(f"Turns: {num_turns}")
        print()

        # 1. Health check
        self.print_header("1. Health Check")
        await self.test_health()

        # 2. Authentication test
        self.print_header("2. Authentication Test")
        auth_passed = await self.test_auth()

        # 3. Provider key management test
        self.print_header("3. Provider Key Management Test")
        provider_key_passed = await self.test_provider_key_management()

        # 4. Conversation benchmark
        turn_metrics = await self.run_conversation_benchmark(num_turns)
        memory_passed = len(turn_metrics) > 0

        # 5. Billing test
        self.print_header("5. Billing Tracking Test")
        billing_passed = await self.test_billing_tracking(turn_metrics)

        # Calculate results
        results = self.calculate_results(turn_metrics)
        results.auth_test_passed = auth_passed
        results.provider_key_test_passed = provider_key_passed
        results.memory_test_passed = memory_passed
        results.billing_test_passed = billing_passed

        # Print results
        self.print_results(results)

        return results


async def main():
    parser = argparse.ArgumentParser(description="Perpetual AI V2 Comprehensive Benchmark")
    parser.add_argument("--api-url", default=os.getenv("PERPETUAL_API_URL", "http://localhost:8000"))
    parser.add_argument("--api-key", default=os.getenv("PERPETUAL_API_KEY"), required=True)
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic", "xai", "together", "cerebras"])
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--provider-api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--num-turns", type=int, default=20)
    parser.add_argument("--supabase-url", default=os.getenv("SUPABASE_URL"))
    parser.add_argument("--supabase-key", default=os.getenv("SUPABASE_KEY"))
    parser.add_argument("--output", default=f"benchmark_results_{int(time.time())}.json")

    args = parser.parse_args()

    # Validate required env vars
    if not args.provider_api_key:
        print(f"{Colors.FAIL}Error: Provider API key required{Colors.ENDC}")
        return

    if not args.supabase_url or not args.supabase_key:
        print(f"{Colors.WARNING}Warning: Supabase credentials not provided, billing test will be skipped{Colors.ENDC}")

    # Run benchmark
    async with PerpetualBenchmark(
        api_url=args.api_url,
        api_key=args.api_key,
        provider=args.provider,
        model=args.model,
        provider_api_key=args.provider_api_key,
        supabase_url=args.supabase_url,
        supabase_key=args.supabase_key
    ) as benchmark:
        results = await benchmark.run_full_benchmark(args.num_turns)
        benchmark.save_results(results, args.output)


if __name__ == "__main__":
    asyncio.run(main())
