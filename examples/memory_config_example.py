"""
Example: Using memory_config to control retrieval behavior

Shows how users can tune memory settings based on their use case:
- aggressive: Maximum token savings (10 recent + 2 semantic)
- balanced: Good balance (15 recent + 3 semantic) [DEFAULT]
- safe: Higher quality (30 recent + 5 semantic)
- full: Traditional approach (all context, no retrieval)
"""
import requests
import json

BASE_URL = "http://localhost:8000"
API_KEY = "your-api-key-here"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}


def example_balanced_mode():
    """Balanced mode: Default, good for most use cases"""
    print("\n=== Balanced Mode (Default) ===")
    print("15 recent turns + 3 semantic matches")

    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers=headers,
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": "What's the capital of France?"}
            ],
            "conversation_id": "example_balanced",
            "memory_config": {
                "mode": "balanced"  # Default
            }
        }
    )

    result = response.json()
    print(f"Response: {result['choices'][0]['message']['content']}")
    print(f"Metadata: {json.dumps(result['perpetual_metadata'], indent=2)}")


def example_aggressive_mode():
    """Aggressive mode: Maximum savings, simple queries"""
    print("\n=== Aggressive Mode ===")
    print("10 recent turns + 2 semantic matches")
    print("Use for: Simple queries, high-volume traffic, cost-sensitive")

    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers=headers,
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ],
            "conversation_id": "example_aggressive",
            "memory_config": {
                "mode": "aggressive"
            }
        }
    )

    result = response.json()
    print(f"Response: {result['choices'][0]['message']['content']}")
    print(f"Tokens used: {result['usage']['total_tokens']}")


def example_safe_mode():
    """Safe mode: Higher quality, complex reasoning"""
    print("\n=== Safe Mode ===")
    print("30 recent turns + 5 semantic matches")
    print("Use for: Complex reasoning, critical accuracy, legal/medical")

    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers=headers,
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": "Based on everything we discussed about the contract, what are the key risks?"}
            ],
            "conversation_id": "example_safe",
            "memory_config": {
                "mode": "safe"
            }
        }
    )

    result = response.json()
    print(f"Response: {result['choices'][0]['message']['content']}")
    print(f"Memories used: {result['perpetual_metadata']['memories_used']}")


def example_full_mode():
    """Full mode: Traditional approach, no retrieval"""
    print("\n=== Full Mode (No Retrieval) ===")
    print("Send ALL conversation history (traditional approach)")
    print("Use for: A/B testing, baseline comparison")

    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers=headers,
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": "Summarize everything we've discussed."}
            ],
            "conversation_id": "example_full",
            "memory_config": {
                "mode": "full"  # No retrieval, send everything
            }
        }
    )

    result = response.json()
    print(f"Response: {result['choices'][0]['message']['content']}")
    print(f"Retrieval latency: {result['perpetual_metadata']['retrieval_latency_ms']}ms (should be ~0)")


def example_custom_settings():
    """Custom settings: Fine-grained control"""
    print("\n=== Custom Settings ===")
    print("20 recent turns + 4 semantic matches (custom)")
    print("Use for: Fine-tuning based on your specific use case")

    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers=headers,
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": "What did we decide about the pricing strategy?"}
            ],
            "conversation_id": "example_custom",
            "memory_config": {
                "recent_turns": 20,
                "semantic_top_k": 4,
                "min_similarity_threshold": 0.6
            }
        }
    )

    result = response.json()
    print(f"Response: {result['choices'][0]['message']['content']}")


def compare_modes():
    """Compare token usage across modes"""
    print("\n=== Mode Comparison ===")

    modes = ["aggressive", "balanced", "safe", "full"]
    results = {}

    for mode in modes:
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            headers=headers,
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "user", "content": "Hello!"}
                ],
                "conversation_id": f"compare_{mode}",
                "memory_config": {"mode": mode}
            }
        )

        result = response.json()
        results[mode] = {
            "tokens": result['usage']['total_tokens'],
            "retrieval_ms": result['perpetual_metadata']['retrieval_latency_ms'],
            "memories": result['perpetual_metadata']['memories_used']
        }

    print("\nMode Comparison:")
    print(f"{'Mode':<12} {'Tokens':<10} {'Retrieval (ms)':<15} {'Memories':<10}")
    print("-" * 50)
    for mode, data in results.items():
        print(f"{mode:<12} {data['tokens']:<10} {data['retrieval_ms']:<15.1f} {data['memories']:<10}")


if __name__ == "__main__":
    print("Perpetual AI - Memory Config Examples")
    print("=" * 50)

    # Run examples
    try:
        example_balanced_mode()
        example_aggressive_mode()
        example_safe_mode()
        example_full_mode()
        example_custom_settings()
        compare_modes()

        print("\n✅ All examples completed!")
        print("\nRecommendations:")
        print("  - Use 'balanced' for most cases (default)")
        print("  - Use 'aggressive' for high-volume, simple queries")
        print("  - Use 'safe' for complex reasoning, critical accuracy")
        print("  - Use 'full' for A/B testing or fallback")
        print("  - Use custom settings for fine-tuning")

    except requests.exceptions.RequestException as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure the API server is running:")
        print("  python -m uvicorn api.main:app --reload")
