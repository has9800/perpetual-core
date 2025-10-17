# Perpetual AI V2 Comprehensive Benchmark Suite

Production-grade end-to-end testing suite for Perpetual AI V2 proxy system.

## What It Tests

### ✅ Authentication
- Validates Supabase API key authentication
- Tests rate limiting middleware
- Verifies request authorization

### ✅ Provider Key Management
- Tests encrypted API key storage
- Validates key retrieval and decryption
- Verifies provider list endpoint

### ✅ Memory & Retrieval
- Tests Qdrant context retrieval
- Measures retrieval latency
- Validates memory isolation per conversation
- Tracks memory usage across turns

### ✅ Proxy Forwarding
- Tests multi-provider routing (OpenAI, Anthropic, xAI, etc.)
- Validates request/response format conversion
- Measures forwarding latency

### ✅ Billing Tracking
- Verifies token usage tracking
- Validates cost calculation
- Tests Supabase usage log creation

### ✅ Performance Metrics
- Latency (avg, P50, P95, P99)
- Throughput (tokens/second)
- Token reduction vs full history
- Cost savings analysis

## Prerequisites

```bash
# Required environment variables
export PERPETUAL_API_URL="http://localhost:8000"  # Your proxy URL
export PERPETUAL_API_KEY="your-perpetual-api-key"  # Your Perpetual API key
export OPENAI_API_KEY="sk-..."  # For testing OpenAI forwarding
export SUPABASE_URL="https://xxx.supabase.co"
export SUPABASE_KEY="your-supabase-anon-key"

# Optional: For testing other providers
export ANTHROPIC_API_KEY="sk-ant-..."
export XAI_API_KEY="xai-..."
```

## Usage

### Basic Run (20 turns with OpenAI)
```bash
python tests/v2_comprehensive_benchmark.py \
  --api-key $PERPETUAL_API_KEY \
  --provider openai \
  --model gpt-4o-mini \
  --num-turns 20
```

### Test with Anthropic
```bash
python tests/v2_comprehensive_benchmark.py \
  --api-key $PERPETUAL_API_KEY \
  --provider anthropic \
  --model claude-3-5-sonnet-20241022 \
  --provider-api-key $ANTHROPIC_API_KEY \
  --num-turns 15
```

### Full Production Test (50 turns)
```bash
python tests/v2_comprehensive_benchmark.py \
  --api-url https://api.perpetual.ai \
  --api-key $PERPETUAL_API_KEY \
  --provider openai \
  --model gpt-4o \
  --num-turns 50 \
  --output production_benchmark_results.json
```

### On Vast.ai GPU
```bash
# SSH into your Vast.ai instance
ssh root@<vast-ip> -p <port>

# Clone repo and install dependencies
cd /workspace
git clone <your-repo>
cd perpetual-core
pip install -r requirements.txt

# Set environment variables
export PERPETUAL_API_URL="https://your-deployed-api.com"
export PERPETUAL_API_KEY="..."
export OPENAI_API_KEY="..."

# Run benchmark
python tests/v2_comprehensive_benchmark.py --num-turns 30
```

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--api-url` | `http://localhost:8000` | Perpetual API URL |
| `--api-key` | `$PERPETUAL_API_KEY` | Your Perpetual API key |
| `--provider` | `openai` | Provider to test (openai, anthropic, xai, together, cerebras) |
| `--model` | `gpt-4o-mini` | Model name |
| `--provider-api-key` | `$OPENAI_API_KEY` | Provider API key |
| `--num-turns` | `20` | Number of conversation turns |
| `--supabase-url` | `$SUPABASE_URL` | Supabase project URL |
| `--supabase-key` | `$SUPABASE_KEY` | Supabase anon key |
| `--output` | `benchmark_results_<timestamp>.json` | Output file |

## Output

The benchmark produces:

### 1. Real-time Console Output
Beautiful colored terminal output showing:
- Test progress
- Per-turn metrics
- Real-time latency/token counts
- Pass/fail status for each test

### 2. JSON Results File
Comprehensive results including:
```json
{
  "provider": "openai",
  "model": "gpt-4o-mini",
  "num_turns": 20,
  "avg_latency_ms": 350.5,
  "p95_latency_ms": 850.0,
  "tokens_per_second": 45.2,
  "total_input_tokens": 2840,
  "total_output_tokens": 1950,
  "token_reduction_percent": 96.5,
  "estimated_cost_usd": 0.0012,
  "cost_savings_vs_full_history_usd": 0.0298,
  "auth_test_passed": true,
  "provider_key_test_passed": true,
  "memory_test_passed": true,
  "billing_test_passed": true,
  "turn_metrics": [...]
}
```

## Interpreting Results

### ✅ Success Criteria

**Authentication:**
- ✓ API key validates successfully
- ✓ Unauthorized requests blocked

**Provider Keys:**
- ✓ Key encrypted and stored
- ✓ Key retrievable and decrypted
- ✓ Provider list accurate

**Memory:**
- ✓ All turns complete successfully
- ✓ Retrieval latency < 100ms (p95)
- ✓ Memories retrieved and used

**Billing:**
- ✓ Usage logs created in Supabase
- ✓ Token counts accurate
- ✓ Cost calculated correctly

**Performance:**
- ✓ Avg latency < 500ms
- ✓ P95 latency < 1000ms
- ✓ Token reduction > 95%
- ✓ Throughput > 20 tokens/s

### 📊 Key Metrics

**Token Reduction:**
- Good: > 95% reduction
- Excellent: > 98% reduction
- Target: 99.5% (only ~200 tokens vs 10K+)

**Latency:**
- Fast: < 300ms average
- Good: < 500ms average
- Acceptable: < 1000ms average

**Cost Savings:**
- Typical: 90-95% savings
- Excellent: 98%+ savings

## Troubleshooting

### "Authentication failed"
- Check `PERPETUAL_API_KEY` is valid
- Verify API key in Supabase `api_keys` table
- Check middleware is not blocking requests

### "No API key found for provider"
- Ensure provider API key was added via `/v1/provider-keys/add`
- Check encryption key `API_KEY_ENCRYPTION_KEY` is set consistently
- Verify user_id matches between requests

### "Billing logs not found"
- Check Supabase credentials
- Verify `usage_logs` table exists
- Check RLS policies allow read access

### "Memory retrieval failed"
- Verify Qdrant is running and accessible
- Check collection exists
- Ensure embeddings are being generated

## Example Output

```
================================================================================
                  Perpetual AI V2 Comprehensive Benchmark
================================================================================

Configuration:
ℹ API URL: http://localhost:8000
ℹ Provider: openai
ℹ Model: gpt-4o-mini
ℹ Turns: 20

================================================================================
                              1. Health Check
================================================================================

✓ Health check passed: healthy
ℹ Version: 2.0.0, Uptime: 3600s

================================================================================
                         2. Authentication Test
================================================================================

✓ Authentication passed

================================================================================
                    3. Provider Key Management Test
================================================================================

ℹ Adding openai API key...
✓ Provider key added successfully
✓ Provider key verified in list: ['openai']
ℹ Supported providers: 8 total

================================================================================
              Running 20-Turn Conversation Benchmark
================================================================================

ℹ Turn 1/20: What is a vector database?...
  → Tokens: 142 in / 95 out
  → Latency: 450ms (retrieval: 15ms, generation: 420ms)
  → Memories used: 0

ℹ Turn 2/20: How does it differ from a traditional SQL database?...
  → Tokens: 158 in / 110 out
  → Latency: 380ms (retrieval: 22ms, generation: 345ms)
  → Memories used: 1

[... 18 more turns ...]

================================================================================
                         5. Billing Tracking Test
================================================================================

✓ Billing logs found: 20 entries
ℹ Total cost tracked: $0.0012

================================================================================
                            Benchmark Results
================================================================================

Configuration:
Provider................................ openai
Model................................... gpt-4o-mini
Conversation ID......................... benchmark_1705234567
Timestamp............................... 2025-01-14T15:30:00

Performance Metrics:
Total Duration.......................... 8.50s
Avg Latency............................. 395ms
P50 Latency............................. 380ms
P95 Latency............................. 550ms
P99 Latency............................. 600ms
Throughput.............................. 42.5 tokens/s

Token Metrics:
Total Turns............................. 20
Total Input Tokens...................... 2,840
Total Output Tokens..................... 1,950
Total Tokens............................ 4,790
Avg Input Tokens/Turn................... 142
Avg Output Tokens/Turn.................. 97

Memory & Retrieval:
Avg Retrieval Latency................... 18.5ms
Avg Memories Used....................... 2.8
Token Reduction......................... 96.8%

Cost Analysis:
Estimated Cost.......................... $0.0012
Cost Savings vs Full History............ $0.0298
Savings Percentage...................... 96.1%

Test Results:
  Authentication: PASS
  Provider Keys: PASS
  Memory Retrieval: PASS
  Billing Tracking: PASS

✓ Results saved to benchmark_results_1705234567.json
```

## Next Steps

After successful benchmark:

1. **Review Results:** Check all tests passed and metrics are acceptable
2. **Compare Providers:** Run with different providers to compare performance
3. **Load Testing:** Increase `--num-turns` to test long conversations
4. **Production Validation:** Run against production URL before launch
5. **Monitor Costs:** Track actual savings vs estimates

## Support

For issues or questions:
- Check logs in `/var/log/perpetual/`
- Review Prometheus metrics at `/prometheus`
- Open issue on GitHub
