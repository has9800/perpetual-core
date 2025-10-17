# Quality A/B Benchmark for Perpetual AI

**Purpose:** Validate that semantic memory retrieval maintains high response quality compared to full context.

This benchmark runs **locally on your Vast.ai GPU** with quantized Mistral-7B, requiring **no API keys**.

---

## What This Tests

The quality benchmark compares two approaches across 4 different scenarios:

1. **Full Context (Baseline):** Send entire conversation history (traditional approach)
2. **Retrieval (Perpetual AI):** Send only recent turns + semantically similar memories

### Test Scenarios

| Test | Description | Challenges |
|------|-------------|------------|
| **Multi-Hop Reasoning** | Information spans multiple turns (A→B→C) | Model must connect facts across turns |
| **Context-Dependent Queries** | Query references specific earlier info | Model must retrieve exact referenced content |
| **Code Understanding** | Cross-file dependencies in code discussion | Model must track technical details |
| **Long-Form Generation** | Generate coherent content from scattered info | Model must synthesize multiple facts |

---

## Metrics

For each test, we measure:

- **Text Similarity:** Exact text match (0-1) using `difflib.SequenceMatcher`
- **Semantic Similarity:** Meaning match (0-1) using cosine similarity of embeddings
- **Token Savings:** % reduction in prompt tokens

**Success Criteria:**
- ✅ **Excellent (≥90%):** Production ready, retrieval matches full context quality
- ⚠️ **Good (≥80%):** Acceptable for most use cases
- ⚠️ **Fair (≥70%):** Consider tuning memory_config (increase recent_turns or semantic_top_k)
- ❌ **Poor (<70%):** Retrieval needs improvement

---

## Prerequisites

### 1. Vast.ai GPU Instance

Start an instance with:
- **GPU:** RTX 3090 or better (24GB VRAM minimum)
- **Model:** `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel`
- **Disk:** 50GB+

### 2. Install Dependencies

```bash
# On Vast.ai instance
cd /workspaces/perpetual-core

# Install vLLM
pip install vllm

# Install dependencies
pip install -r requirements.txt

# Install quality benchmark requirements
pip install sentence-transformers numpy
```

### 3. Qdrant Setup

**Option A: Use Existing Qdrant Cluster (Recommended for Vast.ai)**

If you already have a Qdrant Cloud cluster or remote instance:

```bash
# Set environment variables
export QDRANT_CLOUD_URL="https://xxx-xxx-xxx.aws.cloud.qdrant.io"
export QDRANT_API_KEY="your-api-key-here"

# Or pass as arguments
python tests/quality_benchmark.py \
  --qdrant-url "https://xxx.aws.cloud.qdrant.io" \
  --qdrant-api-key "your-key"
```

**Option B: Start Local Qdrant (if no existing cluster)**

Note: Docker-in-Docker doesn't work on Vast.ai. If you need local Qdrant, use direct installation:

```bash
# Download and run Qdrant binary
wget https://github.com/qdrant/qdrant/releases/download/v1.7.4/qdrant-x86_64-unknown-linux-gnu.tar.gz
tar -xzf qdrant-x86_64-unknown-linux-gnu.tar.gz
./qdrant
```

Or use Docker if available:
```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

**Verify Qdrant is accessible:**
```bash
curl http://localhost:6333/health
# Or for remote:
curl https://xxx.cloud.qdrant.io/health
# Should return: {"status":"ok"}
```

---

## Running the Benchmark

### Basic Usage (with environment variables)

```bash
# Set Qdrant connection (if using remote cluster)
export QDRANT_CLOUD_URL="https://xxx.aws.cloud.qdrant.io"
export QDRANT_API_KEY="your-api-key"

# Run benchmark
python tests/quality_benchmark.py
```

### With Command-Line Arguments

```bash
# With remote Qdrant cluster
python tests/quality_benchmark.py \
  --qdrant-url "https://xxx.aws.cloud.qdrant.io" \
  --qdrant-api-key "your-key"

# With custom model
python tests/quality_benchmark.py \
  --model TheBloke/Mistral-7B-Instruct-v0.2-GPTQ \
  --qdrant-url "https://xxx.aws.cloud.qdrant.io" \
  --qdrant-api-key "your-key"

# With local Qdrant (default if no args)
python tests/quality_benchmark.py
```

### Supported Models

```bash
# Mistral 7B (default, recommended)
python tests/quality_benchmark.py --model TheBloke/Mistral-7B-Instruct-v0.2-GPTQ

# Llama 2 7B
python tests/quality_benchmark.py --model TheBloke/Llama-2-7B-Chat-GPTQ

# Llama 3 8B
python tests/quality_benchmark.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

---

## Expected Output

```
============================================================
  PERPETUAL AI - QUALITY A/B BENCHMARK
============================================================

Test 1: Multi-Hop Reasoning
Scenario: Information spans multiple turns (A→B→C)

Running with retrieval (15 recent + 3 semantic)...
Running with full context (baseline)...

Results:
  Retrieval tokens: 342
  Full context tokens: 1,287
  Token savings: 73.4%
  Text similarity: 87%
  Semantic similarity: 92%

Test 2: Context-Dependent Queries
Scenario: Query references specific earlier information

Running with retrieval (15 recent + 3 semantic)...
Running with full context (baseline)...

Results:
  Retrieval tokens: 298
  Full context tokens: 1,145
  Token savings: 74.0%
  Text similarity: 91%
  Semantic similarity: 95%

Test 3: Code Understanding
Scenario: Code discussion with cross-file references

Running with retrieval (15 recent + 3 semantic)...
Running with full context (baseline)...

Results:
  Retrieval tokens: 356
  Full context tokens: 1,423
  Token savings: 75.0%
  Text similarity: 89%
  Semantic similarity: 93%

Test 4: Long-Form Content Generation
Scenario: Generate coherent content based on scattered info

Running with retrieval (15 recent + 3 semantic)...
Running with full context (baseline)...

Results:
  Retrieval tokens: 387
  Full context tokens: 1,501
  Token savings: 74.2%
  Text similarity: 85%
  Semantic similarity: 90%

============================================================
  AGGREGATE RESULTS
============================================================

Average Text Similarity: 88%
Average Semantic Similarity: 92.5%
Average Token Savings: 74.1%
Total Duration: 47.3s

Quality Assessment:
  ✓ EXCELLENT (≥90%) - Production ready!

Results saved to: quality_benchmark_results_1234567890.json
```

---

## Interpreting Results

### Semantic Similarity (Most Important)

This measures whether the **meaning** of the response is preserved:

- **≥90%:** Retrieval produces nearly identical responses → **Ship it!**
- **80-89%:** Slight differences but core meaning preserved → **Acceptable**
- **70-79%:** Noticeable quality degradation → **Tune memory_config**
- **<70%:** Significant quality loss → **Investigate retrieval logic**

### Token Savings

Expected savings with default config (15 recent + 3 semantic):
- **Short conversations (10-20 turns):** 50-70% savings
- **Medium conversations (30-50 turns):** 70-85% savings
- **Long conversations (100+ turns):** 90-98% savings

### Text Similarity

Less important than semantic similarity. Lower text similarity is acceptable if semantic similarity is high (meaning is preserved even if wording differs).

---

## Tuning Memory Configuration

If semantic similarity is below 80%, try tuning:

### Increase Recent Turns

```python
# Default: 15 recent turns
memory_config = {"recent_turns": 30, "semantic_top_k": 3}
```

**Trade-off:** More tokens but better context

### Increase Semantic Top-K

```python
# Default: 3 semantic matches
memory_config = {"recent_turns": 15, "semantic_top_k": 5}
```

**Trade-off:** More tokens but captures more relevant context

### Use Preset Modes

In production API, users can choose:

```json
{
  "messages": [...],
  "memory_config": {
    "mode": "safe"  // aggressive | balanced | safe | full
  }
}
```

| Mode | recent_turns | semantic_top_k | Use Case |
|------|-------------|----------------|----------|
| `aggressive` | 10 | 2 | Maximum savings, simple queries |
| `balanced` | 15 | 3 | Default, good for most cases |
| `safe` | 30 | 5 | Complex reasoning, critical accuracy |
| `full` | All | 0 | Fall back to traditional (no retrieval) |

---

## Troubleshooting

### Error: "No module named 'vllm'"

```bash
pip install vllm
```

### Error: "CUDA out of memory"

Reduce `gpu_memory_utilization`:

```python
# In quality_benchmark.py, line 55
self.llm = VLLMEngine(
    model_name=model_name,
    quantization="gptq",
    gpu_memory_utilization=0.7,  # Reduce from 0.9 to 0.7
    max_model_len=4096
)
```

### Error: "Connection refused" (Qdrant)

**For remote Qdrant:**
- Verify URL is correct: `curl https://xxx.cloud.qdrant.io/health`
- Check API key is valid
- Ensure firewall allows access from Vast.ai IP

**For local Qdrant:**
Make sure Qdrant is running:
```bash
# Check if running
curl http://localhost:6333/health

# If using binary
./qdrant

# If using Docker (and Docker works on your system)
docker ps | grep qdrant
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### Semantic Similarity < 80%

1. **Check if retrieval is finding relevant context:**
   - Inspect `retrieved_context_count` in results JSON
   - If 0 matches, your queries aren't finding relevant memories

2. **Try different memory configs:**
   ```bash
   # Edit quality_benchmark.py to test different configs
   # Line 138, change memory_config parameter
   memory_config={"recent_turns": 30, "semantic_top_k": 5}
   ```

3. **Inspect actual responses:**
   - Look at `quality_benchmark_results_*.json`
   - Compare `retrieval.response` vs `full_context.response`
   - Identify what information is missing

---

## Output Files

### JSON Results

File: `quality_benchmark_results_<timestamp>.json`

```json
{
  "timestamp": "2025-01-15T10:30:00",
  "model": "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
  "tests": [
    {
      "test_name": "multi_hop_reasoning",
      "retrieval": {
        "response": "...",
        "retrieval_latency_ms": 23.4,
        "generation_latency_ms": 892.1,
        "prompt_tokens": 342,
        "response_tokens": 156,
        "total_tokens": 498
      },
      "full_context": {
        "response": "...",
        "retrieval_latency_ms": 0,
        "generation_latency_ms": 1124.3,
        "prompt_tokens": 1287,
        "response_tokens": 162,
        "total_tokens": 1449
      },
      "text_similarity": 0.87,
      "semantic_similarity": 0.92,
      "token_savings_pct": 73.4
    }
  ],
  "aggregate": {
    "avg_text_similarity": 0.88,
    "avg_semantic_similarity": 0.925,
    "avg_token_savings_pct": 74.1,
    "total_duration_seconds": 47.3
  }
}
```

---

## Next Steps

### If Quality is Excellent (≥90%)

1. **Ship it!** Your retrieval approach is production-ready
2. Run the performance benchmark: `python tests/v2_comprehensive_benchmark.py`
3. Deploy to production with confidence

### If Quality is Good (80-89%)

1. Document the trade-off: "Slight quality variation for 75% cost savings"
2. Give users control via `memory_config` API parameter
3. Consider this acceptable for most SaaS use cases

### If Quality is Fair (70-79%)

1. Tune memory_config defaults (increase recent_turns or semantic_top_k)
2. Investigate retrieval logic in `core/vector_db.py`
3. Consider improving HyDE or SPLADE weighting

### If Quality is Poor (<70%)

1. Debug retrieval: Are relevant memories being found?
2. Check embedding model quality (Qwen3 vs alternatives)
3. Consider hybrid approach: always include more recent turns
4. Investigate if test scenarios are too complex for retrieval

---

## Comparison with Production APIs

| Scenario | Full Context (OpenAI) | Perpetual AI (Retrieval) |
|----------|----------------------|--------------------------|
| 100-turn conversation | 10,000+ tokens | 140 tokens |
| Cost per request | $0.015 | $0.0002 |
| Context limit | 128k tokens | Infinite |
| Response quality | 100% (baseline) | 90-95% (this test) |

**The question:** Is 90-95% quality worth 99% cost savings?

For most SaaS companies: **Yes.**

---

## Support

If you're getting poor results (<80% semantic similarity), please share:

1. Full output from benchmark run
2. `quality_benchmark_results_*.json` file
3. Your memory_config settings
4. Model used

We'll help you tune the system for your use case.
