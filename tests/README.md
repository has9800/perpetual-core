# Perpetual AI - Testing Suite

This directory contains benchmarks and tests for validating Perpetual AI's quality and performance.

---

## Quick Start (Vast.ai)

1. **Follow the checklist:** [CHECKLIST.md](CHECKLIST.md)
2. **Read quickstart:** [VAST_AI_QUICKSTART.md](VAST_AI_QUICKSTART.md)
3. **Run setup:** `./VAST_AI_SETUP.sh`
4. **Run benchmark:** `./RUN_QUALITY_BENCHMARK.sh`

**Total time:** 30 minutes
**Total cost:** <$1

---

## Files in This Directory

### Setup & Guides
- **`CHECKLIST.md`** - Step-by-step checklist for Vast.ai setup
- **`VAST_AI_QUICKSTART.md`** - Comprehensive quickstart guide
- **`VAST_AI_SETUP.sh`** - Automated setup script for fresh GPU instances

### Benchmarks
- **`quality_benchmark.py`** - A/B quality testing (retrieval vs full context)
- **`v2_comprehensive_benchmark.py`** - End-to-end API benchmark
- **`RUN_QUALITY_BENCHMARK.sh`** - Convenience script for quality benchmark

### Documentation
- **`README_QUALITY_BENCHMARK.md`** - Detailed quality benchmark documentation
- **`README_V2_BENCHMARK.md`** - API benchmark documentation
- **`EXPECTED_RESULTS_EXAMPLE.json`** - Example output format

### Unit Tests
- **`unit/test_provider_config.py`** - Provider detection tests
- **`unit/test_api_key_service.py`** - Encryption tests
- **`unit/test_helpers.py`** - Utility function tests

---

## Benchmarks Overview

### 1. Quality Benchmark (Recommended First)

**Purpose:** Prove that retrieval maintains quality vs full context

**What it tests:**
- Multi-hop reasoning
- Context-dependent queries
- Code understanding
- Long-form generation

**Metrics:**
- Semantic similarity (key metric)
- Text similarity
- Token savings

**How to run:**
```bash
python3 quality_benchmark.py \
  --qdrant-url "https://xxx.cloud.qdrant.io" \
  --qdrant-api-key "your-key"
```

**Success:** ≥90% semantic similarity with ~75% token savings

---

### 2. API Benchmark

**Purpose:** End-to-end API testing with real providers

**What it tests:**
- Health endpoints
- Authentication
- Provider key management
- Multi-turn conversations
- Memory retrieval
- Billing tracking

**Requirements:**
- Running API server
- Supabase credentials
- Provider API key (OpenAI, Anthropic, etc.)

**How to run:**
```bash
python3 v2_comprehensive_benchmark.py \
  --provider openai \
  --model gpt-4o-mini \
  --num-turns 20
```

---

### 3. Unit Tests

**Purpose:** Test individual components without external dependencies

**What it tests:**
- Provider auto-detection
- API key encryption/decryption
- Conversation ID resolution
- Token counting
- Text sanitization

**How to run:**
```bash
python3 -m pytest unit/ -v
```

**No API keys or external services required**

---

## Recommended Testing Flow

### Phase 1: Quality Validation (Vast.ai)
1. Set up GPU instance
2. Run quality benchmark
3. Verify ≥90% semantic similarity
4. **Result:** Proof that retrieval works

### Phase 2: Unit Testing (Local)
1. Run unit tests locally
2. Verify all pass
3. **Result:** Core logic validated

### Phase 3: API Testing (Local/Staging)
1. Start API server
2. Run comprehensive benchmark
3. Verify end-to-end flow
4. **Result:** Production readiness confirmed

---

## System Requirements

### Quality Benchmark
- **GPU:** 24GB+ VRAM (RTX 3090, RTX 4090, A6000)
- **Disk:** 100GB+ (for models and packages)
- **RAM:** 16GB+
- **Duration:** 1-2 minutes (after setup)

### API Benchmark
- **No GPU required** (uses external APIs)
- **Requirements:** Running API server, Supabase, provider API keys
- **Duration:** 2-5 minutes

### Unit Tests
- **No GPU required**
- **No external services required**
- **Duration:** 10 seconds

---

## Output Files

After running benchmarks, you'll find:

```
tests/
├── quality_benchmark_results_<timestamp>.json
└── benchmark_results_<timestamp>.json
```

**Important:** Download these before destroying your GPU instance!

---

## Troubleshooting

### "No module named 'vllm'"
```bash
# Install to mounted drive
pip3 install --target=/workspace/python_packages vllm
export PYTHONPATH=/workspace/python_packages:$PYTHONPATH
```

### "No space left on device"
```bash
# Check disk space
df -h

# Ensure you're using mounted drive (/workspace or /data)
# Re-run VAST_AI_SETUP.sh
```

### "CUDA out of memory"
```bash
# Check GPU memory
nvidia-smi

# Reduce memory usage in quality_benchmark.py line 69:
gpu_memory_utilization=0.7  # Down from 0.9
```

### "Connection refused" (Qdrant)
```bash
# Test connection
curl https://xxx.cloud.qdrant.io/health

# Verify credentials
echo $QDRANT_CLOUD_URL
echo $QDRANT_API_KEY
```

See detailed troubleshooting in:
- [README_QUALITY_BENCHMARK.md](README_QUALITY_BENCHMARK.md)
- [VAST_AI_QUICKSTART.md](VAST_AI_QUICKSTART.md)

---

## Cost Estimates

| Test | Duration | GPU Cost | Total |
|------|----------|----------|-------|
| Quality (first run) | 15 min | $0.40/hr | ~$0.10 |
| Quality (cached) | 1 min | $0.40/hr | ~$0.01 |
| API Benchmark | 3 min | N/A | $0.00 (uses APIs) |
| Unit Tests | 10 sec | N/A | $0.00 |

**Total for complete testing:** <$1

---

## Support

If you encounter issues:

1. Check [CHECKLIST.md](CHECKLIST.md) - Step-by-step validation
2. Read [VAST_AI_QUICKSTART.md](VAST_AI_QUICKSTART.md) - Detailed guide
3. Review [README_QUALITY_BENCHMARK.md](README_QUALITY_BENCHMARK.md) - Technical details
4. Check your setup: `df -h`, `nvidia-smi`, `echo $PYTHONPATH`

---

## Next Steps After Testing

Once you achieve ≥90% semantic similarity:

1. ✅ **Document results** - Save JSON files
2. ✅ **Update README** - Add benchmark results
3. ✅ **Marketing materials** - Use proof in pitch decks
4. ✅ **Deploy with confidence** - You have data-backed proof
5. ✅ **Share results** - Twitter, Reddit, HN, blog posts

**Key claim you can now make:**
> "Proven 90%+ quality preservation with 75% token savings across multi-hop reasoning, code discussions, and context-dependent queries."

---

## Files Reference

```
tests/
├── README.md                          # This file
├── CHECKLIST.md                       # Step-by-step checklist
├── VAST_AI_QUICKSTART.md             # Quickstart guide
├── VAST_AI_SETUP.sh                  # Automated setup
├── quality_benchmark.py              # Quality A/B test
├── RUN_QUALITY_BENCHMARK.sh          # Convenience runner
├── v2_comprehensive_benchmark.py     # API benchmark
├── README_QUALITY_BENCHMARK.md       # Quality benchmark docs
├── README_V2_BENCHMARK.md            # API benchmark docs
├── EXPECTED_RESULTS_EXAMPLE.json     # Example output
└── unit/
    ├── test_provider_config.py       # Unit tests
    ├── test_api_key_service.py
    └── test_helpers.py
```
