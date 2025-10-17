# Vast.ai Quick Start Guide

## Step 1: Rent GPU Instance

**Recommended specs:**
- **GPU:** RTX 3090, RTX 4090, or A6000 (24GB+ VRAM)
- **Disk:** 100GB+ (important!)
- **Template:** PyTorch 2.1+ with CUDA 12.1+
- **Mounted storage:** Ensure you have a large disk mounted at `/workspace` or `/data`

**On Vast.ai:**
1. Search for instances with 24GB+ VRAM
2. Filter: `disk_space >= 100`
3. Select instance with `/workspace` or `/data` mounted
4. Click "Rent" and wait for instance to start

---

## Step 2: SSH to Instance

```bash
# Get SSH command from Vast.ai dashboard
ssh -p <PORT> root@<IP>
```

---

## Step 3: Upload Setup Script

**Option A: Via Git (if repo is public)**
```bash
cd /workspace  # or /data
git clone https://github.com/your-username/perpetual-core.git
cd perpetual-core
```

**Option B: Via SCP (from your local machine)**
```bash
# From your local machine
scp -P <PORT> -r /path/to/perpetual-core root@<IP>:/workspace/
```

**Option C: Via wget (if you host the script)**
```bash
cd /workspace
wget https://raw.githubusercontent.com/your-username/perpetual-core/main/tests/VAST_AI_SETUP.sh
chmod +x VAST_AI_SETUP.sh
```

---

## Step 4: Run Setup Script

```bash
cd /workspace/perpetual-core
./tests/VAST_AI_SETUP.sh
```

**What this does:**
- ✅ Detects mounted drive (`/workspace` or `/data`)
- ✅ Creates directories for packages, cache, models
- ✅ Installs vLLM, sentence-transformers, qdrant-client
- ✅ Configures environment variables
- ✅ Optionally downloads Mistral-7B model
- ✅ Saves Qdrant credentials
- ✅ Verifies CUDA is available

**Duration:** ~10-15 minutes

---

## Step 5: Run Quality Benchmark

```bash
cd /workspace/perpetual-core

# If you set Qdrant credentials during setup
./tests/RUN_QUALITY_BENCHMARK.sh

# Or run directly
python3 tests/quality_benchmark.py \
  --qdrant-url "https://xxx.aws.cloud.qdrant.io" \
  --qdrant-api-key "your-key"
```

**Duration:** ~45-60 seconds

---

## Expected Output

```
Configuration:
  Model: TheBloke/Mistral-7B-Instruct-v0.2-GPTQ
  Qdrant URL: https://xxx.aws.cloud.qdrant.io
  Qdrant API Key: Set

Initializing Quality Benchmark...
Loading model: TheBloke/Mistral-7B-Instruct-v0.2-GPTQ
Connecting to Qdrant at: https://xxx.aws.cloud.qdrant.io
Loading Qwen3-Embedding-0.6B...
✅ Qwen3 loaded
Loading SPLADE...
✅ SPLADE loaded
✓ Initialization complete

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

[... 3 more tests ...]

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

## Troubleshooting

### "No space left on device"
- Your root filesystem is full
- Make sure you selected an instance with **100GB+ disk**
- Ensure disk is mounted at `/workspace` or `/data`
- Re-run setup script - it will use the mounted drive

### "python: command not found"
- Use `python3` instead
- The setup script handles this automatically

### "CUDA out of memory"
- Your GPU doesn't have enough VRAM
- Rent an instance with 24GB+ VRAM (RTX 3090 or better)
- Or reduce `gpu_memory_utilization` in the benchmark

### "Connection refused" (Qdrant)
- Check your Qdrant URL is correct
- Verify API key is valid
- Test: `curl https://xxx.cloud.qdrant.io/health`

### "Module not found" errors
- Run setup script again: `./tests/VAST_AI_SETUP.sh`
- Verify PYTHONPATH: `echo $PYTHONPATH`
- Should include: `/workspace/python_packages` or `/data/python_packages`

---

## What You Need

### Before Starting

1. **Qdrant Cluster**
   - URL: `https://xxx.aws.cloud.qdrant.io`
   - API Key: Get from Qdrant Cloud dashboard
   - Free tier works fine for testing

2. **Vast.ai Account**
   - Add credit (~$5 is enough for several hours)
   - GPU costs ~$0.30-$0.70/hour

### From This Repo

The setup script needs:
- `tests/VAST_AI_SETUP.sh` - Main setup script
- `tests/quality_benchmark.py` - Benchmark code
- `tests/RUN_QUALITY_BENCHMARK.sh` - Convenience runner
- `core/llm_wrapper.py` - vLLM wrapper
- `core/vector_db.py` - Qdrant adapter
- `core/memory_manager.py` - Memory management

**Easiest:** Just clone the entire repo on Vast.ai

---

## After Benchmark Completes

1. **Download results:**
   ```bash
   # From your local machine
   scp -P <PORT> root@<IP>:/workspace/perpetual-core/quality_benchmark_results_*.json ./
   ```

2. **Share results** if you hit ≥90% semantic similarity

3. **Destroy instance** to stop charges
   - Go to Vast.ai dashboard
   - Click "Destroy" on your instance

---

## Quick Commands Reference

```bash
# Check disk space
df -h

# Check GPU
nvidia-smi

# Check CUDA in Python
python3 -c "import torch; print(torch.cuda.is_available())"

# Check installed packages
pip3 list | grep -E "vllm|sentence|qdrant"

# View environment
env | grep -E "PYTHON|PIP|HF|QDRANT"

# Test Qdrant connection
curl $QDRANT_CLOUD_URL/health

# Re-run just the benchmark
cd /workspace/perpetual-core
python3 tests/quality_benchmark.py \
  --qdrant-url "$QDRANT_CLOUD_URL" \
  --qdrant-api-key "$QDRANT_API_KEY"
```

---

## Cost Estimate

- **GPU rental:** ~$0.50/hour × 1 hour = $0.50
- **Setup time:** 15 minutes (one-time)
- **Benchmark time:** 1 minute
- **Total:** Less than $1 for complete testing

**Tip:** Destroy instance after benchmark to avoid idle charges

---

## Files Created

After running benchmark, you'll have:

```
/workspace/perpetual-core/
├── quality_benchmark_results_<timestamp>.json  # Detailed results
└── tests/
    └── VAST_AI_SETUP.sh  # Setup script (can reuse)

/workspace/
├── python_packages/  # All pip packages (~10GB)
├── models/          # HuggingFace models (~15GB)
└── pip_cache/       # Pip cache (~5GB)
```

Download the JSON results file before destroying the instance!
