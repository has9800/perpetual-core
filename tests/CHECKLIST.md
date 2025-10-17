# Vast.ai Setup Checklist

## Before You Start

- [ ] Have Qdrant Cloud cluster URL
- [ ] Have Qdrant API key
- [ ] Have Vast.ai account with credit (~$5)
- [ ] Have this repository downloaded locally

---

## Step-by-Step Checklist

### 1. Rent GPU Instance

- [ ] Go to Vast.ai
- [ ] Search for GPU with:
  - [ ] 24GB+ VRAM (RTX 3090, RTX 4090, A6000)
  - [ ] 100GB+ disk space
  - [ ] PyTorch 2.1+ template
  - [ ] **Important:** Mounted storage at `/workspace` or `/data`
- [ ] Click "Rent" and wait for instance to start
- [ ] Copy SSH command from dashboard

### 2. Connect to Instance

- [ ] SSH to instance: `ssh -p <PORT> root@<IP>`
- [ ] Verify you're connected
- [ ] Check mounted drive exists: `df -h | grep -E "workspace|data"`
- [ ] Check GPU available: `nvidia-smi`

### 3. Upload Code

**Choose one method:**

**Option A: Git Clone** (if repo is public/accessible)
```bash
- [ ] cd /workspace
- [ ] git clone https://github.com/your-username/perpetual-core.git
- [ ] cd perpetual-core
- [ ] ls -la tests/  # Verify files exist
```

**Option B: SCP Upload** (from your local machine)
```bash
- [ ] From local: scp -P <PORT> -r /path/to/perpetual-core root@<IP>:/workspace/
- [ ] Back to SSH: cd /workspace/perpetual-core
- [ ] ls -la tests/  # Verify files exist
```

### 4. Run Setup Script

- [ ] `cd /workspace/perpetual-core`
- [ ] `chmod +x tests/VAST_AI_SETUP.sh` (if needed)
- [ ] `./tests/VAST_AI_SETUP.sh`
- [ ] Enter Qdrant URL when prompted
- [ ] Enter Qdrant API key when prompted
- [ ] Choose whether to pre-download model (recommended: yes)
- [ ] Wait for setup to complete (~15 minutes)
- [ ] Verify success message appears

### 5. Verify Installation

- [ ] `python3 -c "import vllm; print('vLLM OK')"`
- [ ] `python3 -c "import sentence_transformers; print('transformers OK')"`
- [ ] `python3 -c "import qdrant_client; print('qdrant OK')"`
- [ ] `python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"`
- [ ] `echo $PYTHONPATH`  # Should include `/workspace/python_packages`
- [ ] `echo $QDRANT_CLOUD_URL`  # Should show your URL
- [ ] `curl $QDRANT_CLOUD_URL/health`  # Should return {"status":"ok"}

### 6. Run Benchmark

- [ ] `cd /workspace/perpetual-core`
- [ ] Run: `python3 tests/quality_benchmark.py --qdrant-url "$QDRANT_CLOUD_URL" --qdrant-api-key "$QDRANT_API_KEY"`
- [ ] Or: `./tests/RUN_QUALITY_BENCHMARK.sh`
- [ ] Wait for benchmark to complete (~1 minute)
- [ ] Verify you see "AGGREGATE RESULTS" section
- [ ] Note your quality score (should be ≥90%)

### 7. Download Results

**From your LOCAL machine** (open new terminal):
```bash
- [ ] scp -P <PORT> root@<IP>:/workspace/perpetual-core/quality_benchmark_results_*.json ./
- [ ] Open JSON file and verify results
- [ ] Check semantic_similarity score in aggregate section
```

### 8. Cleanup

- [ ] Go to Vast.ai dashboard
- [ ] Click "Destroy" on your instance
- [ ] Confirm you've downloaded the results JSON file
- [ ] Instance destroyed (stops billing)

---

## What to Check If Something Goes Wrong

### Setup Script Fails

- [ ] Check you have enough disk space: `df -h`
- [ ] Verify mounted drive: `ls /workspace` or `ls /data`
- [ ] Check internet connection: `ping google.com`
- [ ] Try running setup again (it's idempotent)

### Benchmark Fails to Start

- [ ] Check PYTHONPATH: `echo $PYTHONPATH`
- [ ] Verify packages installed: `ls /workspace/python_packages`
- [ ] Source bashrc: `source ~/.bashrc`
- [ ] Check Qdrant connection: `curl $QDRANT_CLOUD_URL/health`

### Out of Memory Errors

- [ ] Check GPU has 24GB+: `nvidia-smi`
- [ ] Check available GPU memory: `nvidia-smi --query-gpu=memory.free --format=csv`
- [ ] Reduce `gpu_memory_utilization` in `tests/quality_benchmark.py` (line 69)

### Benchmark Takes Too Long

- [ ] Ctrl+C to cancel
- [ ] Check if model is downloaded: `ls /workspace/models`
- [ ] Pre-download model using setup script option
- [ ] First run always slower (downloading embedding models)

---

## Expected Timeline

| Step | Duration | Notes |
|------|----------|-------|
| Rent GPU | 1-5 min | Wait for instance to start |
| SSH connect | 30 sec | Instant once instance ready |
| Upload code | 1-2 min | Depends on connection speed |
| Run setup | 10-15 min | vLLM installation is largest |
| Verify install | 1 min | Quick checks |
| **First benchmark run** | **5-10 min** | Downloads embedding models |
| **Subsequent runs** | **45-60 sec** | Models cached |
| Download results | 30 sec | Small JSON file |
| Destroy instance | 30 sec | Stop billing |

**Total time:** ~30 minutes for first-time setup
**Total cost:** ~$0.50-$1.00

---

## Success Criteria

✅ You know you succeeded when:

1. Benchmark completes without errors
2. You see "AGGREGATE RESULTS" section
3. Semantic similarity is ≥80% (ideally ≥90%)
4. Token savings is ~70-75%
5. You have the JSON results file downloaded
6. You can share: *"We tested retrieval vs full context and got 92% semantic similarity with 74% token savings"*

---

## After Success

- [ ] Share results (Twitter, Reddit, pitch decks)
- [ ] Add results to README
- [ ] Use in marketing: "Proven 90%+ quality preservation"
- [ ] Deploy with confidence
- [ ] Consider running benchmark with different memory_config settings to show tuning options

---

## Quick Reference

**Your Qdrant URL:**
```
https://ba1c175b-6b78-4356-bbc9-a926696e3696.us-east-1-1.aws.cloud.qdrant.io
```

**SSH Command:** (get from Vast.ai dashboard)
```
ssh -p <PORT> root@<IP>
```

**Key Commands:**
```bash
# Setup
cd /workspace/perpetual-core && ./tests/VAST_AI_SETUP.sh

# Run benchmark
python3 tests/quality_benchmark.py \
  --qdrant-url "$QDRANT_CLOUD_URL" \
  --qdrant-api-key "$QDRANT_API_KEY"

# Download results (from local machine)
scp -P <PORT> root@<IP>:/workspace/perpetual-core/quality_benchmark_results_*.json ./
```

---

## Files You Need

Ensure these exist in your repo:

- [x] `tests/VAST_AI_SETUP.sh` - Setup script
- [x] `tests/quality_benchmark.py` - Benchmark code
- [x] `tests/RUN_QUALITY_BENCHMARK.sh` - Convenience runner
- [x] `core/llm_wrapper.py` - vLLM wrapper
- [x] `core/vector_db.py` - Qdrant adapter
- [x] `core/memory_manager.py` - Memory manager

All included in this repo ✅
