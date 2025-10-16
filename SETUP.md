```bash
export HF_HOME=/data/huggingface_cache
export TRANSFORMERS_CACHE=/data/huggingface_cache
export HF_DATASETS_CACHE=/data/huggingface_cache

cat >> ~/.bashrc << 'EOF'
export HF_HOME=/data/huggingface_cache
export TRANSFORMERS_CACHE=/data/huggingface_cache
export HF_DATASETS_CACHE=/data/huggingface_cache
EOF

mkdir -p /data/huggingface_cache

pip install --upgrade pip setuptools wheel
pip install vllm
pip install fastapi
pip install uvicorn[standard]
pip install python-dotenv
pip install chromadb
pip install sentence-transformers==2.2.2 --upgrade
pip install python-multipart
pip install psutil
pip install splade-model
pip install qdrant-client
pip install sentence-transformers
pip install langchain langchain-openai openai

python3 perpetual-core/tests/gpu_benchmark.py

# optional to kill processes
pkill -9 python
pkill -9 python3

# Wait for cleanup
sleep 2
```
