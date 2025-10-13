#!/bin/bash
#
# LAUNCH SCRIPT - Infinite Memory Inference API
# Sets up and launches the production API
#

set -e

echo "================================================================================"
echo "🚀 Infinite Memory Inference API - Launch Script"
echo "================================================================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✅ Python $python_version detected"
echo ""

# Create directories
echo "Creating necessary directories..."
mkdir -p data/chroma_db
mkdir -p data/logs
echo "✅ Directories created"
echo ""

# Check/Install dependencies
echo "Checking dependencies..."

if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "Installing dependencies from requirements.txt..."
    pip3 install -r config/requirements.txt
else
    echo "✅ Dependencies already installed"
fi
echo ""

# Check .env configuration
echo "Checking .env configuration..."

if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Creating default..."
    cat > .env << EOF
MODEL_NAME=TheBloke/Mistral-7B-Instruct-v0.2-GPTQ
MODEL_QUANTIZATION=int8
GPU_MEMORY_UTILIZATION=0.9
MAX_MODEL_LEN=8192
VECTOR_DB_BACKEND=chromadb
CHROMA_PERSIST_DIR=./data/chroma_db
API_HOST=0.0.0.0
API_PORT=8000
API_KEYS=
CACHE_CAPACITY=1000
TTL_DAYS=90
CONTEXT_RETRIEVAL_K=3
LOG_LEVEL=INFO
LOG_FILE=./data/logs/api.log
EOF
    echo "✅ Default .env created. Edit it before running in production!"
else
    echo "✅ .env file found"
fi
echo ""

# Check GPU availability
echo "Checking GPU availability..."

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "✅ GPU detected"
else
    echo "⚠️  nvidia-smi not found. GPU may not be available."
    echo "   The API will fail if vLLM cannot access GPU."
fi
echo ""

# Pre-flight checks
echo "Running pre-flight checks..."

if [ ! -f "core/memory_manager.py" ]; then
    echo "❌ core/memory_manager.py not found!"
    exit 1
fi

if [ ! -f "core/vector_db_adapters.py" ]; then
    echo "❌ core/vector_db_adapters.py not found!"
    exit 1
fi

if [ ! -f "core/vllm_wrapper_production.py" ]; then
    echo "❌ core/vllm_wrapper_production.py not found!"
    exit 1
fi

if [ ! -f "core/api_server_production.py" ]; then
    echo "❌ core/api_server_production.py not found!"
    exit 1
fi

echo "✅ All required files present"
echo ""

# Start the API
echo "================================================================================"
echo "Starting Infinite Memory Inference API..."
echo "================================================================================"
echo ""
echo "Loading vLLM model (this will take 30-120 seconds)..."
echo "Press Ctrl+C to stop the server"
echo ""

# Run the API
python3 core/api_server_production.py

echo ""
echo "API stopped."