#!/bin/bash
# Vast.ai GPU Setup Script for Quality Benchmark
# Run this on your fresh Vast.ai instance with mounted /workspace or /data drive

set -e  # Exit on error

echo "=========================================="
echo "  Perpetual AI - Vast.ai Setup"
echo "=========================================="
echo ""

# Detect mounted drive (Vast.ai uses /workspace or /data)
if [ -d "/workspace" ]; then
    MOUNT_DIR="/workspace"
elif [ -d "/data" ]; then
    MOUNT_DIR="/data"
else
    echo "❌ Error: No mounted drive found (/workspace or /data)"
    echo "Please ensure you have a large enough disk mounted on Vast.ai"
    exit 1
fi

echo "✓ Using mounted drive: $MOUNT_DIR"
echo ""

# Create directories
echo "Creating directories..."
mkdir -p $MOUNT_DIR/python_packages
mkdir -p $MOUNT_DIR/pip_cache
mkdir -p $MOUNT_DIR/perpetual-core
mkdir -p $MOUNT_DIR/models  # For vLLM model cache
echo "✓ Directories created"
echo ""

# Set environment variables
echo "Configuring environment variables..."
export PIP_CACHE_DIR=$MOUNT_DIR/pip_cache
export PYTHONPATH=$MOUNT_DIR/python_packages:$PYTHONPATH
export HF_HOME=$MOUNT_DIR/models  # HuggingFace model cache
export TRANSFORMERS_CACHE=$MOUNT_DIR/models
export SENTENCE_TRANSFORMERS_HOME=$MOUNT_DIR/models

# Make permanent
echo "export PIP_CACHE_DIR=$MOUNT_DIR/pip_cache" >> ~/.bashrc
echo "export PYTHONPATH=$MOUNT_DIR/python_packages:\$PYTHONPATH" >> ~/.bashrc
echo "export HF_HOME=$MOUNT_DIR/models" >> ~/.bashrc
echo "export TRANSFORMERS_CACHE=$MOUNT_DIR/models" >> ~/.bashrc
echo "export SENTENCE_TRANSFORMERS_HOME=$MOUNT_DIR/models" >> ~/.bashrc
echo "✓ Environment variables set"
echo ""

# Check Python version
echo "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✓ Python installed: $PYTHON_VERSION"
else
    echo "❌ Error: python3 not found"
    exit 1
fi
echo ""

# Check CUDA
echo "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "✓ CUDA available"
else
    echo "❌ Warning: nvidia-smi not found. GPU may not be available."
fi
echo ""

# Install dependencies
echo "Installing Python packages to $MOUNT_DIR/python_packages..."
echo "This will take 5-10 minutes. Please be patient..."
echo ""

# Install packages one by one with progress
echo "[1/6] Installing PyTorch..."
pip3 install --no-cache-dir --target=$MOUNT_DIR/python_packages \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "[2/6] Installing vLLM (this is the big one, ~5 min)..."
pip3 install --no-cache-dir --target=$MOUNT_DIR/python_packages vllm

echo "[3/6] Installing sentence-transformers..."
pip3 install --no-cache-dir --target=$MOUNT_DIR/python_packages sentence-transformers

echo "[4/6] Installing qdrant-client..."
pip3 install --no-cache-dir --target=$MOUNT_DIR/python_packages qdrant-client

echo "[5/6] Installing numpy and scipy..."
pip3 install --no-cache-dir --target=$MOUNT_DIR/python_packages numpy scipy

echo "[6/6] Installing transformers and other deps..."
pip3 install --no-cache-dir --target=$MOUNT_DIR/python_packages \
    transformers accelerate huggingface-hub pydantic fastapi

echo ""
echo "✓ All packages installed"
echo ""

# Verify installations
echo "Verifying installations..."
python3 -c "import torch; print(f'  ✓ PyTorch {torch.__version__}')"
python3 -c "import vllm; print('  ✓ vLLM installed')"
python3 -c "import sentence_transformers; print('  ✓ sentence-transformers installed')"
python3 -c "import qdrant_client; print('  ✓ qdrant-client installed')"
python3 -c "import torch; print(f'  ✓ CUDA available: {torch.cuda.is_available()}')"
echo ""

# Clone repository if not exists
if [ ! -d "$MOUNT_DIR/perpetual-core/.git" ]; then
    echo "Cloning Perpetual AI repository..."
    cd $MOUNT_DIR
    git clone https://github.com/your-username/perpetual-core.git
    echo "✓ Repository cloned"
else
    echo "✓ Repository already exists"
    cd $MOUNT_DIR/perpetual-core
    git pull
fi
echo ""

# Set up Qdrant credentials
echo "=========================================="
echo "  Qdrant Configuration"
echo "=========================================="
echo ""
echo "Please set your Qdrant credentials:"
echo ""
read -p "Qdrant URL: " QDRANT_URL
read -p "Qdrant API Key: " QDRANT_KEY

if [ -n "$QDRANT_URL" ] && [ -n "$QDRANT_KEY" ]; then
    echo "export QDRANT_CLOUD_URL='$QDRANT_URL'" >> ~/.bashrc
    echo "export QDRANT_API_KEY='$QDRANT_KEY'" >> ~/.bashrc
    export QDRANT_CLOUD_URL="$QDRANT_URL"
    export QDRANT_API_KEY="$QDRANT_KEY"
    echo "✓ Qdrant credentials saved"
else
    echo "⚠️  Skipping Qdrant configuration. You can set later:"
    echo "  export QDRANT_CLOUD_URL='https://xxx.aws.cloud.qdrant.io'"
    echo "  export QDRANT_API_KEY='your-key'"
fi
echo ""

# Download model (optional but recommended)
echo "=========================================="
echo "  Model Download (Optional)"
echo "=========================================="
echo ""
echo "Would you like to pre-download the Mistral-7B model?"
echo "This will take ~15 minutes but speeds up first benchmark run."
echo ""
read -p "Download now? (y/n): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Downloading Mistral-7B-Instruct-v0.2-GPTQ..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='TheBloke/Mistral-7B-Instruct-v0.2-GPTQ',
    cache_dir='$MOUNT_DIR/models'
)
print('✓ Model downloaded')
"
    echo "✓ Model cached in $MOUNT_DIR/models"
else
    echo "⚠️  Model will be downloaded on first run"
fi
echo ""

# Final summary
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Install location: $MOUNT_DIR/python_packages"
echo "  - Model cache: $MOUNT_DIR/models"
echo "  - Pip cache: $MOUNT_DIR/pip_cache"
echo "  - Repository: $MOUNT_DIR/perpetual-core"
echo ""
echo "Qdrant:"
echo "  - URL: ${QDRANT_CLOUD_URL:-Not set}"
echo "  - API Key: ${QDRANT_API_KEY:+Set}"
echo ""
echo "To run the quality benchmark:"
echo ""
echo "  cd $MOUNT_DIR/perpetual-core"
echo "  python3 tests/quality_benchmark.py \\"
echo "    --qdrant-url \"\$QDRANT_CLOUD_URL\" \\"
echo "    --qdrant-api-key \"\$QDRANT_API_KEY\""
echo ""
echo "Or use the convenience script:"
echo ""
echo "  cd $MOUNT_DIR/perpetual-core"
echo "  ./tests/RUN_QUALITY_BENCHMARK.sh"
echo ""
echo "Environment variables saved to ~/.bashrc"
echo "Run 'source ~/.bashrc' to load them now"
echo ""
echo "=========================================="
