#!/bin/bash
# Convenience script to run clustering benchmark
# Compares No Clustering vs K-means vs GPT-4o semantic clustering

set -e

echo "========================================="
echo "  Clustering A/B/C Benchmark"
echo "========================================="
echo ""

# Check OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    echo ""
    echo "Set it with:"
    echo "  export OPENAI_API_KEY='sk-...'"
    echo ""
    exit 1
fi

# Check Qdrant
if [ -z "$QDRANT_URL" ] && [ -z "$QDRANT_CLOUD_URL" ]; then
    echo "WARNING: Using default Qdrant at http://localhost:6333"
    echo ""
fi

echo "Configuration:"
echo "  OpenAI Key: ${OPENAI_API_KEY:0:8}..."
echo "  Qdrant: ${QDRANT_URL:-${QDRANT_CLOUD_URL:-http://localhost:6333}}"
echo ""

# Run benchmark
python tests/clustering_benchmark.py \
    --model "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ" \
    "${@}"

echo ""
echo "========================================="
echo "  Benchmark Complete!"
echo "========================================="
