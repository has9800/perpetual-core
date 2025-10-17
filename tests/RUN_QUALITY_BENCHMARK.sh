#!/bin/bash
# Quick script to run quality benchmark on Vast.ai with your Qdrant cluster

echo "=========================================="
echo "  Perpetual AI - Quality Benchmark"
echo "=========================================="
echo ""

# Check if QDRANT_CLOUD_URL is set
if [ -z "$QDRANT_CLOUD_URL" ]; then
    echo "⚠️  Warning: QDRANT_CLOUD_URL not set"
    echo ""
    echo "Please set your Qdrant credentials:"
    echo "  export QDRANT_CLOUD_URL='https://xxx.aws.cloud.qdrant.io'"
    echo "  export QDRANT_API_KEY='your-key-here'"
    echo ""
    echo "Or pass as arguments:"
    echo "  python tests/quality_benchmark.py --qdrant-url 'https://xxx.cloud.qdrant.io' --qdrant-api-key 'your-key'"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Show configuration
echo "Configuration:"
echo "  Model: ${MODEL:-TheBloke/Mistral-7B-Instruct-v0.2-GPTQ}"
echo "  Qdrant URL: ${QDRANT_CLOUD_URL:-http://localhost:6333}"
echo "  Qdrant API Key: ${QDRANT_API_KEY:+Set}"
echo ""

# Run benchmark
echo "Starting quality benchmark..."
echo ""

python tests/quality_benchmark.py \
    ${MODEL:+--model "$MODEL"} \
    ${QDRANT_CLOUD_URL:+--qdrant-url "$QDRANT_CLOUD_URL"} \
    ${QDRANT_API_KEY:+--qdrant-api-key "$QDRANT_API_KEY"}

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Benchmark completed successfully!"
    echo "   Quality score: ≥80% (PASS)"
else
    echo "❌ Benchmark quality below threshold"
    echo "   Quality score: <80% (FAIL)"
fi
echo "=========================================="

exit $EXIT_CODE
