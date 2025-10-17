# =============================================================================
# PERPETUAL AI - PRODUCTION DOCKERFILE
# Multi-stage build for optimal image size
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Base image with CUDA
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# -----------------------------------------------------------------------------
# Stage 2: Dependencies
# -----------------------------------------------------------------------------
FROM base AS dependencies

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 3: Application
# -----------------------------------------------------------------------------
FROM dependencies AS application

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p /app/data/qdrant_db

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["python3", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
