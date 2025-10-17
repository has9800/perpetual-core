# Perpetual AI V2 - Production Deployment Guide

Complete guide for deploying Perpetual AI V2 to production.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Database Setup](#database-setup)
- [Deployment Options](#deployment-options)
- [Configuration](#configuration)
- [Security](#security)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Services
- **Supabase** (PostgreSQL + Auth)
- **Qdrant** (Vector database - Cloud or self-hosted)
- **Redis** (Rate limiting & caching)
- **Polar.sh** (Optional: Billing)

### Required API Keys
- User provider keys (OpenAI, Anthropic, etc.) - stored encrypted by users
- Supabase project credentials
- Qdrant Cloud API key (if using cloud)
- Polar.sh API key (if using billing)

### System Requirements
- **Proxy Mode (Recommended for MVP):**
  - 2 CPU cores
  - 4GB RAM
  - 20GB storage
  - No GPU required

- **Local Mode (with vLLM):**
  - 8+ CPU cores
  - 32GB+ RAM
  - 80GB+ storage
  - NVIDIA GPU (16GB+ VRAM)

---

## Environment Setup

### 1. Generate Encryption Key

```bash
# Generate Fernet encryption key for user API keys
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Save this key securely - you'll need it for `API_KEY_ENCRYPTION_KEY`.

### 2. Create Environment File

Create `.env` in project root:

```bash
# App Configuration
APP_NAME="Perpetual AI"
VERSION="2.0.0"
DEBUG=false
MODE=proxy  # or "local" for vLLM

# API Server
API_HOST=0.0.0.0
API_PORT=8000
API_PREFIX=/v1

# Supabase (Auth & Database)
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_KEY=your-anon-key
SUPABASE_JWT_SECRET=your-jwt-secret

# Qdrant (Vector Database)
QDRANT_CLOUD_URL=https://xxxxx.cloud.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION=conversations

# Redis (Rate Limiting)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your-redis-password

# Security
API_KEY_ENCRYPTION_KEY=your-fernet-key-here  # From step 1

# Polar.sh (Optional: Billing)
POLAR_API_KEY=your-polar-api-key
POLAR_ORGANIZATION_ID=your-org-id

# Rate Limits (per tier)
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_DAY=10000
```

---

## Database Setup

### 1. Supabase Tables

Run the following SQL in your Supabase SQL editor:

```sql
-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email TEXT UNIQUE NOT NULL,
    tier TEXT DEFAULT 'free',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    total_requests INT DEFAULT 0,
    total_tokens INT DEFAULT 0,
    current_balance_usd FLOAT DEFAULT 0.0
);

-- API Keys table
CREATE TABLE IF NOT EXISTS api_keys (
    key_hash TEXT PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    tier TEXT DEFAULT 'free',
    rate_limit_per_minute INT DEFAULT 10,
    rate_limit_per_day INT DEFAULT 1000,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_used TIMESTAMP WITH TIME ZONE
);

-- Usage logs table
CREATE TABLE IF NOT EXISTS usage_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    conversation_id TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    input_tokens INT,
    output_tokens INT,
    total_tokens INT,
    retrieval_calls INT DEFAULT 0,
    retrieval_latency_ms FLOAT,
    cost_usd FLOAT,
    model TEXT,
    endpoint TEXT,
    status_code INT,
    latency_ms FLOAT
);

-- User provider keys table (encrypted)
CREATE TABLE IF NOT EXISTS user_provider_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    encrypted_key TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, provider)
);

-- Indexes
CREATE INDEX idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX idx_usage_logs_user_id ON usage_logs(user_id);
CREATE INDEX idx_usage_logs_timestamp ON usage_logs(timestamp);
CREATE INDEX idx_user_provider_keys_user_id ON user_provider_keys(user_id);

-- Enable RLS
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE usage_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_provider_keys ENABLE ROW LEVEL SECURITY;

-- RLS Policies
CREATE POLICY "Users can read own data" ON users FOR SELECT USING (auth.uid() = id);
CREATE POLICY "Users can manage own API keys" ON api_keys FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "Users can read own usage logs" ON usage_logs FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can manage own provider keys" ON user_provider_keys FOR ALL USING (auth.uid() = user_id);
```

### 2. Qdrant Setup

**Option A: Qdrant Cloud (Recommended)**
```bash
# Sign up at https://cloud.qdrant.io
# Create a cluster
# Copy URL and API key to .env
```

**Option B: Self-hosted Qdrant**
```bash
docker run -p 6333:6333 qdrant/qdrant
```

---

## Deployment Options

### Option 1: Docker (Recommended)

```bash
# 1. Build image
docker build -t perpetual-ai:v2 .

# 2. Run container
docker run -d \
  --name perpetual-ai \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  perpetual-ai:v2

# 3. Check logs
docker logs -f perpetual-ai

# 4. Check health
curl http://localhost:8000/health
```

### Option 2: Docker Compose

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Option 3: Kubernetes

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: perpetual-ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: perpetual-ai
  template:
    metadata:
      labels:
        app: perpetual-ai
    spec:
      containers:
      - name: perpetual-ai
        image: perpetual-ai:v2
        ports:
        - containerPort: 8000
        env:
        - name: MODE
          value: "proxy"
        envFrom:
        - secretRef:
            name: perpetual-secrets
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: perpetual-ai
spec:
  selector:
    app: perpetual-ai
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

Deploy:
```bash
kubectl apply -f k8s/deployment.yaml
```

### Option 4: Railway/Render/Fly.io

1. Connect GitHub repo
2. Add environment variables from `.env`
3. Set start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
4. Deploy

---

## Configuration

### Mode Selection

**Proxy Mode (Recommended for SaaS):**
```bash
MODE=proxy
```
- Forwards to external APIs (OpenAI, Anthropic, etc.)
- No GPU required
- Users provide their own API keys
- Instant startup

**Local Mode (For GPU Inference):**
```bash
MODE=local
MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
GPU_MEMORY_UTILIZATION=0.85
```
- Runs local vLLM inference
- Requires GPU
- 30-60s startup time
- Full control over models

### Tier Configuration

Edit `config/settings.py` or use environment variables:

```python
# Free tier
FREE_TIER_RATE_LIMIT_PER_MINUTE = 10
FREE_TIER_RATE_LIMIT_PER_DAY = 1000

# Pro tier
PRO_TIER_RATE_LIMIT_PER_MINUTE = 60
PRO_TIER_RATE_LIMIT_PER_DAY = 10000

# Enterprise tier
ENTERPRISE_TIER_RATE_LIMIT_PER_MINUTE = 1000
ENTERPRISE_TIER_RATE_LIMIT_PER_DAY = 1000000
```

---

## Security

### 1. HTTPS/TLS

**Using Nginx:**
```nginx
server {
    listen 443 ssl http2;
    server_name api.perpetual.ai;

    ssl_certificate /etc/letsencrypt/live/api.perpetual.ai/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.perpetual.ai/privkey.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

**Using Caddy:**
```
api.perpetual.ai {
    reverse_proxy localhost:8000
}
```

### 2. API Key Security

- ✅ All user provider keys encrypted with Fernet
- ✅ Encryption key stored in environment (not in code)
- ✅ Keys never logged or exposed in responses
- ✅ HTTPS required for all API calls

### 3. Rate Limiting

Configured per-tier in Redis:
- Prevents abuse
- Protects infrastructure
- Fair usage across users

### 4. Row Level Security (RLS)

Supabase RLS ensures:
- Users only access their own data
- API keys isolated per user
- Usage logs protected

---

## Monitoring

### 1. Health Checks

```bash
# Basic health
curl https://api.perpetual.ai/health

# Response:
{
  "status": "healthy",
  "version": "2.0.0",
  "uptime_seconds": 86400,
  "models_loaded": false,  # false in proxy mode
  "vector_db_connected": true
}
```

### 2. Prometheus Metrics

```bash
# Metrics endpoint
curl https://api.perpetual.ai/prometheus
```

**Key Metrics:**
- `perpetual_requests_total` - Total requests by endpoint/status
- `perpetual_request_duration_seconds` - Request latency histogram
- `perpetual_tokens_total` - Token usage by type/model
- `perpetual_retrievals_total` - Memory retrievals
- `perpetual_retrieval_duration_seconds` - Retrieval latency
- `perpetual_errors_total` - Errors by endpoint/type
- `perpetual_cost_usd_total` - Total cost by user/model

**Grafana Dashboard:**
```json
{
  "dashboard": {
    "title": "Perpetual AI Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "targets": ["rate(perpetual_requests_total[5m])"]
      },
      {
        "title": "P95 Latency",
        "targets": ["histogram_quantile(0.95, perpetual_request_duration_seconds)"]
      },
      {
        "title": "Token Usage",
        "targets": ["rate(perpetual_tokens_total[1h])"]
      }
    ]
  }
}
```

### 3. Logging

**Application Logs:**
```bash
# Docker
docker logs perpetual-ai

# Kubernetes
kubectl logs -f deployment/perpetual-ai

# File
tail -f /var/log/perpetual/app.log
```

**Log Levels:**
- `INFO` - Normal operation
- `WARNING` - Non-critical issues
- `ERROR` - Request failures
- `CRITICAL` - System failures

---

## Troubleshooting

### Common Issues

**1. "No API key found for provider"**
```bash
# User needs to add their provider API key
curl -X POST https://api.perpetual.ai/v1/provider-keys/add \
  -H "Authorization: Bearer $PERPETUAL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai",
    "api_key": "sk-..."
  }'
```

**2. "Authentication failed"**
- Check `PERPETUAL_API_KEY` is valid
- Verify API key in Supabase `api_keys` table
- Check `is_active = true`

**3. "Vector DB not connected"**
- Verify Qdrant URL and API key
- Check network connectivity
- Ensure collection exists

**4. "Rate limit exceeded"**
- Check Redis is running
- Verify rate limits for user's tier
- Check `/metrics` for current usage

**5. "Encryption error"**
- Verify `API_KEY_ENCRYPTION_KEY` is set consistently
- Don't change encryption key after storing keys
- Regenerate user keys if key changed

### Performance Tuning

**High Latency:**
```bash
# Check Prometheus metrics
curl https://api.perpetual.ai/prometheus | grep latency

# Check Qdrant performance
# Increase HNSW ef parameter

# Check provider API latency
# May be external API issue
```

**High Memory Usage:**
```bash
# Reduce Qdrant cache
# QDRANT_CACHE_SIZE=1000

# Reduce memory manager cache
# In code: cache_capacity=500
```

**Connection Pool Exhausted:**
```python
# Increase httpx client limits
timeout = httpx.Timeout(120.0, connect=10.0, pool=100.0)
```

---

## Pre-Launch Checklist

### Configuration
- [ ] All environment variables set
- [ ] Encryption key generated and stored securely
- [ ] Mode selected (proxy/local)
- [ ] Rate limits configured per tier

### Database
- [ ] Supabase tables created
- [ ] RLS policies enabled
- [ ] Indexes created
- [ ] Test data cleaned

### Security
- [ ] HTTPS/TLS configured
- [ ] API keys tested and working
- [ ] Rate limiting functional
- [ ] No secrets in code/logs

### Testing
- [ ] Health endpoint responds
- [ ] Auth flow works end-to-end
- [ ] Provider key management works
- [ ] Memory retrieval functional
- [ ] Billing tracking verified
- [ ] Benchmark tests pass

### Monitoring
- [ ] Prometheus metrics exporting
- [ ] Grafana dashboard set up
- [ ] Alerts configured
- [ ] Logs centralized

### Documentation
- [ ] API documentation published
- [ ] User guide created
- [ ] Troubleshooting guide available
- [ ] Support contact info provided

---

## Production Best Practices

### 1. High Availability
- Run 3+ replicas behind load balancer
- Use managed services (Supabase, Qdrant Cloud)
- Implement health checks and auto-restart

### 2. Scalability
- Horizontal scaling via container orchestration
- Redis for distributed rate limiting
- Qdrant Cloud auto-scales

### 3. Disaster Recovery
- Daily Supabase backups
- Qdrant snapshots
- Store encryption key in secrets manager (AWS Secrets Manager, HashiCorp Vault)

### 4. Cost Optimization
- Use spot instances for non-critical workloads
- Monitor token usage to prevent abuse
- Set billing alerts

### 5. Compliance
- Log retention policy (30-90 days)
- GDPR: User data deletion on request
- Encrypt data at rest and in transit

---

## Support

For issues:
- GitHub Issues: https://github.com/your-repo/issues
- Email: support@perpetual.ai
- Discord: https://discord.gg/perpetual

For enterprise support:
- Email: enterprise@perpetual.ai
- SLA: 24-hour response time
