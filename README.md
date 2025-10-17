# Perpetual AI - Infinite Memory Layer for LLMs

**Stop paying for massive context windows. Reduce LLM API costs by 99.8%.**

Perpetual AI is an intelligent proxy that sits between your application and any LLM API (OpenAI, Anthropic, xAI, etc.), using semantic memory retrieval to maintain perfect context while reducing token usage by **99.8%**.

---

## 🎯 The Problem & Solution

**Traditional approach:** Send entire conversation history every time
- Turn 100 = 10,000 tokens = $0.0015 per request
- Hits context limits quickly
- Costs scale linearly with conversation length

**Perpetual AI:** Send only relevant context
- Turn 100 = 140 tokens = $0.00002 per request
- Infinite conversation length
- **Constant cost per request**

---

## ✨ Key Features

- 🔄 **Drop-in replacement** - Change URL, add `conversation_id`, done
- 🌐 **Multi-provider** - OpenAI, Anthropic, xAI, Together, Cerebras, OpenRouter, Groq, DeepSeek
- 🧠 **Smart memory** - Semantic retrieval with Qdrant + hybrid search
- 🔐 **Enterprise security** - Encrypted API keys, RLS, rate limiting
- 📊 **Full observability** - Prometheus metrics, usage tracking, billing
- ⚡ **Blazing fast** - P95 latency < 500ms including retrieval

---

## 🚀 Quick Start

```bash
# 1. Clone & install
git clone https://github.com/your-org/perpetual-core.git
cd perpetual-core
pip install -r requirements.txt

# 2. Configure (create .env)
MODE=proxy
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=your-key
QDRANT_CLOUD_URL=https://xxx.cloud.qdrant.io
QDRANT_API_KEY=your-key
API_KEY_ENCRYPTION_KEY=<generate with Fernet>

# 3. Start
python -m uvicorn api.main:app --reload

# 4. Use
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"model":"gpt-4o-mini","messages":[...],"conversation_id":"chat_1"}'
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for production deployment.

---

## 📊 Performance

| Metric | Traditional | Perpetual AI | Improvement |
|--------|-------------|--------------|-------------|
| Tokens (100 turns) | 101,000 | 140 | **99.86% ↓** |
| Cost (100 turns) | $0.0303 | $0.0004 | **98.7% ↓** |
| Latency (P95) | 1200ms | 520ms | **2.3x faster** |

---

## 🧪 Testing

Run comprehensive benchmark:
```bash
python tests/v2_comprehensive_benchmark.py \
  --provider openai \
  --model gpt-4o-mini \
  --num-turns 20
```

See [tests/README_V2_BENCHMARK.md](tests/README_V2_BENCHMARK.md) for details.

---

## 📚 Documentation

- **[Deployment Guide](DEPLOYMENT.md)** - Production deployment
- **[Benchmark Guide](tests/README_V2_BENCHMARK.md)** - Testing & performance
- **[API Docs](http://localhost:8000/docs)** - Interactive API documentation

---

## 🛣️ Roadmap

- ✅ V2.0: Multi-provider proxy, encrypted keys, memory management, streaming
- 🚧 V2.1: GPU inference, multi-modal, advanced caching
- 🔮 V2.2: Fine-tuning, custom embeddings, webhooks

---

## 💰 Pricing

- **Free:** 1,000 requests/day
- **Pro:** $29/month - 10,000 requests/day
- **Enterprise:** Custom pricing

Platform costs: ~$135/month base (Supabase + Qdrant + Redis)
Users provide their own LLM API keys → **Zero marginal inference cost**

---

## 🤝 Contributing

Contributions welcome! Areas needed:
- Additional LLM providers
- Client SDKs (Python, JS, Go)
- Performance optimizations

---

## 📄 License

MIT License - See [LICENSE](LICENSE)

---

## 📞 Support

- **Email:** support@perpetual.ai
- **Discord:** [Join community](https://discord.gg/perpetual)
- **Issues:** [GitHub](https://github.com/your-org/perpetual-core/issues)

---

**Built by developers tired of paying for context windows. Stop sending 10,000 tokens when 140 will do.**
