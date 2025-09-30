# OpenAI Integration Setup Guide

This guide explains how to use OpenAI models (GPT-5, GPT-4, etc.) in the TPN RAG evaluation system.

## 🚀 Quick Start

### 1. Install OpenAI SDK (Already Done)
```bash
uv add openai
```

### 2. Set Your OpenAI API Key

Choose one of the following methods:

#### Option A: Environment Variable (Recommended)
```bash
export OPENAI_API_KEY="your_api_key_here"
```

#### Option B: Create .env File
Create a `.env` file in the project root:
```bash
# .env
OPENAI_API_KEY=your_openai_api_key_here
```

#### Option C: Set in Your Shell Profile
Add to `~/.zshrc` or `~/.bashrc`:
```bash
export OPENAI_API_KEY="your_api_key_here"
```

### 3. Get Your API Key

1. Go to https://platform.openai.com/api-keys
2. Click "Create new secret key"
3. Copy the key (starts with `sk-...`)
4. Use it in the methods above

## 🎯 How It Works

The evaluation system now supports **both** Ollama (local) and OpenAI (cloud) models:

```
Available Models:
  1. [OLLAMA]  mistral:7b        (7B parameters)
  2. [OLLAMA]  phi4:latest        (Phi-4, 14B params)
  3. [OPENAI]  gpt-5              (Most capable)
  4. [OPENAI]  gpt-5-mini         (Balanced)
  5. [OPENAI]  gpt-5-nano         (Fast & cheap)
```

### Model Selection
When you run the evaluation:
```bash
uv run python eval/tpn_rag_evaluation.py
```

You'll be able to select from **ALL** available models (Ollama + OpenAI).

## 💰 Cost Estimation (GPT-5)

For 48 MCQ questions with retrieved context:

| Model | Input Cost | Output Cost | Total/Run |
|-------|-----------|-------------|-----------|
| GPT-5 | ~$0.31 | ~$0.01 | **~$0.32** |
| GPT-5 Mini | ~$0.06 | ~$0.01 | **~$0.07** |
| GPT-5 Nano | ~$0.01 | ~$0.01 | **~$0.02** |

**Benchmark mode** (all models): Cost = number_of_models × cost_per_run

## 🔧 Architecture

### Provider Pattern
```python
# Abstract interface
class LLMProvider(ABC):
    async def generate(prompt, model, temperature, max_tokens) -> str
    async def available_models() -> List[str]

# Implementations
- OllamaLLMProvider  # Local models (free)
- OpenAILLMProvider  # Cloud models (paid)
```

### Auto-Detection
The system automatically:
1. Detects available Ollama models (if running)
2. Detects available OpenAI models (if API key set)
3. Combines them in the selection menu
4. Routes to the correct provider based on selection

## 📊 Benchmark Comparison

You can compare Ollama vs OpenAI models:

```bash
uv run python eval/tpn_rag_evaluation.py
# Select mode: 2 (Benchmark ALL models)
```

This will test all models and generate a comparison report:
```
RANKING BY ACCURACY:
Rank  Model                              Accuracy    Correct     Avg Time
🥇 1    [OPENAI] gpt-5                    95.83%      46/48       850ms
🥈 2    [OLLAMA] phi4:latest              89.58%      43/48       1250ms
🥉 3    [OPENAI] gpt-5-mini               87.50%      42/48       650ms
   4    [OLLAMA] mistral:7b               81.25%      39/48       1100ms
```

## 🔍 Troubleshooting

### Error: "OpenAI API is not accessible"
**Causes:**
1. API key not set → Check `echo $OPENAI_API_KEY`
2. Invalid API key → Verify on OpenAI dashboard
3. Insufficient credits → Add credits to your account
4. Network issue → Check internet connection

**Solution:**
```bash
# Verify API key is set
echo $OPENAI_API_KEY

# Test API connection
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### No OpenAI Models Showing Up
**Cause:** API key not set

**Solution:**
```bash
export OPENAI_API_KEY="sk-..."
# Then rerun the evaluation script
```

### Rate Limiting
If you hit rate limits:
1. Add delays between requests
2. Use a higher tier API plan
3. Use GPT-5 Nano (faster, fewer rate limits)

## 🎓 Example Usage

### Single Model Evaluation
```bash
export OPENAI_API_KEY="sk-..."
uv run python eval/tpn_rag_evaluation.py

# Select mode: 1 (Single model)
# Choose: [OPENAI] gpt-5
# Questions: all
```

### Benchmark Mode
```bash
export OPENAI_API_KEY="sk-..."
uv run python eval/tpn_rag_evaluation.py

# Select mode: 2 (Benchmark ALL)
# Questions: 10 (for quick testing)
```

## 📝 Notes

- **Ollama is still free** - No API key needed for local models
- **OpenAI is optional** - System works with just Ollama
- **Hybrid evaluation** - Compare local vs cloud models
- **Cost tracking** - Results include model type for cost analysis
- **Same RAG system** - Both providers use identical retrieval pipeline

## 🔐 Security

⚠️ **Never commit your API key to Git!**

The `.env` file is already in `.gitignore`. Always use:
- Environment variables
- `.env` file (gitignored)
- Secret management tools (for production)

## 📚 Additional Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [OpenAI Pricing](https://openai.com/pricing)
- [Rate Limits Guide](https://platform.openai.com/docs/guides/rate-limits)
