# Medical RAG System with Ollama

🏥 **Private, secure medical RAG system** using **Ollama** for embeddings and LLM inference - **no external API keys required!**

## 🚀 Quick Start (Ollama Version)

### 1. Install Ollama Models

```bash
# Install embedding model (required)
ollama pull nomic-embed-text

# Install LLM model (choose one or more)
ollama pull gpt-oss          # 12.8 GB - Good for medical queries
ollama pull mistral:7b       # 4.1 GB - Faster, smaller
ollama pull llama3.1:8b      # 4.7 GB - Alternative option
```

### 2. Load Medical Documents

```bash
# Load all 52 medical documents into ChromaDB
python load_medical_docs.py
```

This will:
- Process all 52 medical PDF documents (ASPEN guidelines, NICU protocols, TPN guides)
- Create embeddings using Ollama's `nomic-embed-text` model
- Store everything in ChromaDB for fast search
- Takes ~5-10 minutes for complete knowledge base

### 3. Start Medical RAG System

```bash
# Interactive medical Q&A system
python ollama_rag.py
```

## 🏥 Medical Knowledge Base

Your system includes **52 authoritative medical documents**:

- **ASPEN Guidelines**: Nutrition support, parenteral nutrition safety
- **NICU Protocols**: Neonatal nutrition, premature infant care  
- **TPN Administration**: Guidelines from multiple hospitals
- **Electrolyte Management**: Fluids, acid-base disorders
- **Nutrition Assessment**: Malnutrition prevention and treatment
- **Specialized Populations**: Pediatric, neonatal requirements

## 💻 Usage Examples

### Interactive Mode
```bash
python ollama_rag.py
```

Sample questions you can ask:
- "What are the indications for parenteral nutrition in neonates?"
- "How do you calculate TPN requirements for premature infants?"
- "What are normal serum potassium levels in pediatric patients?"
- "How do you prevent refeeding syndrome?"
- "What are the ASPEN guidelines for lipid administration?"

### Programmatic Usage
```python
from app.ollama_search import OllamaRAGSearch
from ollama_rag import ollama_rag_answer

# Initialize search
rag_search = OllamaRAGSearch()

# Get medical answer
result = ollama_rag_answer(
    "What are the signs of essential fatty acid deficiency?",
    model="gpt-oss"
)

print(result["answer"])
```

## 🔧 System Architecture

```
Medical Documents (52 PDFs)
           ↓
    Parsed Content (.md files)
           ↓
    Ollama Embeddings (nomic-embed-text)
           ↓
    ChromaDB Vector Storage
           ↓
    Semantic Search + Context Retrieval
           ↓
    Ollama LLM Generation (gpt-oss/mistral)
           ↓
    Clinical Answer with Citations
```

## 📊 Features

✅ **Fully Offline**: No external API calls or internet required
✅ **Medical Accuracy**: Evidence-based answers from authoritative sources  
✅ **Fast Search**: ChromaDB vector similarity search (~100ms)
✅ **Source Citations**: Every answer includes document sources
✅ **Scalable**: Handles 3,000+ medical document chunks
✅ **GPU Accelerated**: Uses RTX 4090 for fast inference on Vast.ai

## 🛠️ Advanced Configuration

### Custom Ollama Setup
```python
# Use different embedding model
rag_search = OllamaRAGSearch(
    ollama_url="http://localhost:11434",
    embedding_model="nomic-embed-text"  # or other embedding models
)

# Use different LLM model
result = ollama_rag_answer(
    question="Your medical question",
    model="mistral:7b",  # or gpt-oss, llama3.1:8b
    search_limit=5
)
```

### Environment Variables
```bash
# Optional - customize Ollama connection
export OLLAMA_URL=http://localhost:11434
export EMBEDDING_MODEL=nomic-embed-text
export DEFAULT_LLM_MODEL=gpt-oss
```

## 🔍 Troubleshooting

### Ollama Connection Issues
```bash
# Check Ollama status
curl http://localhost:11434/api/version

# Restart Ollama service
systemctl restart ollama
```

### ChromaDB Issues
```bash
# Clear and reload documents
rm -rf data/embeddings/chromadb/
python load_medical_docs.py
```

### Model Issues
```bash
# Check available models
ollama list

# Pull missing models
ollama pull nomic-embed-text
ollama pull gpt-oss
```

## 📈 Performance

**Vast.ai RTX 4090 Performance:**
- **Document Loading**: ~3,000 chunks in 5-10 minutes
- **Search Speed**: ~100-200ms per query
- **Answer Generation**: 2-5 seconds depending on model
- **Memory Usage**: ~2-4GB RAM, ~8GB VRAM for gpt-oss

## 🔐 Privacy & Security

- ✅ **Fully Local**: All processing happens on your VM
- ✅ **No External APIs**: No data sent to external servers
- ✅ **Private Knowledge**: Your medical documents stay private
- ✅ **Secure Deployment**: Ready for hospital/clinical environments

## 📚 Original Features Still Available

The original Mistral API version is still available in:
- `test_rag.py` - Original RAG system
- `app/main.py` - Original CLI interface
- `evaluate_rag.py` - Model evaluation system

## 🤝 Contributing

This system is designed for medical education and clinical decision support. Always validate medical information with current clinical guidelines and consult healthcare professionals for patient care decisions.
