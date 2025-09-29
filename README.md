# Document RAG System v2.0

A modern, clean RAG (Retrieval-Augmented Generation) system with FastAPI, ChromaDB, and Ollama integration.

## 🏗️ Architecture

```
📦 Modern RAG System
├── 🏗️  src/rag/
│   ├── core/              # Business Logic
│   │   ├── models/        # Data models (Pydantic)
│   │   ├── services/      # RAG & Document services
│   │   └── interfaces/    # Abstract interfaces
│   ├── infrastructure/    # External Integrations
│   │   ├── embeddings/    # Ollama embeddings
│   │   ├── vector_stores/ # ChromaDB adapter
│   │   └── llm_providers/ # Ollama LLM client
│   ├── api/              # FastAPI Application
│   │   ├── routes/       # API endpoints
│   │   └── schemas/      # Request/Response models
│   └── config/           # Configuration
├── 📄 ocr_pipeline/       # Mistral OCR (preserved)
└── 📊 data/              # Documents & embeddings
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

### 2. Setup Environment

Create `.env` file:

```bash
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBED_MODEL=nomic-embed-text

# Optional: Mistral API (for OCR pipeline)
MISTRAL_API_KEY=your_key_here
```

### 3. Start Ollama

```bash
# Install and start Ollama
ollama serve

# Pull required models
ollama pull nomic-embed-text
ollama pull mistral:7b
```

### 4. Initialize System

```bash
# Initialize and load documents
python main.py init

# Run interactive demo
python main.py demo

# Start API server
python main.py serve
```

## 📡 API Endpoints

- **Health Check**: `GET /health`
- **Search Documents**: `POST /api/v1/search`
- **Ask Question**: `POST /api/v1/ask`
- **Get Stats**: `GET /api/v1/stats`

Full API documentation: `http://localhost:8000/docs`

## 📄 OCR Pipeline (Preserved)

For processing new PDFs:

```bash
# Add PDFs to data/raw_pdfs/
python -m ocr_pipeline.main test-ingest

# Create embeddings
python -m ocr_pipeline.main create-embeddings

# Reload into vector store
python main.py init
```

## 🔧 Development

```bash
# Format code
uv run black src/

# Type checking
uv run mypy src/

# Run tests
uv run pytest
```

## 📊 Features

- ✅ Clean, modern architecture
- ✅ FastAPI with automatic docs
- ✅ ChromaDB vector storage
- ✅ Ollama LLM & embeddings
- ✅ Async/await throughout
- ✅ Type safety with Pydantic
- ✅ Connection pooling & caching
- ✅ Preserved OCR pipeline
- ✅ UV package management
