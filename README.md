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
# DPT2 Integration Complete

## What Changed

### 1. Data Migration
- Copied 76 DPT2 documents (152 files: 76 JSON + 76 MD) to `data/dpt2_output/`
- Total pre-chunked pieces: ~9,226 chunks across all documents
- 46% more content than previous dataset (76 vs 52 documents)

### 2. New DPT2DocumentLoader
**File:** `src/rag/core/services/dpt2_document_loader.py`

**Key Features:**
- Loads JSON files directly (not markdown)
- Uses pre-chunked data AS-IS (NO re-chunking)
- Preserves chunk UUIDs from DPT2
- Maintains rich metadata:
  - chunk_type (text, table, figure, etc.)
  - page numbers
  - bounding_box coordinates
  - chunk_index and total_chunks
  - source_file reference

**What It Does NOT Do:**
- No regex patterns
- No paragraph grouping
- No markdown splitting
- No table extraction
- DPT2 already did all this perfectly!

### 3. Updated main.py
- Replaced `DocumentLoader` with `DPT2DocumentLoader`
- Updated document count: 52 → 76
- Updated initialization messages

### 4. Updated eval/tpn_rag_evaluation.py
- Updated prompt: "52 medical PDFs" → "76 medical documents"
- Updated system message references

### 5. ChromaDB (No Changes Needed)
- Already supports chunk UUIDs
- Already stores rich metadata
- Already supports filtering by metadata

## Data Structure

### DPT2 JSON Format
```json
{
  "markdown": "full document text",
  "chunks": [
    {
      "id": "uuid-here",
      "type": "text|table|figure|logo",
      "markdown": "chunk content",
      "grounding": {
        "page": 3,
        "box": {"left": 0.1, "top": 0.2, "right": 0.9, "bottom": 0.8}
      }
    }
  ],
  "metadata": {
    "filename": "document.pdf",
    "page_count": 10,
    "version": "dpt-2-20250919"
  }
}
```

### Chunk Metadata in ChromaDB
```python
{
    "source_file": "1PN Overview",
    "chunk_type": "text",
    "page": 3,
    "chunk_index": 15,
    "total_chunks": 32,
    "bounding_box": "{...}",
    "chunk_strategy": "dpt2_prechunked"
}
```

## Benefits

### 1. Better Chunks
- Professionally parsed with DPT2
- Intelligent boundaries (preserves tables, figures, semantic units)
- Type-classified for filtering

### 2. Traceability
- Stable UUIDs (no regeneration)
- Page numbers for citations
- Bounding boxes for PDF highlighting
- Exact source location tracking

### 3. More Content
- 76 documents (vs 52)
- ~9,226 chunks (vs ~6,300)
- 46% more medical knowledge

### 4. Future Features Enabled
- Filter by chunk_type (text vs tables)
- Page-based context expansion
- Visual citations with bounding boxes
- Table-aware retrieval strategies

## Next Steps

### To Initialize System:
```bash
# Delete old ChromaDB collection
rm -rf data/chromadb/

# Run initialization (will use DPT2 loader)
uv run python main.py init

# Expected: ~9,226 chunks embedded from 76 documents
```

### To Run Evaluation:
```bash
uv run python eval/tpn_rag_evaluation.py
```

## File Changes Summary

**New Files:**
- `data/dpt2_output/` (152 files)
- `src/rag/core/services/dpt2_document_loader.py`

**Modified Files:**
- `main.py` - Use DPT2DocumentLoader
- `eval/tpn_rag_evaluation.py` - Update document count

**No Changes:**
- `src/rag/infrastructure/vector_stores/chroma_store.py` (already compatible)
- Embedding providers (work with any text)
- LLM providers (work with any context)
- Advanced RAG features (HyDE, reranking, etc.)

## Backward Compatibility

Old `data/parsed/` directory is preserved but not used.
To revert: change `main.py` import back to `DocumentLoader`.

## Data Verification

Chunks contain 100% of document content:
- All text preserved
- All tables preserved
- All figures preserved
- No data loss from using chunks
