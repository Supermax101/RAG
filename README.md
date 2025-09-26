# MistralOCR RAG 

Transform PDFs into a searchable knowledge base using Mistral OCR and embeddings.

## Quick start

1) Python 3.10+

2) Install deps:

```bash
pip install -r requirements.txt
```

3) Configure environment

- Create `.env` file and set your values:

```
MISTRAL_API_KEY=your_api_key_here
OCR_MODEL=mistral-ocr-latest
EMBED_MODEL=mistral-embed
OCR_BASE_URL=https://api.mistral.ai
EMBED_BASE_URL=https://api.mistral.ai
```

The app loads `.env` automatically.

4) Prepare folders (auto-created on first run):
- `data/raw_pdfs/` (drop your PDF files here)
- `data/parsed/`
- `data/embeddings/`
- `logs/`

## Workflow

### Phase 1: OCR Processing

```bash
# Process PDFs in batches (default: 10 at a time)
python -m app.main test-ingest

# Process specific batch size
python -m app.main test-ingest --batch-size 5

# Force reprocessing
python -m app.main test-ingest --force
```

### Phase 2: Create Embeddings

```bash
# Test embedding creation for one document (like test-ingest)
python -m app.main test-embedding

# Create embeddings for processed documents (batch of 10)
python -m app.main create-embeddings

# Process batch of 5 documents
python -m app.main create-embeddings --batch-size 5

# Force recreation
python -m app.main create-embeddings --force
```

### Phase 3: Search Setup

```bash
# Load embeddings into ChromaDB for fast search
python -m app.main load-chromadb

# Force reload
python -m app.main load-chromadb --force
```

### Phase 4: Search

```bash
# Search your documents
python -m app.main search "nutrition requirements"

# More results
python -m app.main search "parenteral nutrition" --limit 10

# Without content preview
python -m app.main search "electrolyte disorders" --no-content
```

## What gets created

### OCR Output Structure

For a PDF named `myfile.pdf` with derived `doc_id` like `myfile__a9f3c2`:

```
data/
├── parsed/
│   └── myfile/                              # Individual folder per PDF
│       ├── myfile.md                        # Pure Markdown
│       ├── myfile.rmd                       # R Markdown with YAML header
│       ├── myfile__a9f3c2.index.json        # Block metadata for embeddings
│       └── images/
│           ├── a1b2c3d4.png                 # SHA-named image files
│           └── e5f6g7h8.png
│
├── embeddings/
│   ├── chromadb/                            # ChromaDB vector database
│   │   ├── chroma.sqlite3
│   │   └── collection_data/
│   ├── vectors/                             # File-based embedding backup
│   │   ├── myfile__a9f3c2.embeddings.json  # Embedding vectors + metadata
│   │   └── ...
│   └── metadata/                            # Global indexes (future)
│       └── global_index.json
│
└── raw_pdfs/                                # Original PDF files
    ├── myfile.pdf
    └── ...
```

### File Details

- **myfile.md**: Clean markdown (Mistral's raw output)
- **myfile.rmd**: R Markdown with YAML header for R/RStudio
- **myfile__a9f3c2.index.json**: Block-level metadata with line numbers, sections, image references
- **images/**: SHA-named image files for global uniqueness
- **embeddings.json**: Complete embedding vectors with chunk metadata for backup/portability
- **chromadb/**: Fast vector similarity search database

## Architecture Features

✅ **Hybrid Storage**: ChromaDB for speed + files for backup  
✅ **Batch Processing**: Automatic batching with smart skip logic  
✅ **Image Preservation**: SHA-based naming prevents duplicates  
✅ **Multimodal Ready**: Image-text relationships preserved  
✅ **LLM Friendly**: Clean markdown output, no base64 mess  
✅ **Portable**: File-based embeddings work without databases  

## API Limits & Costs

- **OCR**: ~50 MB, 1,000 pages per PDF ([Mistral docs](https://docs.mistral.ai/capabilities/vision/))
- **Embeddings**: ~$0.0001-0.0003 per 1K tokens (estimated)
- **Dimensions**: 1024 per embedding vector (manageable size)

## Commands Reference

| Command | Purpose |
|---------|---------|
| `test-ingest` | Process PDFs with OCR (batch) |
| `test-embedding` | Test embedding for one document |
| `create-embeddings` | Generate vector embeddings (batch) |
| `load-chromadb` | Load embeddings into vector DB |
| `search` | Semantic search through documents |

## Next Steps

This system is ready for RAG integration with any LLM:
- **Text-only models**: Use the clean markdown + embedding search
- **Vision models**: Access images via the preserved image paths
- **Local or API**: Works with both ChromaDB and file-based storage
