"""
Mistral Embeddings API integration for creating vector embeddings from parsed documents.

This module handles:
- Text chunking for optimal embedding size
- Batch processing with the Mistral embeddings API
- ChromaDB storage for fast vector similarity search
- File-based embedding backup for portability
- Image-text relationship preservation for RAG

Official Mistral Embeddings Documentation:
https://docs.mistral.ai/capabilities/embeddings/text_embeddings/
"""

from __future__ import annotations
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict

import numpy as np
from mistralai import Mistral

from .config import (
    MISTRAL_API_KEY,
    DEFAULT_EMBED_MODEL,
    EMBED_BASE_URL,
    EMBED_ENDPOINT_PATH,
    EMBED_BATCH_SIZE,
    EMBED_CHUNK_SIZE,
    VECTORS_DIR,
    METADATA_DIR,
    get_logger,
)


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata for embedding."""
    content: str
    chunk_id: str
    doc_id: str
    chunk_index: int
    section: str
    page_num: Optional[int] = None
    chunk_type: str = "text"  # text, heading, table, image_caption
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    nearby_images: List[str] = None  # SHA hashes of nearby images

    def __post_init__(self):
        if self.nearby_images is None:
            self.nearby_images = []


@dataclass
class EmbeddingResult:
    """Result from embedding API call."""
    chunk_id: str
    embedding: List[float]
    model: str
    tokens_used: int
    request_id: Optional[str] = None


@dataclass
class DocumentEmbeddings:
    """Complete embedding data for a document."""
    doc_id: str
    original_filename: str
    chunks: List[TextChunk]
    embeddings: List[EmbeddingResult]
    model: str
    created_at: str
    total_tokens: int
    processing_time_seconds: float


logger = get_logger("embedding-runner")


def _chunk_text(text: str, chunk_size: int = EMBED_CHUNK_SIZE, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks for embedding."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary if possible
        if end < len(text):
            # Look for sentence endings within the last 100 chars
            sentence_end = text.rfind('.', start + chunk_size - 100, end)
            if sentence_end != -1 and sentence_end > start + chunk_size // 2:
                end = sentence_end + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks


def _extract_chunks_from_index(index_path: Path) -> List[TextChunk]:
    """Extract text chunks from a document's index.json file."""
    try:
        with open(index_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        
        doc_id = index_data.get('doc_id', '')
        blocks = index_data.get('blocks', [])
        
        chunks = []
        chunk_counter = 0
        
        for block in blocks:
            block_type = block.get('type', 'text')
            section = block.get('section', '')
            line_num = block.get('line')
            
            # Get content based on block type
            if block_type == 'heading':
                content = f"# {section}"
            elif block_type == 'image':
                content = block.get('content', '')
                block_type = 'image_caption'
            elif block_type == 'table':
                content = block.get('preview', '')
                block_type = 'table'
            else:  # text
                content = block.get('preview', '')
            
            if not content or not content.strip():
                continue
            
            # Split large content into smaller chunks
            text_chunks = _chunk_text(content)
            
            for i, chunk_text in enumerate(text_chunks):
                chunk_id = f"{doc_id}__chunk_{chunk_counter:04d}"
                
                chunk = TextChunk(
                    content=chunk_text,
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    chunk_index=chunk_counter,
                    section=section,
                    chunk_type=block_type,
                    line_start=line_num,
                    line_end=line_num,
                )
                
                chunks.append(chunk)
                chunk_counter += 1
        
        logger.info(f"Extracted {len(chunks)} chunks from {index_path.name}")
        return chunks
        
    except Exception as e:
        logger.error(f"Failed to extract chunks from {index_path}: {e}")
        return []


def _call_embeddings_api(texts: List[str], model: str = DEFAULT_EMBED_MODEL) -> Tuple[List[List[float]], int, Optional[str]]:
    """Call Mistral embeddings API with batch of texts."""
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY not found in environment")
    
    client = Mistral(api_key=MISTRAL_API_KEY)
    
    try:
        logger.debug(f"Calling embeddings API with {len(texts)} texts")
        
        response = client.embeddings.create(
            model=model,
            inputs=texts
        )
        
        # Extract embeddings in the correct order
        embeddings = []
        for data in sorted(response.data, key=lambda x: x.index):
            embeddings.append(data.embedding)
        
        total_tokens = response.usage.total_tokens if response.usage else 0
        request_id = getattr(response, 'id', None)
        
        logger.debug(f"API call successful: {len(embeddings)} embeddings, {total_tokens} tokens")
        return embeddings, total_tokens, request_id
        
    except Exception as e:
        logger.error(f"Embeddings API call failed: {e}")
        raise


def _process_chunks_in_batches(chunks: List[TextChunk], model: str = DEFAULT_EMBED_MODEL) -> List[EmbeddingResult]:
    """Process chunks in batches through the embeddings API."""
    results = []
    total_tokens = 0
    
    # Process in batches
    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch_chunks = chunks[i:i + EMBED_BATCH_SIZE]
        batch_texts = [chunk.content for chunk in batch_chunks]
        
        logger.info(f"Processing batch {i // EMBED_BATCH_SIZE + 1}/{(len(chunks) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE}")
        
        try:
            embeddings, tokens_used, request_id = _call_embeddings_api(batch_texts, model)
            total_tokens += tokens_used
            
            # Create results for this batch
            for chunk, embedding in zip(batch_chunks, embeddings):
                result = EmbeddingResult(
                    chunk_id=chunk.chunk_id,
                    embedding=embedding,
                    model=model,
                    tokens_used=tokens_used // len(batch_chunks),  # Approximate per chunk
                    request_id=request_id
                )
                results.append(result)
            
            # Rate limiting - be nice to the API
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Failed to process batch {i // EMBED_BATCH_SIZE + 1}: {e}")
            # Continue with other batches
            continue
    
    logger.info(f"Processed {len(results)} chunks using {total_tokens} tokens")
    return results


def _save_embeddings_to_file(doc_embeddings: DocumentEmbeddings) -> Path:
    """Save embeddings to file for backup and portability."""
    filename = f"{doc_embeddings.doc_id}.embeddings.json"
    file_path = VECTORS_DIR / filename
    
    # Convert to serializable format
    data = {
        "doc_id": doc_embeddings.doc_id,
        "original_filename": doc_embeddings.original_filename,
        "model": doc_embeddings.model,
        "created_at": doc_embeddings.created_at,
        "total_tokens": doc_embeddings.total_tokens,
        "processing_time_seconds": doc_embeddings.processing_time_seconds,
        "chunks": [asdict(chunk) for chunk in doc_embeddings.chunks],
        "embeddings": [asdict(embedding) for embedding in doc_embeddings.embeddings]
    }
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved embeddings to {file_path}")
    return file_path


def create_embeddings_for_document(index_path: Path, model: str = DEFAULT_EMBED_MODEL) -> Optional[DocumentEmbeddings]:
    """Create embeddings for a single document from its index.json file."""
    start_time = time.time()
    
    try:
        # Extract chunks from index
        chunks = _extract_chunks_from_index(index_path)
        if not chunks:
            logger.warning(f"No chunks extracted from {index_path}")
            return None
        
        # Get document metadata
        with open(index_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        
        doc_id = index_data.get('doc_id', '')
        original_filename = index_data.get('original_filename', '')
        
        logger.info(f"Creating embeddings for {original_filename} ({len(chunks)} chunks)")
        
        # Process chunks through embeddings API
        embedding_results = _process_chunks_in_batches(chunks, model)
        
        if not embedding_results:
            logger.error(f"No embeddings created for {original_filename}")
            return None
        
        # Create document embeddings object
        processing_time = time.time() - start_time
        total_tokens = sum(result.tokens_used for result in embedding_results)
        
        doc_embeddings = DocumentEmbeddings(
            doc_id=doc_id,
            original_filename=original_filename,
            chunks=chunks,
            embeddings=embedding_results,
            model=model,
            created_at=time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            total_tokens=total_tokens,
            processing_time_seconds=processing_time
        )
        
        # Save to file
        _save_embeddings_to_file(doc_embeddings)
        
        logger.info(f"Completed embeddings for {original_filename} in {processing_time:.1f}s")
        return doc_embeddings
        
    except Exception as e:
        logger.error(f"Failed to create embeddings for {index_path}: {e}")
        return None


def embeddings_exist_for_doc(doc_id: str) -> bool:
    """Check if embeddings already exist for a document."""
    embedding_file = VECTORS_DIR / f"{doc_id}.embeddings.json"
    return embedding_file.exists()


def load_document_embeddings(doc_id: str) -> Optional[DocumentEmbeddings]:
    """Load existing embeddings for a document."""
    embedding_file = VECTORS_DIR / f"{doc_id}.embeddings.json"
    
    if not embedding_file.exists():
        return None
    
    try:
        with open(embedding_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunks = [TextChunk(**chunk_data) for chunk_data in data['chunks']]
        embeddings = [EmbeddingResult(**emb_data) for emb_data in data['embeddings']]
        
        return DocumentEmbeddings(
            doc_id=data['doc_id'],
            original_filename=data['original_filename'],
            chunks=chunks,
            embeddings=embeddings,
            model=data['model'],
            created_at=data['created_at'],
            total_tokens=data['total_tokens'],
            processing_time_seconds=data['processing_time_seconds']
        )
        
    except Exception as e:
        logger.error(f"Failed to load embeddings for {doc_id}: {e}")
        return None
