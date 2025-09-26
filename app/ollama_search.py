"""
RAG Search functionality using ChromaDB and Ollama embeddings.

This is an Ollama-compatible version that replaces Mistral API calls
with local Ollama embedding generation.
"""

from __future__ import annotations
import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings

from .config import (
    CHROMADB_DIR,
    CHROMA_COLLECTION_NAME, 
    PARSED_DIR,
    get_logger,
)
from .ollama_embeddings import get_ollama_embedding

logger = get_logger("ollama-search")


@dataclass
class SearchResult:
    """Single search result with metadata."""
    chunk_id: str
    doc_id: str
    content: str
    score: float
    document_name: str
    section: str
    chunk_type: str = "text"


@dataclass
class SearchResults:
    """Collection of search results with metadata."""
    results: List[SearchResult]
    query: str
    total_results: int
    search_time: float
    model_used: str


class OllamaRAGSearch:
    """RAG search system using Ollama embeddings and ChromaDB."""
    
    def __init__(self, 
                 ollama_url: str = "http://localhost:11434",
                 embedding_model: str = "nomic-embed-text"):
        self.ollama_url = ollama_url
        self.embedding_model = embedding_model
        self.client = chromadb.PersistentClient(path=str(CHROMADB_DIR))
        
        try:
            self.collection = self.client.get_collection(name=CHROMA_COLLECTION_NAME)
            logger.info(f"Connected to existing ChromaDB collection: {CHROMA_COLLECTION_NAME}")
        except:
            self.collection = self.client.create_collection(name=CHROMA_COLLECTION_NAME)
            logger.info(f"Created new ChromaDB collection: {CHROMA_COLLECTION_NAME}")
    
    def load_all_documents(self) -> int:
        """
        Load all medical documents into ChromaDB with Ollama embeddings.
        
        Returns:
            Number of chunks loaded
        """
        logger.info("Loading all medical documents with Ollama embeddings...")
        
        # Get all document directories
        docs = [d for d in PARSED_DIR.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        # Clear existing collection
        try:
            self.client.delete_collection(name=CHROMA_COLLECTION_NAME)
            self.collection = self.client.create_collection(name=CHROMA_COLLECTION_NAME)
            logger.info("Cleared and recreated ChromaDB collection")
        except Exception as e:
            logger.warning(f"Could not clear collection: {e}")
        
        all_texts = []
        all_metadatas = []
        all_ids = []
        all_embeddings = []
        
        chunk_count = 0
        processed_docs = 0
        
        for doc_dir in docs:
            doc_name = doc_dir.name
            md_file = doc_dir / f"{doc_name}.md"
            
            if md_file.exists():
                try:
                    with open(md_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Split into chunks
                    chunk_size = 1500  # Slightly larger chunks for better context
                    doc_chunks = 0
                    for i in range(0, len(content), chunk_size):
                        chunk_text = content[i:i+chunk_size].strip()
                        if chunk_text and len(chunk_text) > 100:  # Only meaningful chunks
                            chunk_id = f"{doc_name}_{i//chunk_size}_{hashlib.md5(chunk_text.encode()).hexdigest()[:8]}"
                            
                            # Get embedding from Ollama
                            embedding = get_ollama_embedding(
                                chunk_text, 
                                self.ollama_url, 
                                self.embedding_model
                            )
                            
                            all_texts.append(chunk_text)
                            all_metadatas.append({
                                'document': doc_name,
                                'chunk_index': i//chunk_size,
                                'source': 'medical_documents',
                                'doc_id': doc_name
                            })
                            all_ids.append(chunk_id)
                            all_embeddings.append(embedding)
                            doc_chunks += 1
                    
                    chunk_count += doc_chunks
                    processed_docs += 1
                    
                    if processed_docs % 10 == 0:
                        logger.info(f"Processed {processed_docs}/{len(docs)} docs, {chunk_count} chunks so far...")
                        
                except Exception as e:
                    logger.error(f"Error processing {doc_name}: {e}")
        
        # Add all chunks to ChromaDB in batches
        logger.info(f"Adding {chunk_count} chunks to ChromaDB...")
        batch_size = 100
        
        for i in range(0, len(all_texts), batch_size):
            batch_end = min(i + batch_size, len(all_texts))
            batch_num = i//batch_size + 1
            total_batches = (len(all_texts)-1)//batch_size + 1
            
            logger.info(f"Adding batch {batch_num}/{total_batches}...")
            
            try:
                self.collection.add(
                    documents=all_texts[i:batch_end],
                    embeddings=all_embeddings[i:batch_end],
                    metadatas=all_metadatas[i:batch_end],
                    ids=all_ids[i:batch_end]
                )
            except Exception as e:
                logger.error(f"Error adding batch {batch_num}: {e}")
        
        logger.info(f"Successfully loaded {chunk_count} medical chunks from {processed_docs} documents!")
        return chunk_count
    
    def search(self, query: str, limit: int = 5) -> SearchResults:
        """
        Search for relevant documents using Ollama embeddings.
        
        Args:
            query: Search query text
            limit: Maximum number of results
            
        Returns:
            SearchResults object with results and metadata
        """
        import time
        start_time = time.time()
        
        try:
            # Get query embedding from Ollama
            query_embedding = get_ollama_embedding(
                query,
                self.ollama_url,
                self.embedding_model
            )
            
            # Search ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit
            )
            
            # Convert to SearchResult objects
            search_results = []
            if results['documents'][0]:  # Check if we have results
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0], 
                    results['distances'][0]
                )):
                    search_results.append(SearchResult(
                        chunk_id=results['ids'][0][i],
                        doc_id=metadata.get('doc_id', metadata.get('document', 'unknown')),
                        content=doc,
                        score=1.0 - distance,  # Convert distance to similarity score
                        document_name=metadata.get('document', 'unknown'),
                        section=metadata.get('section', 'unknown'),
                        chunk_type=metadata.get('chunk_type', 'text')
                    ))
            
            search_time = time.time() - start_time
            
            return SearchResults(
                results=search_results,
                query=query,
                total_results=len(search_results),
                search_time=search_time,
                model_used=self.embedding_model
            )
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return SearchResults(
                results=[],
                query=query,
                total_results=0,
                search_time=time.time() - start_time,
                model_used=self.embedding_model
            )
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the current ChromaDB collection."""
        try:
            count = self.collection.count()
            return {
                'total_documents': count,
                'collection_name': CHROMA_COLLECTION_NAME,
                'embedding_model': self.embedding_model,
                'ollama_url': self.ollama_url
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}
