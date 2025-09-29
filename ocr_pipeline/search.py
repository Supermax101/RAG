"""
RAG Search functionality using ChromaDB and Mistral embeddings.

This module provides:
- ChromaDB collection management
- Vector similarity search 
- Multimodal search (text + images)
- Result ranking and filtering
- Integration with parsed documents and embeddings

For future integration with LLMs for RAG conversations.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

import numpy as np
import chromadb
from chromadb.config import Settings
from sklearn.metrics.pairwise import cosine_similarity

from .config import (
    CHROMADB_DIR,
    CHROMA_COLLECTION_NAME,
    PARSED_DIR,
    VECTORS_DIR,
    get_logger,
)
from .embedding_runner import (
    DocumentEmbeddings,
    load_document_embeddings,
    _call_embeddings_api,
    DEFAULT_EMBED_MODEL,
)


@dataclass
class SearchResult:
    """Single search result with metadata."""
    chunk_id: str
    doc_id: str
    content: str
    score: float
    document_name: str
    section: str
    chunk_type: str  # text, heading, table, image_caption
    page_num: Optional[int] = None
    nearby_images: List[str] = None  # SHA hashes of nearby images
    image_paths: List[str] = None  # Full paths to nearby images

    def __post_init__(self):
        if self.nearby_images is None:
            self.nearby_images = []
        if self.image_paths is None:
            self.image_paths = []


@dataclass
class SearchResults:
    """Collection of search results with metadata."""
    query: str
    results: List[SearchResult]
    total_found: int
    search_time_seconds: float
    model_used: str


logger = get_logger("search")


class ChromaRAGSearch:
    """ChromaDB-based RAG search engine."""
    
    def __init__(self, collection_name: str = CHROMA_COLLECTION_NAME):
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Initialize ChromaDB client with persistent storage (disable telemetry completely)
            import chromadb.telemetry
            chromadb.telemetry.telemetry = None  # Disable telemetry completely
            
            self.client = chromadb.PersistentClient(
                path=str(CHROMADB_DIR),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Connected to existing ChromaDB collection: {self.collection_name}")
            except Exception:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Medical document embeddings for RAG"}
                )
                logger.info(f"Created new ChromaDB collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def add_document_to_collection(self, doc_embeddings: DocumentEmbeddings) -> bool:
        """Add a document's embeddings to the ChromaDB collection."""
        try:
            if not doc_embeddings.embeddings:
                logger.warning(f"No embeddings to add for {doc_embeddings.doc_id}")
                return False
            
            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            metadatas = []
            documents = []
            
            for chunk, embedding in zip(doc_embeddings.chunks, doc_embeddings.embeddings):
                ids.append(chunk.chunk_id)
                embeddings.append(embedding.embedding)
                documents.append(chunk.content)
                
                # Create metadata for this chunk
                metadata = {
                    "doc_id": chunk.doc_id,
                    "document_name": doc_embeddings.original_filename,
                    "section": chunk.section or "",
                    "chunk_type": chunk.chunk_type,
                    "chunk_index": chunk.chunk_index,
                }
                
                # Add optional fields if they exist
                if chunk.page_num is not None:
                    metadata["page_num"] = chunk.page_num
                if chunk.line_start is not None:
                    metadata["line_start"] = chunk.line_start
                if chunk.nearby_images:
                    metadata["nearby_images"] = ",".join(chunk.nearby_images)
                
                metadatas.append(metadata)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            
            logger.info(f"Added {len(embeddings)} chunks from {doc_embeddings.original_filename} to ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document {doc_embeddings.doc_id} to ChromaDB: {e}")
            return False
    
    def search_similar(
        self, 
        query: str, 
        limit: int = 10, 
        model: str = DEFAULT_EMBED_MODEL,
        filter_metadata: Optional[Dict] = None
    ) -> SearchResults:
        """Search for similar chunks using vector similarity."""
        import time
        start_time = time.time()
        
        try:
            # Get query embedding
            query_embeddings, _, _ = _call_embeddings_api([query], model=model)
            query_embedding = query_embeddings[0]
            
            # Search in ChromaDB
            search_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=filter_metadata,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Convert ChromaDB results to our SearchResult objects
            results = []
            
            if search_results['ids'] and search_results['ids'][0]:
                for i, chunk_id in enumerate(search_results['ids'][0]):
                    metadata = search_results['metadatas'][0][i]
                    content = search_results['documents'][0][i]
                    distance = search_results['distances'][0][i]
                    
                    # Convert distance to similarity score (0-1, higher is better)
                    score = max(0, 1 - distance)
                    
                    # Get nearby image paths
                    nearby_images = metadata.get('nearby_images', '').split(',') if metadata.get('nearby_images') else []
                    image_paths = self._resolve_image_paths(metadata.get('doc_id', ''), nearby_images)
                    
                    result = SearchResult(
                        chunk_id=chunk_id,
                        doc_id=metadata.get('doc_id', ''),
                        content=content,
                        score=score,
                        document_name=metadata.get('document_name', ''),
                        section=metadata.get('section', ''),
                        chunk_type=metadata.get('chunk_type', 'text'),
                        page_num=metadata.get('page_num'),
                        nearby_images=nearby_images,
                        image_paths=image_paths
                    )
                    results.append(result)
            
            search_time = time.time() - start_time
            
            return SearchResults(
                query=query,
                results=results,
                total_found=len(results),
                search_time_seconds=search_time,
                model_used=model
            )
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return SearchResults(
                query=query,
                results=[],
                total_found=0,
                search_time_seconds=time.time() - start_time,
                model_used=model
            )
    
    def _resolve_image_paths(self, doc_id: str, image_hashes: List[str]) -> List[str]:
        """Resolve image SHA hashes to full file paths."""
        image_paths = []
        
        # Find the document folder
        for folder in PARSED_DIR.iterdir():
            if folder.is_dir():
                # Check if this folder contains our document
                index_files = list(folder.glob(f"{doc_id}.index.json"))
                if index_files:
                    images_dir = folder / "images"
                    if images_dir.exists():
                        for img_hash in image_hashes:
                            if img_hash.strip():
                                img_path = images_dir / f"{img_hash}.png"
                                if img_path.exists():
                                    image_paths.append(str(img_path))
                    break
        
        return image_paths
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the ChromaDB collection."""
        try:
            count = self.collection.count()
            
            # Get ALL metadata to get accurate document counts
            all_results = self.collection.get(include=['metadatas'])
            
            doc_counts = {}
            chunk_types = {}
            unique_documents = set()
            
            if all_results['metadatas']:
                for metadata in all_results['metadatas']:
                    if metadata:
                        doc_name = metadata.get('document_name', 'unknown')
                        chunk_type = metadata.get('chunk_type', 'text')
                        
                        doc_counts[doc_name] = doc_counts.get(doc_name, 0) + 1
                        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                        unique_documents.add(doc_name)
            
            return {
                "total_chunks": count,
                "total_documents": len(unique_documents),  # Accurate count
                "documents": dict(sorted(doc_counts.items())),
                "chunk_types": chunk_types,
                "collection_name": self.collection_name
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}


def load_all_embeddings_to_chromadb(force: bool = False) -> Tuple[int, int]:
    """Load all available embeddings into ChromaDB with strict duplicate prevention."""
    search_engine = ChromaRAGSearch()
    
    # Get existing chunk IDs and document IDs to avoid duplicates
    existing_chunk_ids = set()
    existing_doc_ids = set()
    
    if not force:
        try:
            # Get all existing data to check for duplicates
            all_results = search_engine.collection.get(include=['metadatas'])
            if all_results['ids']:
                existing_chunk_ids = set(all_results['ids'])
                # Extract document IDs from metadata
                for metadata in all_results['metadatas']:
                    if metadata and 'doc_id' in metadata:
                        existing_doc_ids.add(metadata['doc_id'])
            
            logger.info(f"Found {len(existing_chunk_ids)} existing chunks from {len(existing_doc_ids)} documents in ChromaDB")
        except Exception as e:
            logger.info(f"ChromaDB collection is empty or doesn't exist yet: {e}")
            pass
    
    # Load all embedding files
    embedding_files = list(VECTORS_DIR.glob("*.embeddings.json"))
    loaded_count = 0
    skipped_count = 0
    
    logger.info(f"Processing {len(embedding_files)} embedding files...")
    
    for embedding_file in embedding_files:
        doc_id = embedding_file.stem.replace('.embeddings', '')
        
        try:
            # Check if this document is already loaded (unless force=True)
            if not force and doc_id in existing_doc_ids:
                logger.info(f"⏭️  Skipping {doc_id} - already in ChromaDB")
                skipped_count += 1
                continue
            
            doc_embeddings = load_document_embeddings(doc_id)
            if not doc_embeddings:
                logger.warning(f"Could not load embeddings for {doc_id}")
                continue
            
            # Double-check individual chunks (extra safety)
            doc_chunk_ids = [chunk.chunk_id for chunk in doc_embeddings.chunks]
            if not force and any(chunk_id in existing_chunk_ids for chunk_id in doc_chunk_ids):
                logger.warning(f"⚠️  Document {doc_embeddings.original_filename} has chunks already in ChromaDB - skipping")
                skipped_count += 1
                continue
            
            # Add to ChromaDB
            logger.info(f"📥 Loading {doc_embeddings.original_filename} ({len(doc_embeddings.chunks)} chunks)")
            if search_engine.add_document_to_collection(doc_embeddings):
                loaded_count += 1
                logger.info(f"✅ Loaded {doc_embeddings.original_filename} into ChromaDB")
                # Update our tracking sets
                existing_doc_ids.add(doc_id)
                existing_chunk_ids.update(doc_chunk_ids)
            else:
                logger.error(f"❌ Failed to load {doc_embeddings.original_filename}")
            
        except Exception as e:
            logger.error(f"Failed to load {embedding_file}: {e}")
            continue
    
    return loaded_count, skipped_count


def search_documents(
    query: str, 
    limit: int = 5, 
    show_images: bool = True,
    model: str = DEFAULT_EMBED_MODEL
) -> SearchResults:
    """High-level search function for documents."""
    search_engine = ChromaRAGSearch()
    return search_engine.search_similar(query, limit=limit, model=model)


def print_search_results(results: SearchResults, show_content: bool = True, show_images: bool = True):
    """Pretty print search results."""
    print(f"\n🔍 Search Query: '{results.query}'")
    print(f"📊 Found {results.total_found} results in {results.search_time_seconds:.2f}s using {results.model_used}")
    print("=" * 60)
    
    for i, result in enumerate(results.results, 1):
        print(f"\n{i}. {result.document_name} (Score: {result.score:.3f})")
        print(f"   📄 Section: {result.section}")
        print(f"   🏷️  Type: {result.chunk_type}")
        
        if result.page_num:
            print(f"   📖 Page: {result.page_num}")
        
        if show_content:
            content_preview = result.content[:200] + "..." if len(result.content) > 200 else result.content
            print(f"   📝 Content: {content_preview}")
        
        if show_images and result.image_paths:
            print(f"   🖼️  Images: {len(result.image_paths)} nearby")
            for img_path in result.image_paths[:2]:  # Show first 2 images
                print(f"      • {Path(img_path).name}")
        
        print("-" * 40)
