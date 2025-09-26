"""
Ollama Embeddings Integration - Drop-in replacement for Mistral API

This module provides Ollama-compatible embedding functions that can replace
Mistral API calls while maintaining the same interface.
"""

import requests
import json
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class OllamaEmbeddings:
    """Ollama embeddings client that mimics Mistral API interface."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for a list of texts using Ollama.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (lists of floats)
        """
        embeddings = []
        
        for text in texts:
            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={
                        "model": self.model,
                        "prompt": text
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    embeddings.append(result["embedding"])
                else:
                    logger.error(f"Ollama embedding failed: {response.status_code} - {response.text}")
                    # Return a zero vector as fallback
                    embeddings.append([0.0] * 768)  # Default dimension for nomic-embed-text
                    
            except Exception as e:
                logger.error(f"Error getting embedding from Ollama: {e}")
                # Return a zero vector as fallback
                embeddings.append([0.0] * 768)
        
        return embeddings


def get_ollama_embedding(text: str, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text") -> List[float]:
    """
    Get a single embedding from Ollama (compatible with existing code).
    
    Args:
        text: Text to embed
        base_url: Ollama server URL
        model: Embedding model name
        
    Returns:
        List of floats representing the embedding vector
    """
    try:
        response = requests.post(
            f"{base_url}/api/embeddings",
            json={
                "model": model,
                "prompt": text
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["embedding"]
        else:
            logger.error(f"Ollama embedding failed: {response.status_code} - {response.text}")
            return [0.0] * 768  # Default fallback
            
    except Exception as e:
        logger.error(f"Error getting embedding from Ollama: {e}")
        return [0.0] * 768  # Default fallback


def create_ollama_embeddings_for_document(doc_name: str, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text") -> Optional[Dict]:
    """
    Create embeddings for a document using Ollama (replacement for Mistral version).
    
    Args:
        doc_name: Name of the document directory
        base_url: Ollama server URL  
        model: Embedding model name
        
    Returns:
        Dict with embedding results or None if failed
    """
    from .config import PARSED_DIR
    import hashlib
    
    doc_dir = PARSED_DIR / doc_name
    md_file = doc_dir / f"{doc_name}.md"
    
    if not md_file.exists():
        logger.error(f"Markdown file not found: {md_file}")
        return None
        
    try:
        # Read the markdown content
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Split into chunks
        chunks = []
        chunk_size = 1000
        for i in range(0, len(content), chunk_size):
            chunk_text = content[i:i+chunk_size].strip()
            if chunk_text and len(chunk_text) > 50:
                chunk_id = f"{doc_name}_{i//chunk_size}_{hashlib.md5(chunk_text.encode()).hexdigest()[:8]}"
                
                # Get embedding from Ollama
                embedding = get_ollama_embedding(chunk_text, base_url, model)
                
                chunks.append({
                    'chunk_id': chunk_id,
                    'content': chunk_text,
                    'embedding': embedding,
                    'document': doc_name,
                    'chunk_index': i//chunk_size
                })
        
        logger.info(f"Created {len(chunks)} embeddings for {doc_name} using Ollama")
        return {
            'document': doc_name,
            'chunks': chunks,
            'model': model,
            'total_chunks': len(chunks)
        }
        
    except Exception as e:
        logger.error(f"Error creating Ollama embeddings for {doc_name}: {e}")
        return None
