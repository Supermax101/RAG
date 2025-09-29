"""
Abstract interfaces for embedding providers.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..models.documents import DocumentChunk


class EmbeddingProvider(ABC):
    """Abstract interface for embedding providers."""
    
    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass
    
    @abstractmethod
    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass


class VectorStore(ABC):
    """Abstract interface for vector storage systems."""
    
    @abstractmethod
    async def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
        doc_name: str
    ) -> None:
        """Add document chunks with their embeddings."""
        pass
    
    @abstractmethod
    async def search_similar(
        self,
        query_embedding: List[float],
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks."""
        pass
    
    @abstractmethod
    async def delete_document(self, doc_id: str) -> None:
        """Delete all chunks for a document."""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        pass


class LLMProvider(ABC):
    """Abstract interface for LLM providers."""
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 500
    ) -> str:
        """Generate text response."""
        pass
    
    @property
    @abstractmethod
    def available_models(self) -> List[str]:
        """Return list of available models."""
        pass
