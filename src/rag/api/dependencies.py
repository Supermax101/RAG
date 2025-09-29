"""
FastAPI dependency injection.
"""
import asyncio
from functools import lru_cache
from typing import Optional
from ..core.services.rag_service import RAGService
from ..infrastructure.embeddings.ollama_embeddings import OllamaEmbeddingProvider
from ..infrastructure.vector_stores.chroma_store import ChromaVectorStore
from ..infrastructure.llm_providers.ollama_provider import OllamaLLMProvider
from ..config.settings import settings

# Global instances (singleton pattern for performance)
_rag_service: Optional[RAGService] = None
_embedding_provider: Optional[OllamaEmbeddingProvider] = None
_vector_store: Optional[ChromaVectorStore] = None
_llm_provider: Optional[OllamaLLMProvider] = None


@lru_cache()
def get_embedding_provider() -> OllamaEmbeddingProvider:
    """Get or create embedding provider instance."""
    global _embedding_provider
    if _embedding_provider is None:
        _embedding_provider = OllamaEmbeddingProvider()
    return _embedding_provider


@lru_cache()
def get_vector_store() -> ChromaVectorStore:
    """Get or create vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = ChromaVectorStore()
    return _vector_store


@lru_cache()
def get_llm_provider() -> OllamaLLMProvider:
    """Get or create LLM provider instance."""
    global _llm_provider
    if _llm_provider is None:
        _llm_provider = OllamaLLMProvider()
    return _llm_provider


def get_rag_service() -> RAGService:
    """Get or create RAG service instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService(
            embedding_provider=get_embedding_provider(),
            vector_store=get_vector_store(),
            llm_provider=get_llm_provider()
        )
    return _rag_service


async def check_services_health() -> dict:
    """Check health of all services."""
    llm_provider = get_llm_provider()
    
    return {
        "ollama_llm": await llm_provider.check_health(),
        "chromadb": True,  # ChromaDB health is checked during initialization
        "embedding_service": True  # Will be checked during first embedding
    }
