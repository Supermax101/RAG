"""
Ollama embedding provider implementation.
"""
import asyncio
import httpx
from typing import List, Dict, Any
from ...core.interfaces.embeddings import EmbeddingProvider
from ...config.settings import settings


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama-based embedding provider."""
    
    def __init__(self, model: str = None, base_url: str = None):
        self.model = model or settings.ollama_embed_model
        self.base_url = (base_url or settings.ollama_base_url).rstrip('/')
        self._dimension = None
        
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        async with httpx.AsyncClient(timeout=120.0) as client:  # Increased for larger models like embeddinggemma
            tasks = [self._embed_single(client, text) for text in texts]
            embeddings = await asyncio.gather(*tasks)
            return embeddings
    
    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        async with httpx.AsyncClient(timeout=120.0) as client:  # Increased for larger models like embeddinggemma
            return await self._embed_single(client, query)
    
    async def _embed_single(self, client: httpx.AsyncClient, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            response = await client.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                }
            )
            response.raise_for_status()
            result = response.json()
            embedding = result.get("embedding", [])
            
            # Cache dimension on first call
            if self._dimension is None and embedding:
                self._dimension = len(embedding)
                
            return embedding
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {e}")
    
    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self.model
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        if self._dimension is None:
            raise RuntimeError("Dimension unknown - generate at least one embedding first")
        return self._dimension
