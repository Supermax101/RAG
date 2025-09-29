"""
Ollama embedding provider implementation with robust error handling.
"""
import asyncio
import httpx
from typing import List, Dict, Any, Optional
from ...core.interfaces.embeddings import EmbeddingProvider
from ...config.settings import settings


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama-based embedding provider with concurrency control and retry logic."""
    
    def __init__(self, model: str = None, base_url: str = None, max_concurrent: int = 10):
        self.model = model or settings.ollama_embed_model
        self.base_url = (base_url or settings.ollama_base_url).rstrip('/')
        self._dimension = None
        self.max_concurrent = max_concurrent  # Limit concurrent requests
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts with concurrency control."""
        async with httpx.AsyncClient(timeout=180.0) as client:  # 3 min timeout for retry logic
            # Use semaphore to limit concurrent requests (10 at a time, not 50!)
            tasks = [self._embed_with_semaphore(client, text, idx, len(texts)) for idx, text in enumerate(texts)]
            embeddings = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any failed embeddings
            processed_embeddings = []
            failed_count = 0
            for i, emb in enumerate(embeddings):
                if isinstance(emb, Exception):
                    failed_count += 1
                    # Use zero vector as fallback
                    if self._dimension:
                        processed_embeddings.append([0.0] * self._dimension)
                    else:
                        # Try to get dimension from first successful embedding
                        processed_embeddings.append(None)  # Will handle later
                else:
                    processed_embeddings.append(emb)
            
            # Fill in any None values with zero vectors
            if self._dimension:
                processed_embeddings = [
                    emb if emb is not None else [0.0] * self._dimension 
                    for emb in processed_embeddings
                ]
            
            if failed_count > 0:
                print(f"    WARNING: {failed_count}/{len(texts)} embeddings failed, using zero vectors")
            
            return processed_embeddings
    
    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        async with httpx.AsyncClient(timeout=180.0) as client:
            return await self._embed_with_retry(client, query, max_retries=3)
    
    async def _embed_with_semaphore(self, client: httpx.AsyncClient, text: str, idx: int, total: int) -> List[float]:
        """Generate embedding with semaphore-based concurrency control."""
        async with self.semaphore:
            # Show progress for every 10th embedding
            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"    Embedding {idx+1}/{total}...")
            return await self._embed_with_retry(client, text, max_retries=3)
    
    async def _embed_with_retry(self, client: httpx.AsyncClient, text: str, max_retries: int = 3) -> List[float]:
        """Generate embedding with exponential backoff retry logic."""
        last_error = None
        
        for attempt in range(max_retries):
            try:
                return await self._embed_single(client, text)
            except httpx.ReadTimeout as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    print(f"    Timeout on attempt {attempt+1}/{max_retries}, retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    print(f"    Error on attempt {attempt+1}/{max_retries}: {type(e).__name__}, retrying...")
                    await asyncio.sleep(1)
        
        # All retries failed
        raise RuntimeError(f"Failed after {max_retries} attempts: {last_error}")
    
    async def _embed_single(self, client: httpx.AsyncClient, text: str) -> List[float]:
        """Generate embedding for a single text (no retry)."""
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