"""
Ollama LLM provider implementation.
"""
import httpx
from typing import List, Optional
from ...core.interfaces.embeddings import LLMProvider
from ...config.settings import settings


class OllamaLLMProvider(LLMProvider):
    """Ollama-based LLM provider."""
    
    def __init__(self, base_url: str = None, default_model: str = "mistral:7b"):
        self.base_url = (base_url or settings.ollama_base_url).rstrip('/')
        self.default_model = default_model
        self._available_models = None
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 500
    ) -> str:
        """Generate text response using Ollama."""
        model_name = model or self.default_model
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "num_predict": max_tokens,
                            "temperature": temperature,
                            "top_p": 0.9,
                            "repeat_penalty": 1.1
                        }
                    }
                )
                response.raise_for_status()
                result = response.json()
                
                generated_text = result.get("response", "").strip()
                if not generated_text:
                    raise RuntimeError("Empty response from Ollama")
                
                return generated_text
                
            except httpx.TimeoutException:
                raise RuntimeError("Request to Ollama timed out")
            except Exception as e:
                raise RuntimeError(f"Failed to generate text with Ollama: {e}")
    
    @property
    async def available_models(self) -> List[str]:
        """Return list of available Ollama models."""
        if self._available_models is not None:
            return self._available_models
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
                
                models = []
                for model in data.get("models", []):
                    name = model.get("name", "")
                    if name:
                        models.append(name)
                
                self._available_models = models
                return models
                
            except Exception:
                # Return fallback list if API fails
                return [self.default_model, "mistral:7b", "llama3:8b"]
    
    async def check_health(self) -> bool:
        """Check if Ollama service is healthy."""
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                response = await client.get(f"{self.base_url}/api/version")
                return response.status_code == 200
            except Exception:
                return False
