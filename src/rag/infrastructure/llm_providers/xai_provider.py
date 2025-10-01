"""
xAI (Grok) LLM provider implementation.
"""
from typing import List, Optional
from openai import AsyncOpenAI
from ...core.interfaces.embeddings import LLMProvider
from ...config.settings import settings


class XAILLMProvider(LLMProvider):
    """xAI-based LLM provider (Grok models)."""
    
    def __init__(self, api_key: Optional[str] = None, default_model: str = "grok-4-fast-reasoning"):
        """Initialize xAI provider.
        
        Args:
            api_key: xAI API key (falls back to settings/env)
            default_model: Default model to use
        """
        self.api_key = api_key or settings.xai_api_key
        self.default_model = default_model
        
        if not self.api_key:
            raise ValueError(
                "xAI API key not found. Set XAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # xAI uses OpenAI-compatible API
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.x.ai/v1"
        )
        self._available_models = None
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 500,
        seed: Optional[int] = None
    ) -> str:
        """Generate text response using xAI."""
        model_name = model or self.default_model
        
        try:
            kwargs = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": 60.0,
                "frequency_penalty": 0.3,  # Discourage repetitive phrases
                "presence_penalty": 0.1    # Encourage diverse vocabulary
            }
            
            # Add seed for reproducibility if provided
            if seed is not None:
                kwargs["seed"] = seed
            
            response = await self.client.chat.completions.create(**kwargs)
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate text with xAI: {e}")
    
    @property
    async def available_models(self) -> List[str]:
        """Return list of available xAI models from API."""
        if self._available_models is not None:
            return self._available_models
        
        try:
            # Get models from xAI API
            models_response = await self.client.models.list()
            
            # Filter for only grok-4-fast-reasoning
            models = [
                model.id for model in models_response.data
                if model.id == "grok-4-fast-reasoning"
            ]
            
            self._available_models = models
            return self._available_models
            
        except Exception as e:
            # If API call fails, return empty list
            print(f"Warning: Could not fetch xAI models: {e}")
            return []
    
    async def check_health(self) -> bool:
        """Check if xAI API is accessible."""
        try:
            # Try to list models as a health check
            await self.client.models.list()
            return True
        except Exception:
            return False
