"""
OpenAI LLM provider implementation.
"""
from typing import List, Optional
from openai import AsyncOpenAI
from ...core.interfaces.embeddings import LLMProvider
from ...config.settings import settings


class OpenAILLMProvider(LLMProvider):
    """OpenAI-based LLM provider (GPT-4, GPT-5, O1, O3 reasoning models, etc.)."""
    
    def __init__(self, api_key: Optional[str] = None, default_model: str = "gpt-4o"):
        """Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key (falls back to settings/env)
            default_model: Default model to use
        """
        self.api_key = api_key or settings.openai_api_key
        self.default_model = default_model
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = AsyncOpenAI(api_key=self.api_key)
        self._available_models = None
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 500,
        seed: Optional[int] = None
    ) -> str:
        """Generate text response using OpenAI."""
        model_name = model or self.default_model
        
        try:
            # Detect reasoning models (GPT-5, O1, O3) that use max_completion_tokens
            is_reasoning_model = any(x in model_name.lower() for x in ['gpt-5', 'o1', 'o3'])
            
            kwargs = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "timeout": 60.0,
            }
            
            # Reasoning models (GPT-5, O1, O3) don't support temperature, frequency_penalty, presence_penalty
            # and use max_completion_tokens instead of max_tokens
            if is_reasoning_model:
                kwargs["max_completion_tokens"] = max_tokens
            else:
                kwargs["temperature"] = temperature
                kwargs["max_tokens"] = max_tokens
                kwargs["frequency_penalty"] = 0.3  # Discourage repetitive phrases
                kwargs["presence_penalty"] = 0.1    # Encourage diverse vocabulary
            
            # Add seed for reproducibility if provided (supported by all models)
            if seed is not None:
                kwargs["seed"] = seed
            
            response = await self.client.chat.completions.create(**kwargs)
            
            # Get content, handling potential None for reasoning models
            content = response.choices[0].message.content
            return content.strip() if content else ""
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate text with OpenAI: {e}")
    
    @property
    async def available_models(self) -> List[str]:
        """Return list of available OpenAI models from API."""
        if self._available_models is not None:
            return self._available_models
        
        try:
            # Get models from OpenAI API
            models_response = await self.client.models.list()
            
            # Filter for ONLY GPT-5 and GPT-5 mini (latest reasoning models)
            chat_models = [
                model.id for model in models_response.data
                if ('gpt-5' in model.id.lower() and 'mini' in model.id.lower()) or 
                   model.id.lower() in ['gpt-5', 'gpt-5-2025-08-07', 'gpt-5-chat-latest']
            ]
            
            self._available_models = sorted(chat_models)
            return self._available_models
            
        except Exception as e:
            # If API call fails, return empty list
            print(f"Warning: Could not fetch OpenAI models: {e}")
            return []
    
    async def check_health(self) -> bool:
        """Check if OpenAI API is accessible."""
        try:
            # Try to list models as a health check
            await self.client.models.list()
            return True
        except Exception:
            return False
