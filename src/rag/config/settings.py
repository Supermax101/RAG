"""
Modern configuration management using Pydantic Settings.
"""
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Configuration
    mistral_api_key: Optional[str] = Field(default=None, alias="MISTRAL_API_KEY")
    mistral_base_url: str = Field(default="https://api.mistral.ai", alias="MISTRAL_BASE_URL")
    embed_model: str = Field(default="mistral-embed", alias="EMBED_MODEL")
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    
    # xAI Configuration
    xai_api_key: Optional[str] = Field(default=None, alias="XAI_API_KEY")
    xai_base_url: str = Field(default="https://api.x.ai/v1", alias="XAI_BASE_URL")
    
    # Google Gemini Configuration
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    gemini_base_url: str = Field(default="https://generativelanguage.googleapis.com/v1beta", alias="GEMINI_BASE_URL")
    
    # Ollama Configuration
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_embed_model: str = Field(default="nomic-embed-text", alias="OLLAMA_EMBED_MODEL")
    ollama_llm_model: str = Field(default="mistral:7b", alias="OLLAMA_LLM_MODEL")
    
    # ChromaDB Configuration
    chroma_collection_name: str = Field(default="tpn_documents", alias="CHROMA_COLLECTION_NAME")
    
    # RAG Configuration
    default_search_limit: int = Field(default=5, alias="DEFAULT_SEARCH_LIMIT")
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, alias="CHUNK_OVERLAP")
    
    # Performance
    max_concurrent_requests: int = Field(default=10, alias="MAX_CONCURRENT_REQUESTS")
    cache_ttl_seconds: int = Field(default=3600, alias="CACHE_TTL_SECONDS")
    
    # API Server
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables
    
    @property
    def project_root(self) -> Path:
        """Get project root directory."""
        return Path(__file__).parents[3]
    
    @property
    def data_dir(self) -> Path:
        """Get data directory."""
        return self.project_root / "data"
    
    @property
    def parsed_dir(self) -> Path:
        """Get parsed documents directory."""
        return self.data_dir / "parsed"
    
    @property
    def embeddings_dir(self) -> Path:
        """Get embeddings directory."""
        return self.data_dir / "embeddings"
    
    @property
    def chromadb_dir(self) -> Path:
        """Get ChromaDB directory."""
        return self.embeddings_dir / "chromadb"
    
    @property
    def metadata_dir(self) -> Path:
        """Get metadata directory."""
        return self.embeddings_dir / "metadata"
    
    @property
    def logs_dir(self) -> Path:
        """Get logs directory."""
        return self.project_root / "logs"
        
    def ensure_directories(self) -> None:
        """Create necessary directories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.parsed_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.chromadb_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()