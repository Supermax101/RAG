"""
API request/response schemas.
"""
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Request schema for document search."""
    
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    limit: int = Field(default=5, ge=1, le=20, description="Maximum number of results")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Search filters")


class RAGRequest(BaseModel):
    """Request schema for RAG query."""
    
    question: str = Field(..., min_length=1, max_length=1000, description="Question to answer")
    search_limit: int = Field(default=5, ge=1, le=10, description="Number of documents to retrieve")
    model: Optional[str] = Field(default=None, description="LLM model to use")
    temperature: float = Field(default=0.1, ge=0.0, le=1.0, description="Generation temperature")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(default="healthy")
    version: str = Field(default="2.0.0")
    services: Dict[str, bool] = Field(default_factory=dict)


class StatsResponse(BaseModel):
    """Collection statistics response."""
    
    total_chunks: int
    total_documents: int
    collection_name: str
    embedding_model: str
