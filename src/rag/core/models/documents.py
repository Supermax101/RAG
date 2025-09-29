"""
Core data models for document processing and RAG operations.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """A chunk of document content with metadata."""
    
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    doc_id: str = Field(..., description="Parent document identifier")
    content: str = Field(..., description="Text content of the chunk")
    chunk_type: str = Field(default="text", description="Type of chunk (text, heading, table, etc.)")
    page_num: Optional[int] = Field(default=None, description="Page number if applicable")
    section: Optional[str] = Field(default=None, description="Document section name")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        frozen = True


class SearchResult(BaseModel):
    """A search result with relevance score."""
    
    chunk: DocumentChunk
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    document_name: str = Field(..., description="Human-readable document name")
    
    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id
    
    @property
    def content(self) -> str:
        return self.chunk.content
    
    class Config:
        frozen = True


class SearchQuery(BaseModel):
    """A search query with parameters."""
    
    query: str = Field(..., min_length=1, description="Search query text")
    limit: int = Field(default=5, ge=1, le=50, description="Maximum number of results")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Search filters")
    
    class Config:
        frozen = True


class SearchResponse(BaseModel):
    """Response from search operation."""
    
    query: SearchQuery
    results: List[SearchResult] = Field(default_factory=list)
    total_results: int = Field(ge=0)
    search_time_ms: float = Field(ge=0)
    model_used: Optional[str] = Field(default=None)
    
    class Config:
        frozen = True


class RAGQuery(BaseModel):
    """A RAG query combining search and generation."""
    
    question: str = Field(..., min_length=1, description="Question to answer")
    search_limit: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")
    model: Optional[str] = Field(default=None, description="LLM model to use")
    temperature: float = Field(default=0.1, ge=0.0, le=1.0, description="Generation temperature")
    
    class Config:
        frozen = True


class RAGResponse(BaseModel):
    """Response from RAG operation."""
    
    question: str
    answer: str
    sources: List[SearchResult] = Field(default_factory=list)
    search_time_ms: float = Field(ge=0)
    generation_time_ms: float = Field(ge=0) 
    total_time_ms: float = Field(ge=0)
    model_used: str
    
    class Config:
        frozen = True


@dataclass
class DocumentEmbeddings:
    """Document embeddings with metadata (preserved from OCR pipeline)."""
    
    doc_id: str
    chunks: List[DocumentChunk] = field(default_factory=list)
    embeddings: List[List[float]] = field(default_factory=list)
    total_tokens: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    model_used: str = "unknown"
    
    def __post_init__(self):
        if len(self.chunks) != len(self.embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
