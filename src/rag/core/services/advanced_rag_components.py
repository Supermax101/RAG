"""
Advanced RAG Components following LangChain Best Practices (2025)

Features:
1. Reranking (Cohere/Jina)
2. Context Compression (LLMChainExtractor)
3. Query Decomposition (MultiQueryRetriever pattern)
4. Validation & Polish
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# LangChain imports for advanced RAG features
try:
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain_community.document_compressors import CohereRerank, JinaRerank, LLMChainExtractor
    from langchain_community.document_compressors import EmbeddingsFilter
    from langchain.retrievers.multi_query import MultiQueryRetriever
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_ADVANCED_AVAILABLE = True
except ImportError:
    print("⚠️  Advanced LangChain components not available. Install: pip install langchain langchain-community")
    LANGCHAIN_ADVANCED_AVAILABLE = False
    ContextualCompressionRetriever = None
    CohereRerank = None
    JinaRerank = None
    LLMChainExtractor = None
    EmbeddingsFilter = None
    MultiQueryRetriever = None
    PromptTemplate = None
    StrOutputParser = None


class RerankingConfig(BaseModel):
    """Configuration for reranking"""
    enabled: bool = Field(default=False, description="Enable reranking")
    provider: str = Field(default="cohere", description="Reranking provider: cohere, jina, or embeddings")
    top_n: int = Field(default=15, description="Number of results to return after reranking")
    initial_k: int = Field(default=50, description="Number of initial results to retrieve before reranking")
    cohere_api_key: Optional[str] = Field(default=None, description="Cohere API key")
    jina_api_key: Optional[str] = Field(default=None, description="Jina API key")


class CompressionConfig(BaseModel):
    """Configuration for context compression"""
    enabled: bool = Field(default=False, description="Enable context compression")
    method: str = Field(default="llm", description="Compression method: llm, embeddings")
    similarity_threshold: float = Field(default=0.76, description="Threshold for embeddings filter")


class QueryDecompositionConfig(BaseModel):
    """Configuration for query decomposition"""
    enabled: bool = Field(default=False, description="Enable query decomposition")
    num_queries: int = Field(default=3, description="Number of sub-queries to generate")
    use_fusion: bool = Field(default=True, description="Use reciprocal rank fusion for multi-query results")


class ValidationConfig(BaseModel):
    """Configuration for answer validation"""
    enabled: bool = Field(default=False, description="Enable answer validation")
    check_sources: bool = Field(default=True, description="Verify answer is supported by sources")
    polish_answer: bool = Field(default=True, description="Polish answer for clarity")


class AdvancedRAGComponents:
    """Advanced RAG components following LangChain best practices"""
    
    def __init__(
        self,
        llm_provider,
        embedding_provider,
        rerank_config: Optional[RerankingConfig] = None,
        compression_config: Optional[CompressionConfig] = None,
        decomposition_config: Optional[QueryDecompositionConfig] = None,
        validation_config: Optional[ValidationConfig] = None
    ):
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider
        
        # Configurations
        self.rerank_config = rerank_config or RerankingConfig()
        self.compression_config = compression_config or CompressionConfig()
        self.decomposition_config = decomposition_config or QueryDecompositionConfig()
        self.validation_config = validation_config or ValidationConfig()
        
        # Initialize components
        self._init_reranker()
        self._init_compressor()
        self._init_validator()
        
        print("📊 Advanced RAG Components initialized:")
        print(f"  - Reranking: {'✅' if self.rerank_config.enabled else '❌'} ({self.rerank_config.provider})")
        print(f"  - Compression: {'✅' if self.compression_config.enabled else '❌'} ({self.compression_config.method})")
        print(f"  - Query Decomposition: {'✅' if self.decomposition_config.enabled else '❌'}")
        print(f"  - Validation: {'✅' if self.validation_config.enabled else '❌'}")
    
    def _init_reranker(self):
        """Initialize reranker based on configuration (LangChain Best Practice)"""
        if not self.rerank_config.enabled or not LANGCHAIN_ADVANCED_AVAILABLE:
            self.reranker = None
            return
        
        try:
            if self.rerank_config.provider == "cohere":
                # Cohere Rerank (requires API key)
                if not self.rerank_config.cohere_api_key:
                    print("⚠️  Cohere API key not provided, skipping reranking")
                    self.reranker = None
                    return
                
                self.reranker = CohereRerank(
                    cohere_api_key=self.rerank_config.cohere_api_key,
                    top_n=self.rerank_config.top_n
                )
                print(f"  ✅ Cohere Rerank initialized (top_n={self.rerank_config.top_n})")
                
            elif self.rerank_config.provider == "jina":
                # Jina Rerank (requires API key)
                if not self.rerank_config.jina_api_key:
                    print("⚠️  Jina API key not provided, skipping reranking")
                    self.reranker = None
                    return
                
                self.reranker = JinaRerank(
                    jina_api_key=self.rerank_config.jina_api_key,
                    top_n=self.rerank_config.top_n
                )
                print(f"  ✅ Jina Rerank initialized (top_n={self.rerank_config.top_n})")
                
            elif self.rerank_config.provider == "embeddings":
                # Embeddings-based filtering (no API key needed)
                self.reranker = EmbeddingsFilter(
                    embeddings=self.embedding_provider,
                    similarity_threshold=0.76
                )
                print(f"  ✅ Embeddings Filter initialized (threshold=0.76)")
                
            else:
                print(f"⚠️  Unknown reranking provider: {self.rerank_config.provider}")
                self.reranker = None
                
        except Exception as e:
            print(f"⚠️  Failed to initialize reranker: {e}")
            self.reranker = None
    
    def _init_compressor(self):
        """Initialize context compressor (LangChain Best Practice)"""
        if not self.compression_config.enabled or not LANGCHAIN_ADVANCED_AVAILABLE:
            self.compressor = None
            return
        
        try:
            if self.compression_config.method == "llm":
                # LLM-based compression (extracts relevant parts)
                # This will be initialized when we have an LLM instance
                print("  ✅ LLM-based compression enabled")
                self.compressor = "llm"  # Placeholder
                
            elif self.compression_config.method == "embeddings":
                # Embeddings-based filtering
                self.compressor = EmbeddingsFilter(
                    embeddings=self.embedding_provider,
                    similarity_threshold=self.compression_config.similarity_threshold
                )
                print(f"  ✅ Embeddings-based compression initialized (threshold={self.compression_config.similarity_threshold})")
                
            else:
                print(f"⚠️  Unknown compression method: {self.compression_config.method}")
                self.compressor = None
                
        except Exception as e:
            print(f"⚠️  Failed to initialize compressor: {e}")
            self.compressor = None
    
    def _init_validator(self):
        """Initialize answer validator"""
        if not self.validation_config.enabled:
            self.validator = None
            return
        
        # Validator will be implemented as prompt templates
        self.validator = self._create_validation_prompts()
        print("  ✅ Answer validation prompts initialized")
    
    def _create_validation_prompts(self) -> Dict[str, PromptTemplate]:
        """Create prompts for validation and polishing"""
        
        # Validation prompt: Check if answer is supported by sources
        validation_prompt = PromptTemplate(
            template="""You are validating a medical answer for accuracy.

QUESTION: {question}

RETRIEVED SOURCES:
{sources}

PROPOSED ANSWER: {answer}

Task: Verify if the proposed answer is FULLY SUPPORTED by the retrieved sources.

Respond with:
- "VALID" if the answer is supported by the sources
- "INVALID" if the answer contradicts or is not supported by sources
- "PARTIALLY_VALID" if only parts are supported

Validation: """,
            input_variables=["question", "sources", "answer"]
        )
        
        # Polish prompt: Improve answer clarity and completeness
        polish_prompt = PromptTemplate(
            template="""You are polishing a medical answer for clarity and completeness.

QUESTION: {question}

SOURCES: {sources}

DRAFT ANSWER: {answer}

Task: Rewrite the answer to be:
1. Clinically precise
2. Well-structured
3. Cite specific sources when possible
4. Professional medical tone

Polished Answer: """,
            input_variables=["question", "sources", "answer"]
        )
        
        return {
            "validation": validation_prompt,
            "polish": polish_prompt
        }
    
    async def decompose_query(self, query: str) -> List[str]:
        """Decompose complex query into sub-queries (LangChain MultiQueryRetriever pattern)"""
        if not self.decomposition_config.enabled:
            return [query]
        
        decomposition_prompt = PromptTemplate(
            template="""You are analyzing a complex TPN (Total Parenteral Nutrition) medical question.

Original Question: {question}

Task: Break this into {num_queries} simpler sub-questions that together answer the original question.

Rules:
1. Each sub-question should focus on ONE specific aspect
2. Sub-questions should be answerable from clinical guidelines
3. Together, they should fully cover the original question

Generate {num_queries} sub-questions (one per line):
1.""",
            input_variables=["question", "num_queries"]
        )
        
        try:
            # Generate sub-queries using LLM
            prompt = decomposition_prompt.format(
                question=query,
                num_queries=self.decomposition_config.num_queries
            )
            
            response = await self.llm_provider.generate(
                prompt=prompt,
                temperature=0.3,  # Low temperature for consistent decomposition
                max_tokens=500
            )
            
            # Parse sub-queries
            lines = response.strip().split('\n')
            sub_queries = []
            for line in lines:
                # Remove numbering and clean up
                clean_line = line.strip()
                # Remove leading numbers like "1.", "2.", etc
                import re
                clean_line = re.sub(r'^\d+[\.\)]\s*', '', clean_line)
                if clean_line and len(clean_line) > 10:  # Reasonable question length
                    sub_queries.append(clean_line)
            
            # Always include original query
            if query not in sub_queries:
                sub_queries.insert(0, query)
            
            print(f"🔍 Query decomposition: {len(sub_queries)} sub-queries generated")
            for i, sq in enumerate(sub_queries[:5], 1):  # Show first 5
                print(f"   {i}. {sq[:80]}...")
            
            return sub_queries
            
        except Exception as e:
            print(f"⚠️  Query decomposition failed: {e}")
            return [query]
    
    async def validate_answer(
        self,
        question: str,
        answer: str,
        sources: List[str]
    ) -> Dict[str, Any]:
        """Validate if answer is supported by sources"""
        if not self.validation_config.enabled or not self.validation_config.check_sources:
            return {"is_valid": True, "validation_status": "SKIPPED"}
        
        try:
            sources_text = "\n\n".join(sources[:10])  # Limit to top 10 sources
            
            validation_prompt = self.validator["validation"].format(
                question=question,
                sources=sources_text,
                answer=answer
            )
            
            response = await self.llm_provider.generate(
                prompt=validation_prompt,
                temperature=0.0,  # Deterministic validation
                max_tokens=200
            )
            
            validation_status = response.strip().upper()
            
            if "VALID" in validation_status and "INVALID" not in validation_status:
                is_valid = True
            elif "PARTIALLY" in validation_status:
                is_valid = True  # Accept partial validity
            else:
                is_valid = False
            
            return {
                "is_valid": is_valid,
                "validation_status": validation_status,
                "raw_response": response
            }
            
        except Exception as e:
            print(f"⚠️  Answer validation failed: {e}")
            return {"is_valid": True, "validation_status": "ERROR", "error": str(e)}
    
    async def polish_answer(
        self,
        question: str,
        answer: str,
        sources: List[str]
    ) -> str:
        """Polish answer for clarity and completeness"""
        if not self.validation_config.enabled or not self.validation_config.polish_answer:
            return answer
        
        try:
            sources_text = "\n\n".join(sources[:10])
            
            polish_prompt = self.validator["polish"].format(
                question=question,
                sources=sources_text,
                answer=answer
            )
            
            polished = await self.llm_provider.generate(
                prompt=polish_prompt,
                temperature=0.3,  # Some creativity for polishing
                max_tokens=800
            )
            
            print("  ✨ Answer polished for clarity")
            return polished.strip()
            
        except Exception as e:
            print(f"⚠️  Answer polishing failed: {e}")
            return answer

