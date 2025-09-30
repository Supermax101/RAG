#!/usr/bin/env python3
"""
Test Hybrid RAG (Vector + Graph)
"""

import asyncio
from src.rag.infrastructure.embeddings.ollama_embeddings import OllamaEmbeddingProvider
from src.rag.infrastructure.vector_stores.chroma_store import ChromaVectorStore
from src.rag.infrastructure.llm_providers.ollama_provider import OllamaLLMProvider
from src.rag.core.services.hybrid_rag_service import HybridRAGService
from src.rag.core.models.documents import RAGQuery


async def test_hybrid_rag():
    """Test hybrid retrieval"""
    
    # Initialize providers
    embedding_provider = OllamaEmbeddingProvider()
    vector_store = ChromaVectorStore()
    llm_provider = OllamaLLMProvider(default_model="mistral-nemo:latest")
    
    # Create hybrid RAG service (with Neo4j)
    rag_service = HybridRAGService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        llm_provider=llm_provider,
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="yourpassword"
    )
    
    # Test question
    query = RAGQuery(
        question="What are the complications of TPN in neonates?",
        search_limit=5
    )
    
    print("🔍 Testing Hybrid RAG (Vector + Graph)")
    print(f"Question: {query.question}")
    print("-" * 60)
    
    # Ask with hybrid retrieval
    response = await rag_service.ask(query)
    
    print(f"\n💡 Answer:")
    print(response.answer)
    
    print(f"\n📚 Vector Sources: {len(response.sources)}")
    for i, source in enumerate(response.sources[:3], 1):
        print(f"  {i}. {source.document_name[:50]}...")
    
    if hasattr(response, 'metadata') and response.metadata:
        print(f"\n🔗 Graph Entities:")
        print(response.metadata.get('graph_entities', 'None'))
    
    print(f"\n⏱️  Total time: {response.total_time_ms:.0f}ms")
    
    # Close connections
    rag_service.close()


if __name__ == "__main__":
    asyncio.run(test_hybrid_rag())
