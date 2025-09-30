#!/usr/bin/env python3
"""
Interactive Hybrid RAG Test (ChromaDB + Neo4j + LangChain + LangGraph)
Works exactly like main.py demo with model selection and interactive questions.
"""

import asyncio
import httpx
from src.rag.infrastructure.embeddings.ollama_embeddings import OllamaEmbeddingProvider
from src.rag.infrastructure.vector_stores.chroma_store import ChromaVectorStore
from src.rag.infrastructure.llm_providers.ollama_provider import OllamaLLMProvider
from src.rag.core.services.hybrid_rag_service import HybridRAGService
from src.rag.core.models.documents import RAGQuery


async def get_available_ollama_models():
    """Get list of available Ollama LLM models."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models_data = response.json()
                # Filter for LLM models (exclude embedding models)
                llm_models = []
                for model in models_data.get("models", []):
                    name = model["name"]
                    # Skip embedding models
                    if not any(embed in name.lower() for embed in ["embed", "embedding", "nomic"]):
                        llm_models.append(name)
                
                return sorted(llm_models)
    except Exception:
        pass
    return []


def select_ollama_model(available_models):
    """Interactive model selection."""
    if not available_models:
        print("❌ No Ollama models available. Please pull some models first:")
        print("   ollama pull mistral:7b")
        print("   ollama pull llama3:8b")
        return None
    
    print("\n🤖 Available Ollama Models:")
    for i, model in enumerate(available_models, 1):
        print(f"  {i}. {model}")
    
    print(f"  {len(available_models)+1}. mistral:7b (default)")
    
    while True:
        try:
            choice = input(f"\nSelect model (1-{len(available_models)+1}) or press Enter for default: ").strip()
            
            if not choice:
                return "mistral:7b"  # Default
            elif choice.isdigit():
                choice_num = int(choice)
                if 1 <= choice_num <= len(available_models):
                    return available_models[choice_num - 1]
                elif choice_num == len(available_models) + 1:
                    return "mistral:7b"
                else:
                    print(f"Please enter a number between 1 and {len(available_models)+1}")
        except ValueError:
            print("Please enter a valid number")


async def test_hybrid_rag():
    """Interactive Hybrid RAG test with model selection"""
    
    print("🔍 Checking available Ollama models...")
    available_models = await get_available_ollama_models()
    
    # Let user select model
    selected_model = select_ollama_model(available_models)
    if not selected_model:
        return False
    
    print(f"✅ Selected model: {selected_model}")
    
    # Initialize providers with selected model
    embedding_provider = OllamaEmbeddingProvider()
    vector_store = ChromaVectorStore()
    llm_provider = OllamaLLMProvider(default_model=selected_model)
    
    if not await llm_provider.check_health():
        print("❌ Ollama is not running. Please start Ollama:")
        print("   ollama serve")
        return False
    
    # Create hybrid RAG service (ChromaDB + Neo4j + LangChain + LangGraph)
    rag_service = HybridRAGService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        llm_provider=llm_provider,
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="medicalpass123"
    )
    
    # Check TPN documents are loaded
    stats = await rag_service.get_collection_stats()
    if stats["total_chunks"] == 0:
        print("❌ No TPN documents found. Please run 'python main.py init' first.")
        return False
    
    print(f"\n🏥 TPN Hybrid RAG Test - Interactive Demo (Using {selected_model})")
    print("=" * 65)
    print(f"📊 Ready: {stats['total_chunks']} chunks from {stats['total_documents']} documents")
    print("🔬 System: ChromaDB Vector Search + Neo4j Graph + LangChain + LangGraph")
    print("\n💊 Ask TPN/parenteral nutrition questions or try these examples:")
    print("  • What are the complications of TPN in neonates?")
    print("  • How do you calculate protein requirements for pediatric PN?")
    print("  • What monitoring is required for TPN patients?")
    print("  • When should you transition from TPN to enteral nutrition?")
    print("\nType 'quit', 'exit', or 'q' to end the session.")
    print("-" * 65)
    
    try:
        while True:
            # Get question from user
            question = input("\n🏥 Ask your TPN question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Thank you for testing TPN Hybrid RAG!")
                break
            
            if not question:
                continue
            
            # Create query
            query = RAGQuery(question=question, search_limit=5)
            
            print(f"\n🔍 Processing: {question}")
            print("-" * 40)
            
            # Ask with hybrid retrieval (Vector + Graph + LangChain + LangGraph)
            response = await rag_service.ask(query)
            
            print(f"\n💡 **TPN Specialist Answer:**")
            print(response.answer)
            
            print(f"\n📚 **Sources Used:** ({len(response.sources)})")
            for i, source in enumerate(response.sources[:3], 1):
                section = source.content[:100].replace('\n', ' ')
                print(f"  {i}. {source.document_name[:40]}")
                print(f"     Section: {section}... (relevance: {source.score:.3f})")
            
            # Show graph enhancements if available
            if hasattr(response, 'metadata') and response.metadata:
                graph_info = response.metadata.get('graph_entities')
                if graph_info:
                    print(f"\n🔗 **Graph Knowledge Used:**")
                    print(f"  {graph_info[:200]}...")
                    print(f"  Retrieval: {response.metadata.get('retrieval_type', 'hybrid')}")
            
            print(f"\n⏱️  Response time: {response.total_time_ms:.1f}ms")
            print("-" * 65)
            
    except KeyboardInterrupt:
        print("\nThank you for testing TPN Hybrid RAG!")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Close Neo4j connection
        if hasattr(rag_service, 'close'):
            rag_service.close()
            print("🔌 Neo4j connection closed")


if __name__ == "__main__":
    asyncio.run(test_hybrid_rag())
