"""
TPN Nutrition Specialist System - Main Entry Point
Specialized RAG system for Total Parenteral Nutrition recommendations 
based on 52 ASPEN/TPN clinical guidelines and protocols.
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag.core.services.rag_service import RAGService
from rag.core.services.hybrid_rag_service import HybridRAGService
from rag.core.services.document_loader import DocumentLoader
from rag.core.services.database_manager import DatabaseManager
from rag.infrastructure.embeddings.ollama_embeddings import OllamaEmbeddingProvider
from rag.infrastructure.vector_stores.chroma_store import ChromaVectorStore
from rag.infrastructure.llm_providers.ollama_provider import OllamaLLMProvider
from rag.config.settings import settings


async def initialize_tpn_system():
    """Initialize the TPN Specialist RAG system and load ASPEN documents."""
    print("🏥 Initializing TPN Nutrition Specialist System...")
    print("📚 Based on 52 ASPEN/TPN Clinical Guidelines and Protocols")
    
    # Ensure directories exist
    settings.ensure_directories()
    
    # Initialize providers
    print("🔧 Initializing TPN specialist providers...")
    embedding_provider = OllamaEmbeddingProvider()
    vector_store = ChromaVectorStore()
    llm_provider = OllamaLLMProvider()
    
    # Check Ollama health
    print("🤖 Checking Ollama health...")
    if not await llm_provider.check_health():
        print("❌ Ollama is not running. Please start Ollama service:")
        print("   ollama serve")
        print("   ollama pull nomic-embed-text")
        print("   ollama pull mistral:7b")
        return False
    
    # Create TPN-specialized HYBRID RAG service (ChromaDB + Neo4j + LangChain + LangGraph)
    rag_service = HybridRAGService(
        embedding_provider=embedding_provider, 
        vector_store=vector_store, 
        llm_provider=llm_provider,
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j", 
        neo4j_password="medicalpass123"
    )
    
    # Check if we need to load TPN documents with enhanced processing
    stats = await rag_service.get_collection_stats()
    
    if stats["total_chunks"] == 0:
        print("📄 No TPN documents found in vector store.")
        print("🚀 Loading 52 ASPEN/TPN documents with enhanced chunking...")
        
        # Use document loader with enhanced medical chunking
        document_loader = DocumentLoader(rag_service)
        result = await document_loader.load_all_documents()
        
        if result["loaded"] == 0:
            print("❌ No TPN documents were loaded. Please check your data/parsed directory.")
            return False
        
        print(f"✅ Successfully loaded {result['loaded']} TPN documents with {result['total_chunks']} optimized chunks")
    else:
        print(f"✅ Found {stats['total_chunks']} TPN chunks from {stats['total_documents']} ASPEN documents")
    
    return True


async def get_available_ollama_models():
    """Get list of available Ollama LLM models (excludes embedding models)."""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                data = response.json()
                all_models = [model["name"] for model in data.get("models", [])]
                
                # Filter out embedding-only models
                embedding_keywords = ["embed", "embedding", "nomic-embed"]
                llm_models = [
                    model for model in all_models 
                    if not any(keyword in model.lower() for keyword in embedding_keywords)
                ]
                
                return llm_models
            else:
                return []
    except Exception:
        return []

def select_ollama_model(available_models):
    """Let user select from available Ollama LLM models."""
    if not available_models:
        print("❌ No Ollama LLM models found. Please pull some models first:")
        print("   ollama pull phi4:latest")
        print("   ollama pull mistral:7b")
        print("   ollama pull llama3:8b")
        return None
    
    print(f"\n🤖 Available Ollama LLM Models ({len(available_models)} found):")
    for i, model in enumerate(available_models, 1):
        # Detect model info
        model_info = ""
        
        # Check for parameter sizes
        size_patterns = {
            "120b": "120B parameters",
            "70b": "70B parameters",
            "27b": "27B parameters",
            "14b": "14B parameters",
            "13b": "13B parameters",
            "8b": "8B parameters",
            "7b": "7B parameters",
            "3b": "3B parameters"
        }
        
        for size, desc in size_patterns.items():
            if size in model.lower():
                model_info = f" ({desc})"
                break
        
        # Special handling for phi4 models
        if "phi4" in model.lower():
            if "mini" in model.lower():
                model_info = " (Phi-4 Mini, 14B parameters)"
            else:
                model_info = " (Phi-4, 14B parameters)"
        
        # Special handling for gpt-oss
        if "gpt-oss" in model.lower():
            model_info = " (GPT-OSS, 120B parameters)"
            
        print(f"  {i}. {model}{model_info}")
    
    while True:
        try:
            choice = input(f"\n🔢 Select model (1-{len(available_models)}) or press Enter for default: ").strip()
            
            if not choice:  # Default to first model
                return available_models[0]
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(available_models):
                return available_models[choice_num - 1]
            else:
                print(f"❌ Please enter a number between 1 and {len(available_models)}")
        except ValueError:
            print("❌ Please enter a valid number")

async def run_tpn_specialist_demo():
    """Run TPN Clinical Specialist interactive demo with model selection."""
    from rag.core.models.documents import RAGQuery
    from rag.infrastructure.embeddings.ollama_embeddings import OllamaEmbeddingProvider
    from rag.infrastructure.vector_stores.chroma_store import ChromaVectorStore
    from rag.infrastructure.llm_providers.ollama_provider import OllamaLLMProvider
    from rag.core.services.hybrid_rag_service import HybridRAGService
    
    print("🔍 Checking available Ollama models...")
    available_models = await get_available_ollama_models()
    
    if not available_models:
        print("❌ No Ollama models available. Please pull some models first:")
        print("   ollama pull mistral:7b")
        print("   ollama pull llama3:8b")
        return False
    
    # Let user select model
    selected_model = select_ollama_model(available_models)
    if not selected_model:
        return False
    
    print(f"✅ Selected model: {selected_model}")
    
    # Initialize providers with selected model
    embedding_provider = OllamaEmbeddingProvider()
    vector_store = ChromaVectorStore()
    llm_provider = OllamaLLMProvider(default_model=selected_model)
    
    # Create HYBRID RAG service (ChromaDB + Neo4j + LangChain + LangGraph)  
    rag_service = HybridRAGService(
        embedding_provider=embedding_provider, 
        vector_store=vector_store, 
        llm_provider=llm_provider,
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j", 
        neo4j_password="medicalpass123"
    )
    
    print(f"\n🏥 TPN Clinical Specialist - Interactive Demo (Using {selected_model})")
    print("=" * 60)
    print("💊 Ask TPN/parenteral nutrition questions based on your 52 ASPEN documents")
    print("🔬 Example questions:")
    print("   • Calculate TPN for a 1.2kg preterm infant")
    print("   • Normal potassium range for neonates on TPN")
    print("   • TPN lipid contraindications in IFALD")
    print("   • Monitoring frequency for TPN glucose")
    print("Type 'quit' to exit.")
    print("=" * 60)
    
    while True:
        try:
            question = input("\n🩺 TPN Question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Thank you for using TPN Specialist!")
                break
            
            if not question:
                continue
            
            print("🔍 Searching ASPEN TPN guidelines...")
            
            query = RAGQuery(question=question, search_limit=4)
            response = await rag_service.ask(query)
            
            print(f"\n💡 **TPN Clinical Recommendation:**")
            print(response.answer)
            
            print(f"\n📚 **ASPEN Sources Used ({len(response.sources)} found):**")
            for i, source in enumerate(response.sources, 1):
                doc_name = source.document_name
                if "aspen" in doc_name.lower() or "tpn" in doc_name.lower():
                    doc_indicator = "🏥"
                else:
                    doc_indicator = "📋"
                section = source.chunk.section or "General"
                print(f"  {i}. {doc_indicator} {doc_name[:40]}...")
                print(f"     Section: {section[:50]} (relevance: {source.score:.3f})")
            
            print(f"\n⏱️  Response time: {response.total_time_ms:.1f}ms")
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n👋 Thank you for using TPN Specialist!")
            break
        except Exception as e:
            print(f"❌ TPN System Error: {e}")
            print("Please try rephrasing your TPN question.")
    
    # Close Neo4j connection
    if hasattr(rag_service, 'close'):
        rag_service.close()
        print("🔌 Neo4j connection closed")


async def start_api_server():
    """Start the FastAPI server."""
    import uvicorn
    from rag.api.main import app
    
    print("🌐 Starting FastAPI server...")
    print(f"📖 API Documentation: http://localhost:{settings.api_port}/docs")
    
    config = uvicorn.Config(
        app=app,
        host=settings.api_host,
        port=settings.api_port,
        reload=False  # Don't use reload in production
    )
    server = uvicorn.Server(config)
    await server.serve()


async def main():
    """TPN Specialist System main entry point."""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "init":
            # Initialize TPN system and load ASPEN documents
            success = await initialize_tpn_system()
            if success:
                print("\n🎉 TPN Specialist System initialized successfully!")
                print("\nNext steps:")
                print("  python main.py demo    # Run TPN specialist demo")
                print("  python main.py serve   # Start TPN API server")
                print("  python main.py reset   # Reset and reload with enhanced processing")
            sys.exit(0 if success else 1)
            
        elif command == "demo":
            # Run TPN specialist interactive demo
            if await initialize_tpn_system():
                await run_tpn_specialist_demo()
            sys.exit(0)
            
        elif command == "serve":
            # Start TPN API server
            if await initialize_tpn_system():
                await start_api_server()
            sys.exit(0)
            
        elif command == "reset":
            # Reset ChromaDB and reload with enhanced processing
            if await initialize_tpn_system():
                from rag.api.dependencies import get_rag_service
                db_manager = DatabaseManager(get_rag_service())
                result = await db_manager.reset_and_reload_enhanced(confirm=True)
                print(f"📊 Reset result: {result}")
            sys.exit(0)
            
        else:
            print(f"❌ Unknown command: {command}")
    
    # Default: show TPN system usage
    print("🏥 TPN Nutrition Specialist System v2.0")
    print("📚 Based on 52 ASPEN/TPN Clinical Guidelines")
    print("=" * 50)
    print("Usage:")
    print("  python main.py init    # Initialize TPN system with ASPEN documents")
    print("  python main.py demo    # Run TPN clinical specialist demo")
    print("  python main.py serve   # Start TPN API server (FastAPI)")
    print("  python main.py reset   # Reset ChromaDB with enhanced processing")
    print("\n🔧 For processing new TPN PDFs:")
    print("  python -m ocr_pipeline.main test-ingest")
    print("\n💡 TPN Specialist Features:")
    print("  • ASPEN guideline-based recommendations")
    print("  • Age-specific TPN calculations (preterm/term/pediatric)")
    print("  • Board-style clinical question answering")
    print("  • TPN component dosing and monitoring")
    print("  • Source-constrained responses (52 documents only)")


if __name__ == "__main__":
    asyncio.run(main())
