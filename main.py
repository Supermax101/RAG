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
    
    # Create TPN-specialized RAG service
    rag_service = RAGService(embedding_provider, vector_store, llm_provider)
    
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


async def run_tpn_specialist_demo():
    """Run TPN Clinical Specialist interactive demo."""
    from rag.api.dependencies import get_rag_service
    from rag.core.models.documents import RAGQuery
    
    rag_service = get_rag_service()
    
    print("\n🏥 TPN Clinical Specialist - Interactive Demo")
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
