#!/usr/bin/env python3
"""
Medical Document Loader for Ollama RAG System

This script loads all 52 medical documents into ChromaDB using Ollama embeddings.
Run this once to set up your complete medical knowledge base.
"""

import sys
from pathlib import Path
import time

# Add app directory to path
sys.path.append(str(Path(__file__).parent))

from app.ollama_search import OllamaRAGSearch
from app.config import get_logger

logger = get_logger("doc-loader")


def main():
    """Load all medical documents into ChromaDB."""
    print("🏥 Medical Document Loader")
    print("=" * 50)
    print("This will load all 52 medical documents into ChromaDB using Ollama embeddings.")
    print("This process may take several minutes...")
    print()
    
    # Check if user wants to proceed
    response = input("Continue? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Cancelled.")
        return
    
    # Initialize search system
    print("\n🔄 Initializing Ollama RAG search system...")
    rag_search = OllamaRAGSearch()
    
    # Load all documents
    print("📚 Loading all medical documents...")
    start_time = time.time()
    
    try:
        chunk_count = rag_search.load_all_documents()
        load_time = time.time() - start_time
        
        print(f"\n🎉 SUCCESS!")
        print(f"✅ Loaded {chunk_count} medical chunks")
        print(f"⏱️  Total time: {load_time:.1f} seconds")
        print(f"📊 Average: {chunk_count/load_time:.1f} chunks/second")
        
        # Test the loaded system
        print(f"\n🧪 Testing search functionality...")
        test_results = rag_search.search("parenteral nutrition", limit=2)
        
        if test_results.results:
            print(f"✅ Search test passed! Found {len(test_results.results)} results")
            print(f"Sample result: {test_results.results[0].document_name}")
        else:
            print("⚠️  Search test failed - no results found")
        
        print(f"\n🚀 Your medical RAG system is ready!")
        print(f"Run: python ollama_rag.py")
        
    except Exception as e:
        print(f"❌ Error loading documents: {e}")
        logger.error(f"Document loading failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
