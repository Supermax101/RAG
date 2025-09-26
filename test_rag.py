#!/usr/bin/env python3
"""
Medical Document RAG CLI - Search and answer questions using ChromaDB + Ollama
"""

import json
import requests
import sys
import warnings
from typing import List, Dict, Optional

# Suppress noisy warnings for better UX
warnings.filterwarnings("ignore")
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

# Redirect ChromaDB telemetry errors to suppress them
import sys
from contextlib import redirect_stderr
from io import StringIO

from app.search import search_documents


def get_available_ollama_models() -> List[Dict[str, str]]:
    """Get list of available Ollama models."""
    try:
        print("🔍 Checking available Ollama models...")
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            models = []
            
            for model in data.get('models', []):
                name = model.get('name', '')
                size = model.get('size', 0)
                
                # Convert size to readable format
                if size > 1024**3:  # GB
                    size_str = f"{size / 1024**3:.1f} GB"
                elif size > 1024**2:  # MB
                    size_str = f"{size / 1024**2:.0f} MB"
                else:
                    size_str = f"{size} bytes"
                
                models.append({
                    'name': name,
                    'size': size_str,
                    'modified': model.get('modified_at', 'Unknown')[:10]  # Just date part
                })
            
            return models
        else:
            return []
            
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        print("Make sure Ollama is running: ollama serve")
        return []


def select_ollama_model(models: List[Dict[str, str]]) -> Optional[str]:
    """Let user select an Ollama model."""
    if not models:
        print("❌ No Ollama models found. Please install models first:")
        print("   ollama pull mistral:7b")
        print("   ollama pull phi4:latest")
        return None
    
    print("\n📋 Available Ollama Models:")
    print("-" * 50)
    
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model['name']} ({model['size']})")
    
    print("-" * 50)
    
    while True:
        try:
            choice = input(f"\nSelect model (1-{len(models)}): ").strip()
            
            if not choice:
                continue
                
            choice_num = int(choice)
            if 1 <= choice_num <= len(models):
                selected_model = models[choice_num - 1]['name']
                print(f"✅ Selected: {selected_model}")
                return selected_model
            else:
                print(f"Please enter a number between 1 and {len(models)}")
                
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            return None


def query_ollama(prompt: str, model: str = "mistral:7b", max_tokens: int = 300) -> str:
    """Query a local Ollama model."""
    try:
        # Simplified connection message
        print(f"  🤖 Thinking with {model}...")
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.3,  # Lower temperature for more focused answers
                    "top_p": 0.9
                }
            },
            timeout=60
        )
        
        if response.status_code == 200:
            response_data = response.json()
            result = response_data.get("response", "")
            
            # Debug: Check what we got
            if not result:
                print(f"  ⚠️  Empty response from {model}")
                print(f"  📄 Response data keys: {response_data.keys()}")
                return "No response generated from model"
            
            return result.strip()
        else:
            error_msg = f"HTTP {response.status_code} - {response.text[:200]}"
            print(f"  ❌ Ollama error: {error_msg}")
            return f"Error: {error_msg}"
            
    except Exception as e:
        error_msg = f"Error querying Ollama: {e}"
        print(f"  ❌ Exception: {error_msg}")
        return error_msg


def rag_answer(question: str, model: str = "mistral:7b", search_limit: int = 5) -> dict:
    """
    Perform RAG: Search documents + Generate answer with context.
    """
    print(f"🔍 Finding relevant information...")
    
    # Step 1: Search for relevant documents
    search_results = search_documents(question, limit=search_limit)
    
    if not search_results.results:
        return {
            "question": question,
            "answer": "No relevant documents found for your question.",
            "sources": [],
            "search_time": search_results.search_time_seconds
        }
    
    # Step 2: Build context from search results
    context_parts = []
    sources = []
    
    for i, result in enumerate(search_results.results, 1):
        context_parts.append(f"[Source {i}] {result.content}")
        sources.append({
            "index": i,
            "document": result.document_name,
            "section": result.section,
            "type": result.chunk_type,
            "score": result.score
        })
    
    context = "\n\n".join(context_parts)
    
    # Step 3: Simple conversational RAG prompt
    rag_prompt = f"""Answer this question using only the information below: {question}

{context}

Instructions:
1. Always answer from the current knowledge base only
2. Keep the conversation natural

Answer:"""
    
    # Step 4: Generate answer
    answer = query_ollama(rag_prompt, model=model)
    
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "search_time": search_results.search_time_seconds,
        "model_used": model
    }


def test_single_question():
    """Test with a specific question."""
    question = "What is the daily sodium requirement?"
    print(f"🏥 Medical Document RAG Test")
    print("=" * 50)
    print(f"Question: {question}")
    print()
    
    result = rag_answer(question, model="mistral:7b")
    
    print(f"🤖 Model: {result['model_used']}")
    print(f"⏱️  Search time: {result['search_time']:.2f}s")
    print()
    print("💡 Answer:")
    print(result['answer'])
    print()
    print("📚 Sources used:")
    for source in result['sources']:
        print(f"  • {source['document']} - {source['section']} (score: {source['score']:.3f})")
    print()


def check_chromadb_ready() -> bool:
    """Check if ChromaDB has documents loaded."""
    try:
        from app.search import ChromaRAGSearch
        search_engine = ChromaRAGSearch()
        
        # Try to get collection stats
        stats = search_engine.get_collection_stats()
        total_chunks = stats.get('total_chunks', 0)
        
        if total_chunks > 0:
            print(f"✅ ChromaDB ready: {total_chunks} chunks from {stats.get('total_documents', 0)} documents")
            return True
        else:
            print("❌ ChromaDB is empty. Please run:")
            print("   python -m app.main create-embeddings")
            print("   python -m app.main load-chromadb")
            return False
            
    except Exception as e:
        print(f"❌ ChromaDB not accessible: {e}")
        print("Please run: python -m app.main load-chromadb")
        return False


def main():
    """Interactive RAG CLI with model selection."""
    print("🏥 Medical Document RAG System")
    print("=" * 60)
    
    # Step 1: Check ChromaDB (suppress noisy logs)
    if not check_chromadb_ready():
        sys.exit(1)
    
    # Step 2: Get available Ollama models
    models = get_available_ollama_models()
    if not models:
        sys.exit(1)
    
    # Step 3: Let user select model
    selected_model = select_ollama_model(models)
    if not selected_model:
        sys.exit(1)
    
    # Step 4: Start interactive session
    print(f"\n🚀 Ready! Please ask me questions about nutrition, fluids, and electrolytes.")
    print(f"🤖 Using: {selected_model}")
    print(f"💡 Commands: 'test' for sample question, 'quit' to exit")
    print("=" * 60)
    
    while True:
        try:
            print()
            user_input = input("❓ Your question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thanks for using the RAG system!")
                break
            
            if user_input.lower() == 'test':
                print()
                test_single_question_with_model(selected_model)
                continue
            
            if not user_input:
                continue
            
            print()
            result = rag_answer(user_input, model=selected_model)
            
            # Debug: Check if answer exists
            if 'answer' in result and result['answer']:
                print(f"\n💡 **Answer:**")
                print(f"{result['answer']}")
            else:
                print(f"\n❌ No answer generated. Debug info:")
                print(f"   Result keys: {result.keys()}")
                print(f"   Answer value: {result.get('answer', 'KEY_MISSING')}")
            
            # Compact source display
            if result['sources']:
                print(f"\n📚 Found in {len(result['sources'])} sections:")
                for source in result['sources'][:3]:  # Show top 3 sources
                    section = source['section'][:40] + "..." if len(source['section']) > 40 else source['section']
                    print(f"  • {section}")
                
                if len(result['sources']) > 3:
                    print(f"  • +{len(result['sources']) - 3} more sources")
            
            print(f"\n⏱️  {result['search_time']:.1f}s search")
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Thanks for using the RAG system!")
            break
        except Exception as e:
            print(f"\n❌ Something went wrong: {e}")
            print("Please try again or type 'quit' to exit.")
            print("-" * 60)


def test_single_question_with_model(model: str):
    """Test with a specific question using selected model."""
    question = "What is the daily sodium requirement?"
    print(f"🧪 Testing with: {question}")
    print(f"🤖 Using model: {model}")
    print()
    
    result = rag_answer(question, model=model)
    
    print(f"⏱️  Search time: {result['search_time']:.2f}s")
    print("\n💡 Answer:")
    print(result['answer'])
    print("\n📚 Sources:")
    for source in result['sources']:
        print(f"  • {source['document']} - {source['section']} (score: {source['score']:.3f})")


if __name__ == "__main__":
    main()
