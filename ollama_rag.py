#!/usr/bin/env python3
"""
Ollama-powered Medical RAG System

This is an updated version of the RAG system that uses:
- Ollama for embeddings (nomic-embed-text)
- Ollama for LLM inference (gpt-oss or other models)
- ChromaDB for vector storage
- Your 52 medical documents as knowledge base

No external API keys required!
"""

import requests
import json
import time
import sys
from typing import List, Dict, Optional
from pathlib import Path

# Add app directory to path for imports
sys.path.append(str(Path(__file__).parent))

from app.ollama_search import OllamaRAGSearch
from app.config import get_logger

logger = get_logger("ollama-rag")


def get_available_ollama_models() -> List[Dict[str, str]]:
    """Get list of available Ollama models."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = []
            for model in data.get("models", []):
                name = model.get("name", "unknown")
                size = model.get("size", 0)
                size_gb = size / (1024**3) if size > 0 else 0
                models.append({
                    "name": name,
                    "size": f"{size_gb:.1f} GB" if size_gb > 0 else "Unknown size"
                })
            return models
    except Exception as e:
        logger.error(f"Failed to get Ollama models: {e}")
    return []


def select_ollama_model(models: List[Dict[str, str]]) -> Optional[str]:
    """Interactive model selection."""
    if not models:
        print("❌ No Ollama models found. Please install models with:")
        print("   ollama pull mistral:7b")
        print("   ollama pull gpt-oss")
        return None
    
    print("\n🤖 Available Ollama Models:")
    print("-" * 50)
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model['name']} ({model['size']})")
    print("-" * 50)
    
    while True:
        try:
            choice = input(f"\nSelect model (1-{len(models)}): ").strip()
            if not choice:
                return models[0]["name"]  # Default to first model
            
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                selected = models[idx]["name"]
                print(f"✅ Selected: {selected}")
                return selected
            else:
                print(f"❌ Please enter a number between 1 and {len(models)}")
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            return None


def query_ollama_llm(prompt: str, model: str = "gpt-oss", max_tokens: int = 2000) -> str:
    """Query Ollama LLM with a prompt."""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.1  # Lower temperature for medical accuracy
                }
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "No response received")
        else:
            error_msg = f"Ollama API error: {response.status_code} - {response.text}"
            logger.error(error_msg)
            return error_msg
            
    except requests.exceptions.Timeout:
        error_msg = "⏰ Request timed out. The model might be processing a complex query."
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"❌ Error querying Ollama: {e}"
        logger.error(error_msg)
        return error_msg


def ollama_rag_answer(question: str, model: str = "gpt-oss", search_limit: int = 5) -> dict:
    """
    Generate a RAG answer using Ollama for both search and generation.
    
    Args:
        question: User's question
        model: Ollama model name for generation
        search_limit: Number of documents to retrieve
        
    Returns:
        Dict with answer, sources, and metadata
    """
    start_time = time.time()
    
    # Initialize search system
    rag_search = OllamaRAGSearch()
    
    # Search for relevant documents
    logger.info(f"Searching for: {question}")
    search_results = rag_search.search(question, limit=search_limit)
    
    if not search_results.results:
        return {
            "question": question,
            "answer": "I couldn't find relevant information in the medical documents to answer your question.",
            "sources": [],
            "search_time": search_results.search_time,
            "generation_time": 0,
            "total_time": time.time() - start_time,
            "model": model
        }
    
    # Prepare context from search results
    context_parts = []
    sources = []
    
    for i, result in enumerate(search_results.results, 1):
        context_parts.append(f"[Source {i}] {result.content}")
        sources.append({
            "document": result.document_name,
            "score": result.score,
            "content_preview": result.content[:200] + "..." if len(result.content) > 200 else result.content
        })
    
    context = "\n\n".join(context_parts)
    
    # Create medical RAG prompt
    rag_prompt = f"""You are a medical AI assistant with access to authoritative medical documentation including ASPEN guidelines, NICU protocols, and TPN administration guides.

Based on the following medical documentation, provide a comprehensive and accurate answer to the medical question.

MEDICAL CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Provide evidence-based medical information based only on the provided documentation
- Include specific clinical details, dosages, and protocols when mentioned in the sources
- Organize your response with clear sections and bullet points where appropriate
- If the documentation doesn't contain enough information, state this clearly
- Do not provide medical advice - only educational information from the sources

ANSWER:"""
    
    # Generate response with Ollama
    gen_start = time.time()
    logger.info(f"Generating answer with {model}...")
    answer = query_ollama_llm(rag_prompt, model)
    generation_time = time.time() - gen_start
    
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "search_time": search_results.search_time,
        "generation_time": generation_time,
        "total_time": time.time() - start_time,
        "model": model,
        "search_model": search_results.model_used
    }


def check_chromadb_ready(rag_search: OllamaRAGSearch) -> bool:
    """Check if ChromaDB has medical documents loaded."""
    try:
        stats = rag_search.get_collection_stats()
        doc_count = stats.get('total_documents', 0)
        
        if doc_count == 0:
            print("❌ ChromaDB is empty. Would you like to load all medical documents?")
            response = input("Load documents now? (y/N): ").strip().lower()
            
            if response in ['y', 'yes']:
                print("🔄 Loading all 52 medical documents with Ollama embeddings...")
                chunk_count = rag_search.load_all_documents()
                print(f"✅ Loaded {chunk_count} medical chunks!")
                return True
            else:
                print("Please run the document loading process to use the RAG system.")
                return False
        else:
            print(f"✅ ChromaDB ready with {doc_count} medical documents")
            return True
            
    except Exception as e:
        logger.error(f"Error checking ChromaDB: {e}")
        return False


def interactive_mode(model: str):
    """Run interactive Q&A session."""
    print(f"\n🚀 Ready! Please ask me questions about nutrition, fluids, and electrolytes.")
    print(f"🤖 Using: {model}")
    print("💡 Commands: 'test' for sample question, 'quit' to exit")
    print("=" * 60)
    
    # Initialize search once
    rag_search = OllamaRAGSearch()
    
    # Check if ChromaDB is ready
    if not check_chromadb_ready(rag_search):
        return
    
    sample_questions = [
        "What are the normal serum potassium levels in neonates?",
        "How do you calculate TPN requirements for premature infants?",
        "What are the indications for parenteral nutrition in NICU patients?",
        "How do you monitor for refeeding syndrome?",
        "What are the ASPEN guidelines for lipid administration?"
    ]
    
    while True:
        try:
            print(f"\n❓ Your question: ", end="")
            question = input().strip()
            
            if not question:
                continue
            elif question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif question.lower() == 'test':
                question = sample_questions[0]
                print(f"Using sample question: {question}")
            
            print(f"\n🔍 Finding relevant information...")
            result = ollama_rag_answer(question, model, search_limit=3)
            
            print(f"\n💡 **Answer:**")
            print(result["answer"])
            
            print(f"\n⏱️  {result['total_time']:.1f}s search + generation")
            print("-" * 60)
            
        except KeyboardInterrupt:
            print(f"\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main entry point."""
    print("🏥 Medical Document RAG System")
    print("=" * 60)
    
    # Check Ollama connection
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        if response.status_code != 200:
            print("❌ Cannot connect to Ollama. Please start Ollama service.")
            return
    except:
        print("❌ Cannot connect to Ollama. Please start Ollama service.")
        return
    
    # Get available models
    models = get_available_ollama_models()
    selected_model = select_ollama_model(models)
    
    if not selected_model:
        print("❌ No model selected. Exiting.")
        return
    
    # Run interactive mode
    interactive_mode(selected_model)


if __name__ == "__main__":
    main()
