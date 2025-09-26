#!/usr/bin/env python3
"""
Quick debug script to see what search results look like
"""

from app.search import search_documents

def debug_search(question):
    print(f"Searching for: {question}")
    print("=" * 50)
    
    results = search_documents(question, limit=10)
    
    for i, result in enumerate(results.results, 1):
        print(f"\nResult {i}:")
        print(f"Document: {result.document_name}")
        print(f"Section: {result.section}")
        print(f"Score: {result.score:.3f}")
        print(f"Content length: {len(result.content)} chars")
        print(f"Content: {result.content[:200]}...")
        print("-" * 40)

if __name__ == "__main__":
    # Test with your question
    debug_search("What is 2 in 1 PN?")
