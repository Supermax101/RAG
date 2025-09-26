#!/usr/bin/env python3
"""
EFFICIENT Medical GraphRAG - Optimized for Large Auto Neo4j Knowledge Graphs
Handles 8K+ relationships and 100+ nodes efficiently
"""

import json
import requests
import sys
import warnings
from typing import List, Dict, Optional
from neo4j import GraphDatabase

# Suppress noisy warnings for better UX
warnings.filterwarnings("ignore")
import os
os.environ['PYTHONWARNINGS'] = 'ignore'


class EfficientGraphRAGSearch:
    """High-performance Neo4j GraphRAG search optimized for large automatic graphs"""
    
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
        except Exception as e:
            print(f"❌ Neo4j connection failed: {e}")
            print("Make sure Neo4j is running on localhost:7687")
            sys.exit(1)
    
    def search_medical_entities(self, query_text: str, limit: int = 3) -> List[Dict]:
        """EFFICIENT multi-strategy search leveraging automatic relationships"""
        query_lower = query_text.lower()
        key_terms = [word for word in query_lower.split() if len(word) > 3]
        
        with self.driver.session() as session:
            # STRATEGY 1: Direct content match (fastest)
            print(f"  🎯 Direct content search...")
            direct_results = self.search_direct_content(session, key_terms, limit)
            
            if len(direct_results) >= limit:
                return direct_results[:limit]
            
            # STRATEGY 2: Leverage automatic similarity relationships
            print(f"  🤖 Using auto-similarity network...")
            similarity_results = self.search_via_similarity(session, key_terms, limit)
            
            # STRATEGY 3: Domain-based search using automatic relationships
            print(f"  🏥 Medical domain expansion...")
            domain_results = self.search_via_domains(session, key_terms, limit)
            
            # Combine and deduplicate
            all_results = direct_results + similarity_results + domain_results
            unique_results = self.deduplicate_results(all_results)
            
            # Sort by relevance and return top results
            sorted_results = sorted(unique_results, key=lambda x: x['score'], reverse=True)
            
            print(f"  ✅ Found {len(sorted_results)} unique results")
            return sorted_results[:limit]
    
    def search_direct_content(self, session, key_terms: List[str], limit: int) -> List[Dict]:
        """Fast direct content search with scoring"""
        query = """
        MATCH (cs:ClinicalSection)
        WHERE any(term IN $key_terms WHERE 
            toLower(cs.name) CONTAINS term OR 
            toLower(cs.content) CONTAINS term)
        WITH cs, 
             size([term IN $key_terms WHERE toLower(cs.name) CONTAINS term]) * 10 as name_score,
             size([term IN $key_terms WHERE toLower(cs.content) CONTAINS term]) * 2 as content_score,
             CASE cs.section_type
                WHEN 'electrolyte_physiology' THEN 10
                WHEN 'fluid_management' THEN 9
                WHEN 'clinical_assessment' THEN 8
                WHEN 'treatment_protocols' THEN 7
                WHEN 'iv_fluid_composition' THEN 6
                ELSE 3
             END as type_score
        WITH cs, (name_score + content_score + type_score) as total_score
        WHERE total_score > 5
        RETURN cs.id as id, cs.name as name, cs.content as content, 
               cs.section_type as section_type, total_score as score
        ORDER BY total_score DESC
        LIMIT $limit
        """
        
        results = []
        try:
            result = session.run(query, key_terms=key_terms, limit=limit*2)  # Get more for deduplication
            for record in result:
                results.append({
                    'id': record['id'],
                    'content': f"Section: {record['name']}\n\n{record['content'][:600]}...",
                    'document_name': 'aspen_auto_graph',
                    'section': record['section_type'],
                    'chunk_type': 'DirectMatch',
                    'score': record['score'],
                    'entity_text': record['name'],
                    'search_method': 'direct_content'
                })
        except Exception as e:
            print(f"    ⚠️ Direct search error: {e}")
        
        return results
    
    def search_via_similarity(self, session, key_terms: List[str], limit: int) -> List[Dict]:
        """Leverage automatic CONTENT_SIMILAR relationships for expansion"""
        query = """
        // Find initial matches
        MATCH (cs:ClinicalSection)
        WHERE any(term IN $key_terms WHERE 
            toLower(cs.name) CONTAINS term OR toLower(cs.content) CONTAINS term)
        WITH cs, size([term IN $key_terms WHERE toLower(cs.content) CONTAINS term]) as initial_score
        WHERE initial_score > 0
        
        // Expand via similarity relationships
        MATCH (cs)-[sim:CONTENT_SIMILAR]-(related:ClinicalSection)
        WHERE sim.content_score > 0.3  // Only high similarity
        
        WITH related, max(sim.content_score * initial_score) as expanded_score
        WHERE NOT any(term IN $key_terms WHERE toLower(related.content) CONTAINS term)  // Avoid duplicates
        
        RETURN related.id as id, related.name as name, related.content as content,
               related.section_type as section_type, expanded_score as score
        ORDER BY expanded_score DESC
        LIMIT $limit
        """
        
        results = []
        try:
            result = session.run(query, key_terms=key_terms, limit=limit)
            for record in result:
                results.append({
                    'id': record['id'],
                    'content': f"Related Section: {record['name']}\n\n{record['content'][:500]}...",
                    'document_name': 'aspen_auto_graph',
                    'section': record['section_type'],
                    'chunk_type': 'SimilarityExpansion',
                    'score': record['score'],
                    'entity_text': record['name'],
                    'search_method': 'auto_similarity'
                })
        except Exception as e:
            print(f"    ⚠️ Similarity search error: {e}")
        
        return results
    
    def search_via_domains(self, session, key_terms: List[str], limit: int) -> List[Dict]:
        """Use SAME_DOMAIN and KEYWORD_RELATED relationships for medical context"""
        query = """
        // Find sections matching query
        MATCH (cs:ClinicalSection)
        WHERE any(term IN $key_terms WHERE toLower(cs.content) CONTAINS term)
        WITH cs
        
        // Expand to same medical domain
        MATCH (cs)-[:SAME_DOMAIN]-(domain_related:ClinicalSection)
        WHERE NOT any(term IN $key_terms WHERE toLower(domain_related.content) CONTAINS term)
        
        WITH domain_related, 
             CASE domain_related.section_type
                WHEN 'electrolyte_physiology' THEN 8
                WHEN 'fluid_management' THEN 7
                WHEN 'treatment_protocols' THEN 6
                ELSE 3
             END as domain_score
        
        RETURN domain_related.id as id, domain_related.name as name, 
               domain_related.content as content, domain_related.section_type as section_type,
               domain_score as score
        ORDER BY domain_score DESC
        LIMIT $limit
        """
        
        results = []
        try:
            result = session.run(query, key_terms=key_terms, limit=limit)
            for record in result:
                results.append({
                    'id': record['id'],
                    'content': f"Domain Context: {record['name']}\n\n{record['content'][:400]}...",
                    'document_name': 'aspen_auto_graph',
                    'section': record['section_type'],
                    'chunk_type': 'DomainExpansion',
                    'score': record['score'],
                    'entity_text': record['name'],
                    'search_method': 'medical_domain'
                })
        except Exception as e:
            print(f"    ⚠️ Domain search error: {e}")
        
        return results
    
    def deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate results by ID"""
        seen_ids = set()
        unique_results = []
        
        for result in results:
            if result['id'] not in seen_ids:
                seen_ids.add(result['id'])
                unique_results.append(result)
        
        return unique_results
    
    def get_collection_stats(self) -> Dict:
        """Get efficient stats for the large automatic graph"""
        with self.driver.session() as session:
            # Quick count using index
            result = session.run("MATCH (cs:ClinicalSection) RETURN count(cs) as total_sections")
            sections = result.single()['total_sections']
            
            # Count relationships efficiently
            result = session.run("MATCH ()-[r]->() RETURN count(r) as total_rels")
            rels = result.single()['total_rels']
            
            # Count auto relationship types
            result = session.run("""
                MATCH ()-[r]->() 
                WHERE r.auto_created = true
                RETURN count(r) as auto_rels
            """)
            auto_rels = result.single()['auto_rels']
            
            return {
                'total_chunks': sections,
                'total_documents': 1,  # Single ASPEN document
                'total_relationships': rels,
                'auto_relationships': auto_rels,
                'relationship_density': rels / sections if sections > 0 else 0
            }
    
    def close(self):
        self.driver.close()


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
                    'modified': model.get('modified_at', 'Unknown')[:10]
                })
            
            return models
        else:
            return []
            
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        return []


def select_ollama_model(models: List[Dict[str, str]]) -> Optional[str]:
    """Let user select an Ollama model."""
    if not models:
        print("❌ No Ollama models found. Please install models first:")
        print("   ollama pull mistral:7b")
        print("   ollama pull gemma3:1b")
        return None
    
    print("\n📋 Available Ollama Models:")
    print("-" * 50)
    
    for i, model in enumerate(models, 1):
        medical_indicator = "🏥" if any(med in model['name'].lower() for med in ['med', 'gemma', 'clinical']) else "🤖"
        print(f"  {i}. {medical_indicator} {model['name']} ({model['size']})")
    
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
        print(f"  🤖 Thinking with {model}...")
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.1,  # Lower temperature for medical accuracy
                    "top_p": 0.9
                }
            },
            timeout=60
        )
        
        if response.status_code == 200:
            response_data = response.json()
            result = response_data.get("response", "")
            
            if not result:
                print(f"  ⚠️  Empty response from {model}")
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


def efficient_graphrag_answer(question: str, model: str = "mistral:7b", search_limit: int = 3) -> dict:
    """
    EFFICIENT GraphRAG: Search large Neo4j auto-graph + Generate medical answers
    Optimized for 8K+ relationships and 100+ nodes
    """
    print(f"🚀 Efficient ASPEN GraphRAG Search...")
    
    # Step 1: Efficient search with multiple strategies
    search_engine = EfficientGraphRAGSearch()
    search_results = search_engine.search_medical_entities(question, limit=search_limit)
    
    if not search_results:
        search_engine.close()
        return {
            "question": question,
            "answer": "No relevant medical information found in the comprehensive ASPEN knowledge graph.",
            "sources": [],
            "model_used": model,
            "search_type": "Efficient GraphRAG"
        }
    
    # Step 2: Build efficient context from diverse search methods
    context_parts = []
    sources = []
    
    for i, result in enumerate(search_results, 1):
        # Clean content and add method info
        clean_content = result['content'].replace('$', '').replace('\\', '')
        method_tag = f"[{result['search_method'].upper()}]"
        context_parts.append(f"[Evidence {i}] {method_tag} {clean_content}")
        
        sources.append({
            "index": i,
            "method": result['search_method'],
            "document": result['document_name'],
            "section": result['section'],
            "type": result['chunk_type'],
            "score": result['score'],
            "entity": result['entity_text']
        })
    
    context = "\n\n".join(context_parts)
    
    # Step 3: Optimized medical prompt
    medical_prompt = f"""You are a clinical nutrition expert answering from comprehensive ASPEN guidelines.

QUESTION: {question}

COMPREHENSIVE EVIDENCE FROM AUTOMATIC KNOWLEDGE GRAPH:
{context}

Instructions:
- Always start with "Answer and keep the answer short and concise.
- Provide a precise, evidence-based answer only when prompted.
- Reference specific values, ranges, and clinical recommendations.
- Use professional medical language
- Do not provide any other information than what has been asked.



Clinical Answer:"""
    
    # Step 4: Generate answer
    answer = query_ollama(medical_prompt, model=model, max_tokens=400)
    
    search_engine.close()
    
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "model_used": model,
        "search_type": "Efficient GraphRAG",
        "entities_found": len(search_results),
        "search_methods": list(set([s['method'] for s in sources]))
    }


def check_efficient_graphrag_ready() -> bool:
    """Check if the large automatic Neo4j graph is ready"""
    try:
        search_engine = EfficientGraphRAGSearch()
        stats = search_engine.get_collection_stats()
        
        total_sections = stats.get('total_chunks', 0)
        total_rels = stats.get('total_relationships', 0)
        auto_rels = stats.get('auto_relationships', 0)
        density = stats.get('relationship_density', 0)
        
        if total_sections > 50 and total_rels > 1000:  # Ensure we have the large auto graph
            print(f"🚀 EFFICIENT ASPEN GraphRAG Ready!")
            print(f"   📊 {total_sections} sections, {total_rels} relationships ({auto_rels} automatic)")
            print(f"   ⚡ Density: {density:.1f} relationships per node")
            print(f"   🤖 Auto-relationships from Neo4j APOC similarity detection")
            search_engine.close()
            return True
        else:
            print(f"❌ Automatic graph too small: {total_sections} sections, {total_rels} relationships")
            print(f"Please run: python auto_neo4j_builder.py")
            search_engine.close()
            return False
            
    except Exception as e:
        print(f"❌ GraphRAG not accessible: {e}")
        print("Please make sure Neo4j is running and run: python auto_neo4j_builder.py")
        return False


def main():
    """EFFICIENT ASPEN GraphRAG CLI optimized for large automatic graphs"""
    print("🚀 EFFICIENT ASPEN Medical GraphRAG System")
    print("⚡ Optimized for Large Auto Neo4j Knowledge Graphs")
    print("🤖 Leveraging APOC Similarity + Domain Relationships")
    print("=" * 70)
    
    # Step 1: Check large automatic GraphRAG
    if not check_efficient_graphrag_ready():
        sys.exit(1)
    
    # Step 2: Get Ollama models
    models = get_available_ollama_models()
    if not models:
        sys.exit(1)
    
    # Step 3: Model selection
    selected_model = select_ollama_model(models)
    if not selected_model:
        sys.exit(1)
    
    # Step 4: Interactive session
    print(f"\n🏥 Efficient ASPEN GraphRAG Ready!")
    print(f"🤖 Model: {selected_model}")
    print(f"📚 Knowledge: Auto-connected ASPEN Fluids & Electrolytes")
    print(f"⚡ Search: Multi-strategy (Direct + Similarity + Domain)")
    print(f"Commands: 'test' for demo, 'quit' to exit")
    print("=" * 70)
    
    while True:
        try:
            print()
            user_input = input("Medical Question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n🏥 Thanks for using Efficient ASPEN GraphRAG!")
                break
            
            if user_input.lower() == 'test':
                print()
                test_question = "What's the difference between hypokalemia and hyperkalemia management?"
                print(f"🧪 Testing: {test_question}")
                result = efficient_graphrag_answer(test_question, model=selected_model)
            else:
                if not user_input:
                    continue
                
                print()
                result = efficient_graphrag_answer(user_input, model=selected_model)
            
            # Display results efficiently
            if result.get('answer'):
                print(f"\n💡 Clinical Answer:")
                print(f"{result['answer']}")
                
                print(f"\n📊 Search Analytics:")
                print(f"  Methods used: {', '.join(result.get('search_methods', []))}")
                print(f"  Evidence sources: {result.get('entities_found', 0)}")
                
                if result.get('sources'):
                    print(f"\n📚 Evidence Sources:")
                    for source in result['sources'][:3]:
                        method_tag = f"[{source['method'].upper()}]"
                        entity_short = source['entity'][:50] + "..." if len(source['entity']) > 50 else source['entity']
                        print(f"  • {method_tag} {entity_short}")
                        print(f"    Type: {source['type']}, Score: {source['score']:.1f}")
            else:
                print(f"\n❌ No answer generated")
            
            print("-" * 70)
            
        except KeyboardInterrupt:
            print("\n\n🏥 Thanks for using Efficient ASPEN GraphRAG!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'quit' to exit.")
            print("-" * 70)


if __name__ == "__main__":
    main()
