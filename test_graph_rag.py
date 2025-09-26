#!/usr/bin/env python3
"""
Medical Document GraphRAG CLI - Search and answer questions using Neo4j + Ollama
Focused on ASPEN Fluids and Electrolytes knowledge
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


class GraphRAGSearch:
    """Simple Neo4j GraphRAG search engine for ASPEN medical knowledge"""
    
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
    
    def search_medical_entities(self, query_text: str, limit: int = 5) -> List[Dict]:
        """Intelligent multi-tier search for any medical question"""
        results = []
        query_lower = query_text.lower()
        
        with self.driver.session() as session:
            # STEP 1: Determine question type and search strategy
            question_type = self.classify_question(query_lower)
            print(f"  Question type: {question_type}")
            
            # STEP 2: Use targeted search based on question type
            if question_type == "normal_range":
                results = self.search_normal_ranges(session, query_lower, limit)
            elif question_type == "comparison":
                results = self.search_comparisons(session, query_lower, limit)
            elif question_type == "dosage":
                results = self.search_dosages(session, query_lower, limit)
            elif question_type == "symptoms":
                results = self.search_symptoms(session, query_lower, limit)
            else:
                results = self.search_general_content(session, query_lower, limit)
            
            # STEP 3: Quality filter results
            results = self.filter_quality_results(results, query_lower)
            
            print(f"  Found {len(results)} relevant results")
        
        return results
    
    def classify_question(self, query_lower: str) -> str:
        """Classify the type of medical question to optimize search"""
        if any(word in query_lower for word in ["normal", "range", "reference", "typical"]):
            return "normal_range"
        elif any(word in query_lower for word in ["vs", "versus", "difference", "compare"]):
            return "comparison"
        elif any(word in query_lower for word in ["dose", "dosage", "mg", "meq", "amount"]):
            return "dosage"
        elif any(word in query_lower for word in ["symptom", "sign", "presentation", "clinical"]):
            return "symptoms"
        else:
            return "general"
    
    def search_normal_ranges(self, session, query_lower: str, limit: int) -> List[Dict]:
        """Search for normal ranges and reference values"""
        print(f"    Searching for normal ranges...")
        
        # Extract the parameter we're looking for
        key_terms = [word for word in query_lower.split() if len(word) > 3]
        
        # Search ClinicalSections that contain both the parameter and range info
        query = """
        MATCH (cs:ClinicalSection)
        WHERE any(term IN $key_terms WHERE toLower(cs.content) CONTAINS term)
          AND (toLower(cs.content) CONTAINS 'normal' 
               OR toLower(cs.content) CONTAINS 'range' 
               OR toLower(cs.content) CONTAINS 'reference')
          AND (toLower(cs.content) CONTAINS 'meq' 
               OR toLower(cs.content) CONTAINS 'mg' 
               OR toLower(cs.content) CONTAINS 'mmol')
        WITH cs, size([term IN $key_terms WHERE toLower(cs.content) CONTAINS term]) as match_score
        RETURN cs.name as title, cs.content as content, match_score
        ORDER BY match_score DESC
        LIMIT $limit
        """
        
        results = []
        try:
            result = session.run(query, key_terms=key_terms, limit=limit)
            for record in result:
                # Extract the specific range from content
                content = record['content']
                relevant_excerpt = self.extract_range_excerpt(content, key_terms)
                
                if relevant_excerpt:
                    results.append({
                        'content': relevant_excerpt,
                        'document_name': 'aspen_complete',
                        'section': 'reference_values',
                        'chunk_type': 'MedicalSection',
                        'score': record['match_score'],
                        'entity_text': record['title'],
                        'entity_value': relevant_excerpt,
                        'match_count': record['match_score']
                    })
        except Exception as e:
            print(f"    Warning: Range search error: {e}")
            
        return results
    
    def search_general_content(self, session, query_lower: str, limit: int) -> List[Dict]:
        """General content search with relevance scoring"""
        print(f"    General content search...")
        
        key_terms = [word for word in query_lower.split() if len(word) > 3]
        
        query = """
        MATCH (cs:ClinicalSection)
        WHERE any(term IN $key_terms WHERE 
            toLower(cs.name) CONTAINS term OR 
            toLower(cs.content) CONTAINS term)
        WITH cs, 
             size([term IN $key_terms WHERE toLower(cs.name) CONTAINS term]) * 5 as title_score,
             size([term IN $key_terms WHERE toLower(cs.content) CONTAINS term]) as content_score,
             CASE cs.section_type
                WHEN 'electrolyte_disorders' THEN 10
                WHEN 'electrolyte_physiology' THEN 9
                WHEN 'fluid_management' THEN 8
                WHEN 'clinical_assessment' THEN 8
                WHEN 'treatment_protocols' THEN 7
                WHEN 'iv_fluid_composition' THEN 7
                WHEN 'special_conditions' THEN 6
                WHEN 'general_clinical' THEN 5
                WHEN 'clinical_content' THEN 3
                WHEN 'administrative' THEN 1
                ELSE 2
             END as type_bonus
        WITH cs, (title_score + content_score + type_bonus) as final_score
        WHERE final_score > 2
        RETURN cs.name as title, substring(cs.content, 0, 400) as content, 
               final_score, cs.section_type as section_type
        ORDER BY final_score DESC
        LIMIT $limit
        """
        
        results = []
        try:
            result = session.run(query, key_terms=key_terms, limit=limit)
            for record in result:
                results.append({
                    'content': f"Title: {record['title']}\nContent: {record['content']}",
                    'document_name': 'aspen_complete', 
                    'section': record['section_type'],
                    'chunk_type': 'MedicalSection',
                    'score': record['final_score'],
                    'entity_text': record['title'],
                    'entity_value': record['section_type'],
                    'match_count': record['final_score']
                })
        except Exception as e:
            print(f"    Warning: General search error: {e}")
            
        return results
    
    def extract_range_excerpt(self, content: str, key_terms: List[str]) -> str:
        """Extract the specific range information from content"""
        lines = content.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            # Check if this line contains our key terms and range info
            if (any(term in line_lower for term in key_terms) and 
                ('meq' in line_lower or 'mg' in line_lower or 'mmol' in line_lower) and
                any(word in line_lower for word in ['normal', 'range', '-', 'to'])):
                # Clean up mathematical notation
                clean_line = line.replace('$', '').replace('\\mathrm{', '').replace('}', '').replace('\\', '')
                return clean_line.strip()
        
        # Fallback: return first 200 chars
        clean_content = content.replace('$', '').replace('\\mathrm{', '').replace('}', '').replace('\\', '')
        return clean_content[:200] + "..."
    
    def search_comparisons(self, session, query_lower: str, limit: int) -> List[Dict]:
        """Search for comparative information"""
        print(f"    Searching for comparisons...")
        return self.search_general_content(session, query_lower, limit)
    
    def search_dosages(self, session, query_lower: str, limit: int) -> List[Dict]:
        """Search for dosage information"""
        print(f"    Searching for dosages...")
        return self.search_general_content(session, query_lower, limit)
    
    def search_symptoms(self, session, query_lower: str, limit: int) -> List[Dict]:
        """Search for symptom information"""
        print(f"    Searching for symptoms...")
        return self.search_general_content(session, query_lower, limit)
    
    def filter_quality_results(self, results: List[Dict], query_lower: str) -> List[Dict]:
        """Filter out low-quality results that don't actually answer the question"""
        if not results:
            return results
            
        # Sort by score and take top results
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Filter out results that are too short or don't contain key terms
        key_terms = set(word for word in query_lower.split() if len(word) > 3)
        
        filtered = []
        for result in results:
            content_lower = result['content'].lower()
            
            # Must contain at least one key term
            if any(term in content_lower for term in key_terms):
                # Must have substantial content
                if len(result['content']) > 50:
                    filtered.append(result)
        
        return filtered[:5]  # Return top 5 quality results
    
    def get_collection_stats(self) -> Dict:
        """Get GraphRAG statistics for comprehensive knowledge graph"""
        with self.driver.session() as session:
            # Count all comprehensive medical knowledge nodes
            result = session.run("""
            MATCH (n)
            WHERE n:ClinicalSection OR n:ClinicalTable
            RETURN count(n) as total_entities
            """)
            
            stats = result.single()
            
            # Count documents
            doc_result = session.run("MATCH (d:ClinicalSection) RETURN count(DISTINCT d.doc_source) as total_documents")
            doc_count = doc_result.single()['total_documents']
            
            # Count entity types
            type_result = session.run("""
            MATCH (n) 
            WHERE n:ClinicalSection OR n:ClinicalTable
            RETURN count(DISTINCT labels(n)[0]) as entity_types
            """)
            type_count = type_result.single()['entity_types']
            
            # Get relationship count
            rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as total_relationships")
            rel_count = rel_result.single()['total_relationships']
            
            return {
                'total_chunks': stats['total_entities'],
                'total_documents': doc_count,
                'entity_types': type_count,
                'total_relationships': rel_count
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
        print("   ollama pull alibayram/medgemma:4b")
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


def graphrag_answer(question: str, model: str = "mistral:7b", search_limit: int = 3) -> dict:
    """
    Perform GraphRAG: Search Neo4j ASPEN knowledge + Generate answer with medical context.
    """
    print(f"Searching ASPEN medical knowledge graph...")
    
    # Step 1: Search Neo4j for relevant medical entities
    search_engine = GraphRAGSearch()
    search_results = search_engine.search_medical_entities(question, limit=search_limit)
    
    if not search_results:
        search_engine.close()
        return {
            "question": question,
            "answer": "No relevant medical information found in the ASPEN knowledge graph.",
            "sources": [],
            "model_used": model,
            "search_type": "GraphRAG"
        }
    
    # Step 2: Build medical context from graph results
    context_parts = []
    sources = []
    
    for i, result in enumerate(search_results, 1):
        # Clean up the content for better readability
        clean_content = result['content'].replace('$', '').replace('\\mathrm{', '').replace('}', '').replace('\\', '')
        context_parts.append(f"[Reference {i}] {clean_content}")
        sources.append({
            "index": i,
            "document": result['document_name'],
            "section": result['section'],
            "type": result['chunk_type'],
            "score": result['score'],
            "entity": result['entity_text'],
            "value": result.get('entity_value', ''),
            "unit": result.get('entity_unit', ''),
            "matches": result.get('match_count', 0)
        })
    
    context = "\n\n".join(context_parts)
    
    # Step 3: Clean medical prompt
    medical_prompt = f"""You are a clinical nutrition expert. Answer this question using the medical evidence provided.

QUESTION: {question}

MEDICAL EVIDENCE:
{context}

Instructions:
- Provide a clear, direct answer using the information above
- Include specific values, ranges, and units when available
- Use natural language without excessive formatting
- If the evidence doesn't contain the answer, state this clearly

Answer:"""
    
    # Step 4: Generate answer
    answer = query_ollama(medical_prompt, model=model, max_tokens=400)
    
    search_engine.close()
    
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "model_used": model,
        "search_type": "GraphRAG",
        "entities_found": len(search_results)
    }


def test_single_question():
    """Test with a specific electrolyte question."""
    question = "What is the normal range for serum sodium?"
    print(f"ASPEN Medical GraphRAG Test")
    print("=" * 50)
    print(f"Question: {question}")
    print()
    
    result = graphrag_answer(question, model="gemma3:1b")
    
    print(f"Model: {result['model_used']}")
    print(f"Search: {result['search_type']}")
    print(f"Entities: {result['entities_found']}")
    print()
    print("Answer:")
    print(result['answer'])
    print()
    print("Evidence Used:")
    for source in result['sources']:
        value_info = f" = {source['value']} {source['unit']}" if source['value'] else ""
        print(f"  • {source['entity']}{value_info} ({source['type']})")
        print(f"    From: {source['section']}, Matches: {source['matches']}")
    print()


def check_graphrag_ready() -> bool:
    """Check if Neo4j GraphRAG has ASPEN medical data loaded."""
    try:
        search_engine = GraphRAGSearch()
        
        # Try to get collection stats
        stats = search_engine.get_collection_stats()
        total_entities = stats.get('total_chunks', 0)
        
        if total_entities > 0:
            print(f"ASPEN GraphRAG ready: {total_entities} medical entities from {stats.get('total_documents', 0)} documents")
            print(f"   {stats.get('total_relationships', 0)} relationships, {stats.get('entity_types', 0)} entity types")
            search_engine.close()
            return True
        else:
            print("ASPEN GraphRAG is empty. Please run:")
            print("   python clinical_focused_kg_builder.py")
            print("   This will build the knowledge graph from ASPEN Fluids & Electrolytes")
            search_engine.close()
            return False
            
    except Exception as e:
        print(f"GraphRAG not accessible: {e}")
        print("Please make sure Neo4j is running and run:")
        print("   python clinical_focused_kg_builder.py")
        return False


def main():
    """Interactive ASPEN GraphRAG CLI with model selection."""
    print("ASPEN Medical GraphRAG System")
    print("Fluids & Electrolytes Knowledge Graph + Ollama")
    print("=" * 60)
    
    # Step 1: Check GraphRAG (Neo4j)
    if not check_graphrag_ready():
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
    print(f"\nASPEN GraphRAG Ready! Ask questions about fluids, electrolytes, and clinical nutrition.")
    print(f"Using: {selected_model}")
    print(f"Knowledge: ASPEN Fluids & Electrolytes Course")
    print(f"Commands: 'test' for sample question, 'quit' to exit")
    print("=" * 60)
    
    while True:
        try:
            print()
            user_input = input("Question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nThanks for using ASPEN GraphRAG!")
                break
            
            if user_input.lower() == 'test':
                print()
                test_single_question_with_model(selected_model)
                continue
            
            if not user_input:
                continue
            
            print()
            result = graphrag_answer(user_input, model=selected_model)
            
            # Display results
            if 'answer' in result and result['answer']:
                print(f"\nAnswer:")
                print(f"{result['answer']}")
            else:
                print(f"\nNo answer generated.")
                print(f"   Debug: {result.keys()}")
            
            # Show evidence used
            if result.get('sources'):
                print(f"\nEvidence Used ({len(result['sources'])} sources):")
                for source in result['sources'][:3]:  # Show top 3 sources
                    entity = source['entity'][:50] + "..." if len(source['entity']) > 50 else source['entity']
                    value_clean = source['value'].replace('$', '').replace('\\mathrm{', '').replace('}', '').replace('\\', '') if source['value'] else ""
                    value_info = f" = {value_clean} {source['unit']}" if value_clean else ""
                    print(f"  • {entity}{value_info}")
                    print(f"    Source: {source['section']}")
                
                if len(result['sources']) > 3:
                    print(f"  • +{len(result['sources']) - 3} more sources")
            
            print(f"\nFound {result.get('entities_found', 0)} relevant entities")
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n\nThanks for using ASPEN GraphRAG!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again or type 'quit' to exit.")
            print("-" * 60)


def test_single_question_with_model(model: str):
    """Test with electrolyte question using selected model."""
    question = "What is the difference between hyponatremia and hypernatremia?"
    print(f"🧪 Testing ASPEN GraphRAG with: {question}")
    print(f"🤖 Using model: {model}")
    print()
    
    result = graphrag_answer(question, model=model)
    
    print(f"📊 Found: {result['entities_found']} ASPEN entities")
    print("\n💡 Answer:")
    print(result['answer'])
    print("\n📚 ASPEN Evidence:")
    for source in result['sources'][:3]:
        value_info = f" = {source['value']} {source['unit']}" if source['value'] else ""
        print(f"  • {source['entity']}{value_info} ({source['type']})")


if __name__ == "__main__":
    main()
