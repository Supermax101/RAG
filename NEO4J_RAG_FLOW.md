# 🔄 Neo4j Knowledge Graph - Complete RAG Flow

**Question:** Does the model have access to Neo4j during the RAG process?  
**Answer:** ✅ YES! Neo4j is queried BEFORE the LLM sees the prompt. Here's the complete flow:

---

## 📊 COMPLETE HYBRID RAG PIPELINE

```
┌─────────────────────────────────────────────────────────────────┐
│  USER ASKS QUESTION                                             │
│  "What is the recommended TPN dose for neonatal patients?"      │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: ENTITY EXTRACTION (LangGraph Workflow)                 │
│  File: src/rag/core/services/rag_service.py                     │
│  Method: enhanced_tpn_search() → extract_query_er()             │
│                                                                  │
│  Extracts:                                                       │
│  - Patient entities: ["neonatal", "preterm"]                    │
│  - Component entities: ["TPN", "amino acids", "dextrose"]       │
│  - Clinical entities: ["dosing", "recommendation"]              │
│                                                                  │
│  Stores: self._last_extracted_entities = ["neonatal", "TPN", ...]│
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: CHROMADB VECTOR SEARCH                                 │
│  File: src/rag/infrastructure/vector_stores/chroma_store.py     │
│                                                                  │
│  Searches 52 TPN PDFs (embedded in ChromaDB):                   │
│  - Generates query embedding                                     │
│  - Finds top 15 most similar chunks                             │
│  - Returns: 15 text chunks from medical PDFs                    │
│                                                                  │
│  Example Results:                                                │
│  [1] "ASPEN Guidelines - Neonatal TPN should start with..."     │
│  [2] "Preterm amino acid dosing: 3.5-4 g/kg/day..."            │
│  ...                                                             │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: NEO4J KNOWLEDGE GRAPH SEARCH                           │
│  File: src/rag/core/services/hybrid_rag_service.py              │
│  Method: query_knowledge_graph()                                │
│                                                                  │
│  Uses extracted entities to query Neo4j:                        │
│                                                                  │
│  Strategy 1: Entity Relationship Traversal                      │
│  CYPHER: MATCH (e:Entity)-[r*1..2]-(related:Entity)            │
│          WHERE toLower(e.name) CONTAINS 'neonatal'              │
│  Finds: Related medical concepts, relationships                 │
│                                                                  │
│  Strategy 2: Clinical Section Search                            │
│  CYPHER: MATCH (cs:ClinicalSection)                             │
│          WHERE cs.content =~ '(?i).*neonatal.*TPN.*'            │
│  Finds: Relevant clinical guideline sections                    │
│                                                                  │
│  Strategy 3: Fallback Search                                    │
│  Gets most relevant clinical sections if others fail            │
│                                                                  │
│  Returns: 5-10 graph results with clinical content              │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: CONTEXT MERGING                                        │
│  File: eval/tpn_rag_evaluation.py                               │
│  Lines: 353-369                                                 │
│                                                                  │
│  Builds unified context string:                                 │
│                                                                  │
│  [ChromaDB Vector 1: ASPEN_Neonatal_TPN]                        │
│  Content from PDF chunk 1...                                    │
│                                                                  │
│  [ChromaDB Vector 2: 2018_NICU_Guide]                           │
│  Content from PDF chunk 2...                                    │
│                                                                  │
│  ... (15 total vector results) ...                              │
│                                                                  │
│  --- KNOWLEDGE GRAPH RELATIONSHIPS (Neo4j) ---                  │
│                                                                  │
│  [Neo4j: ASPEN Guidelines - Neonatal Dosing]                    │
│  Neonatal TPN amino acid dosing starts at 3.5 g/kg/day...      │
│                                                                  │
│  [Neo4j: 2022 Fellow Guide - TPN Components]                    │
│  Preterm infants require careful protein titration...           │
│                                                                  │
│  ... (5-10 graph results) ...                                   │
│                                                                  │
│  Total Context: ~8000 tokens (vector + graph)                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: LLM PROMPT CONSTRUCTION                                │
│  File: eval/tpn_rag_evaluation.py                               │
│  Method: build_mcq_prompt_template()                            │
│                                                                  │
│  System Prompt:                                                 │
│  "You are evaluating a RAG system for TPN guidelines.           │
│   Answer STRICTLY ONLY based on retrieved sources below.        │
│   DO NOT use prior medical knowledge from training data."       │
│                                                                  │
│  Retrieved Context: [ALL MERGED CONTENT FROM STEP 4]            │
│                                                                  │
│  Question: "What is the recommended TPN dose for neonatal       │
│            patients? A) 2 g/kg/day B) 3.5 g/kg/day C) 5 g/kg/day"│
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6: LLM GENERATION                                         │
│  File: eval/tpn_rag_evaluation.py                               │
│  Line: 376-381                                                  │
│                                                                  │
│  Model receives FULL CONTEXT including:                         │
│  ✅ 15 ChromaDB vector search results                           │
│  ✅ 5-10 Neo4j knowledge graph results                          │
│  ✅ Strict instructions to ONLY use provided sources            │
│                                                                  │
│  Model thinks: "Based on the retrieved guidelines,              │
│                 neonatal TPN dosing is 3.5 g/kg/day..."         │
│                                                                  │
│  Generates: {"answer": "B", "confidence": "high"}               │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 7: EVALUATION & METRICS                                   │
│  File: eval/tpn_rag_evaluation.py                               │
│                                                                  │
│  Tracks:                                                         │
│  - num_sources: 15 (ChromaDB chunks)                            │
│  - num_graph_results: 8 (Neo4j results)                         │
│  - graph_used: True                                             │
│  - is_correct: True                                             │
│                                                                  │
│  Console Output:                                                 │
│  "✅ Added Neo4j graph context (2450 chars)"                    │
│  "Expected: B, Got: B (high confidence) - CORRECT [+8 graph]"   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔍 KEY POINTS

### 1. **Neo4j is Queried BEFORE the LLM**
The knowledge graph search happens in **STEP 3**, BEFORE the prompt is sent to the LLM. The LLM never directly queries Neo4j - it receives the graph results as part of the context.

### 2. **Three Data Sources in the Prompt**
When the LLM generates an answer, it sees:
1. ✅ **ChromaDB Vector Results** - Semantic similarity search (15 chunks)
2. ✅ **Neo4j Graph Results** - Entity relationships + clinical sections (5-10 results)
3. ✅ **Strict Instructions** - "ONLY use the retrieved sources, NOT your training data"

### 3. **Automatic Entity Extraction**
The system automatically:
- Extracts medical entities from the question (neonatal, TPN, dosing)
- Uses those entities to query the knowledge graph
- Finds related clinical guidelines and relationships
- Merges everything into a single context

### 4. **Hybrid RAG Architecture**
```
Question → Entity Extraction
           ↓
       Vector Search (ChromaDB) ─┐
           +                      ├→ Merged Context → LLM → Answer
       Graph Search (Neo4j) ──────┘
```

---

## 📝 CODE EVIDENCE

### Entity Extraction & Storage (rag_service.py:113-126)
```python
# FIX BUG #1: Store extracted entities for HybridRAGService to use
if er_data and er_data.get("entities"):
    entity_list = []
    for entity_type, entity_values in er_data["entities"].items():
        if isinstance(entity_values, dict):
            entity_list.extend([v for v in entity_values.values() if v])
        elif isinstance(entity_values, list):
            entity_list.extend(entity_values)
    
    self._last_extracted_entities = entity_list
    print(f"📊 Extracted entities: {entity_list[:5]}")
```

### Graph Search Execution (hybrid_rag_service.py:225-253)
```python
if self.neo4j_enabled and hasattr(self, '_last_extracted_entities'):
    entities = self._last_extracted_entities
    
    if entities:
        print(f"🔍 Graph search for entities: {entities[:3]}")
        graph_results = self.query_knowledge_graph(entities, query.query)
        
        if graph_results:
            # Build formatted graph context
            graph_context_parts = []
            for r in graph_results[:10]:
                if r.get('result_type') == 'clinical_section':
                    graph_context_parts.append(
                        f"[Neo4j: {r['source']} - {r['name']}]\n{r.get('content', '')}"
                    )
            
            graph_context = "\n\n".join(graph_context_parts)
            print(f"✅ Graph enhanced: {len(graph_results)} results from Neo4j")
            
            self._graph_context = graph_context
            self._graph_result_count = len(graph_results)
```

### Context Merging (tpn_rag_evaluation.py:353-369)
```python
# Vector search results (ChromaDB)
for i, result in enumerate(search_response.results, 1):
    doc_name = result.document_name[:50]
    context_parts.append(f"[ChromaDB Vector {i}: {doc_name}]\n{result.content}")

# FIX BUG #2: Merge Neo4j graph context if available
if hasattr(self.rag_service, '_graph_context') and self.rag_service._graph_context:
    context_parts.append("\n--- KNOWLEDGE GRAPH RELATIONSHIPS (Neo4j) ---")
    context_parts.append(self.rag_service._graph_context)
    print(f"  ✅ Added Neo4j graph context ({len(self.rag_service._graph_context)} chars)")

context = "\n\n".join(context_parts)
```

---

## ✅ VERIFICATION CHECKLIST

When you run the evaluation, you should see:

1. ✅ `📊 Extracted entities: ['neonatal', 'TPN', 'amino']`
2. ✅ `🔍 Graph search for entities: ['neonatal', 'TPN', 'amino']`
3. ✅ `✅ Graph enhanced: 8 results from Neo4j knowledge graph`
4. ✅ `✅ Added Neo4j graph context (2450 chars)`
5. ✅ `Expected: B, Got: B (high confidence) - CORRECT [+8 graph]`

In the JSON output:
```json
{
  "neo4j_graph_used": true,
  "questions_with_graph_context": 48,
  "total_graph_results": 384,
  "avg_graph_results_per_question": 8.0
}
```

---

## 🎯 ANSWER TO YOUR QUESTION

**Q: "Does the model have access to Neo4j during the RAG process?"**

**A: YES! Here's how:**

1. **Before prompting the model**, the system:
   - Extracts medical entities from the question
   - Queries ChromaDB for similar text chunks (vector search)
   - Queries Neo4j for related clinical content (graph search)
   - Merges both into a single context string

2. **The model receives** a prompt containing:
   - 15 text chunks from ChromaDB (52 medical PDFs)
   - 5-10 clinical sections from Neo4j (knowledge graph)
   - Instructions to ONLY use these provided sources

3. **The model never directly queries Neo4j** - it doesn't need to! All the graph knowledge is already in the prompt context.

This is the **standard RAG pattern**: Retrieve → Augment → Generate

The "Retrieve" step includes BOTH vector search AND graph search, making it true **Hybrid RAG**! 🚀

---

## 📚 RESOURCES

- LangChain Neo4j Integration: https://python.langchain.com/docs/integrations/graphs/neo4j_cypher/
- Neo4j + LangChain Best Practices: https://neo4j.com/labs/genai-ecosystem/langchain/
- Hybrid RAG Architecture: Combines dense retrieval (vectors) with structured retrieval (graphs)

