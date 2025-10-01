"""
Hybrid RAG Service: Combines Vector Search (ChromaDB) + Graph Search (Neo4j)
Following LangChain Best Practices for Neo4j Integration (2025)
"""

from typing import List, Dict, Any, Optional
from ..models.documents import SearchQuery, SearchResult, RAGQuery, RAGResponse
from .rag_service import RAGService

# LangChain Official Neo4j Integration (Best Practice)
try:
    from langchain_community.graphs import Neo4jGraph
    from langchain.chains import GraphCypherQAChain
    LANGCHAIN_NEO4J_AVAILABLE = True
except ImportError:
    print("⚠️  LangChain Neo4j components not available. Install: pip install langchain-community neo4j")
    LANGCHAIN_NEO4J_AVAILABLE = False
    Neo4jGraph = None
    GraphCypherQAChain = None


class HybridRAGService(RAGService):
    """Enhanced RAG with both vector and graph retrieval
    
    Following LangChain Best Practices:
    - Uses Neo4jGraph for connection management
    - Uses GraphCypherQAChain for natural language to Cypher
    - Implements hybrid retrieval (vector + graph)
    - Proper error handling and monitoring
    """
    
    def __init__(
        self,
        embedding_provider,
        vector_store,
        llm_provider,
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="medicalpass123"
    ):
        super().__init__(embedding_provider, vector_store, llm_provider)
        
        # Initialize Neo4j using LangChain's official Neo4jGraph class (BEST PRACTICE)
        self.neo4j_enabled = False
        self.graph = None
        self.cypher_chain = None
        
        if not LANGCHAIN_NEO4J_AVAILABLE:
            print("⚠️  LangChain Neo4j integration not available - vector search only")
            return
        
        try:
            # LangChain Best Practice: Use Neo4jGraph for connection
            self.graph = Neo4jGraph(
                url=neo4j_uri,
                username=neo4j_user,
                password=neo4j_password,
                database="neo4j"  # Explicit database name
            )
            
            # Test connection and get schema
            schema = self.graph.get_schema
            print("✅ Neo4j connected for hybrid retrieval (LangChain Neo4jGraph)")
            print(f"📊 Graph schema: {len(str(schema))} chars")
            
            self.neo4j_enabled = True
            
            # Create indexes for performance (Best Practice)
            self._create_performance_indexes()
            
        except Exception as e:
            print(f"⚠️  Neo4j not available: {e}")
            print("   Falling back to vector search only")
            self.graph = None
            self.neo4j_enabled = False
    
    def _create_performance_indexes(self):
        """Create indexes for query performance (Best Practice)"""
        if not self.graph:
            return
        
        try:
            # Create indexes for Entity nodes
            self.graph.query("""
                CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)
            """)
            
            # Create indexes for ClinicalSection nodes
            self.graph.query("""
                CREATE INDEX clinical_section_name_index IF NOT EXISTS FOR (cs:ClinicalSection) ON (cs.name)
            """)
            
            self.graph.query("""
                CREATE INDEX clinical_section_doc_index IF NOT EXISTS FOR (cs:ClinicalSection) ON (cs.doc_source)
            """)
            
            print("  ✅ Created performance indexes")
        except Exception as e:
            print(f"  ⚠️  Index creation skipped: {e}")
    
    def query_knowledge_graph(self, entities: List[str], query_text: str = "") -> List[Dict[str, Any]]:
        """Query Neo4j graph using optimized Cypher queries (LangChain Best Practice)
        
        Supports both Entity and ClinicalSection node types
        Uses indexed queries for performance
        """
        
        if not self.neo4j_enabled or not self.graph:
            return []
        
        graph_results = []
        
        # Strategy 1: Entity Relationship Traversal (Multi-hop)
        if entities:
            for entity in entities[:3]:  # Limit to top 3 entities
                try:
                    # Optimized Cypher query with index usage
                    entity_cypher = """
                    MATCH path = (e:Entity)-[r*1..2]-(related:Entity)
                    WHERE toLower(e.name) CONTAINS toLower($entity) 
                       OR toLower(related.name) CONTAINS toLower($entity)
                    WITH related, relationships(path) as rels, e
                    RETURN DISTINCT related.name as name, 
                           CASE WHEN size(rels) > 0 THEN type(rels[0]) ELSE 'RELATED_TO' END as relationship,
                           related.type as entity_type,
                           related.source_doc as source
                    LIMIT 5
                    """
                    
                    result = self.graph.query(entity_cypher, params={"entity": entity})
                    
                    for record in result:
                        graph_results.append({
                            'name': record.get('name', ''),
                            'relationship': record.get('relationship', 'RELATED_TO'),
                            'type': record.get('entity_type', 'unknown'),
                            'source': record.get('source', 'unknown'),
                            'result_type': 'entity_relationship'
                        })
                except Exception as e:
                    print(f"  Entity query failed for '{entity}': {e}")
        
        # Strategy 2: Clinical Section Content Search (Full-text)
        if query_text or entities:
            try:
                search_terms = entities[:3] if entities else [query_text.split()[0]]
                
                for term in search_terms:
                    # Optimized Cypher with index on name and doc_source
                    clinical_cypher = """
                    MATCH (cs:ClinicalSection)
                    WHERE toLower(cs.name) CONTAINS toLower($term)
                       OR toLower(cs.doc_source) CONTAINS toLower($term)
                       OR toLower(cs.content) CONTAINS toLower($term)
                    RETURN cs.name as name,
                           cs.doc_source as source,
                           cs.section_type as type,
                           substring(cs.content, 0, 400) as content
                    ORDER BY size(cs.content) DESC
                    LIMIT 3
                    """
                    
                    result = self.graph.query(clinical_cypher, params={"term": term})
                    
                    for record in result:
                        content_text = record.get('content', '')
                        if len(content_text) > 300:
                            content_text = content_text[:300] + "..."
                        
                        graph_results.append({
                            'name': record.get('name', ''),
                            'relationship': 'clinical_guideline',
                            'type': record.get('type', 'clinical'),
                            'source': record.get('source', 'unknown'),
                            'content': content_text,
                            'result_type': 'clinical_section'
                        })
                except Exception as e:
                    print(f"  ClinicalSection query failed: {e}")
        
        # Strategy 3: Graph Semantic Search (if no results yet)
        if not graph_results and query_text:
            try:
                # Fallback: Get most relevant clinical sections by document source
                fallback_cypher = """
                MATCH (cs:ClinicalSection)
                RETURN cs.name as name,
                       cs.doc_source as source,
                       cs.section_type as type,
                       substring(cs.content, 0, 400) as content
                ORDER BY cs.order_index
                LIMIT 5
                """
                
                result = self.graph.query(fallback_cypher)
                
                for record in result:
                    content_text = record.get('content', '')
                    if len(content_text) > 300:
                        content_text = content_text[:300] + "..."
                    
                    graph_results.append({
                        'name': record.get('name', ''),
                        'relationship': 'clinical_guideline',
                        'type': record.get('type', 'clinical'),
                        'source': record.get('source', 'unknown'),
                        'content': content_text,
                        'result_type': 'clinical_section_fallback'
                    })
            except Exception as e:
                print(f"  Fallback query failed: {e}")
        
        return graph_results
    
    async def search(self, query: SearchQuery) -> Any:
        """Hybrid search: Vector + Graph
        
        FIX BUG #2: Properly merge graph context into search results
        """
        
        # Step 1: Standard vector search
        vector_results = await super().search(query)
        
        # Step 2: If Neo4j enabled, enhance with graph search
        if self.neo4j_enabled and hasattr(self, '_last_extracted_entities'):
            entities = self._last_extracted_entities
            
            if entities:
                print(f"🔍 Graph search for entities: {entities[:3]}")
                graph_results = self.query_knowledge_graph(entities, query.query)
                
                if graph_results:
                    # Build formatted graph context
                    graph_context_parts = []
                    for r in graph_results[:10]:  # Use up to 10 graph results
                        if r.get('result_type') == 'clinical_section':
                            # For clinical sections, include the content snippet
                            graph_context_parts.append(
                                f"[Neo4j: {r['source']} - {r['name']}]\n{r.get('content', '')}"
                            )
                        else:
                            # For entity relationships
                            graph_context_parts.append(
                                f"- {r['name']} ({r['relationship']}) [{r.get('type', 'unknown')}] from {r.get('source', 'unknown')}"
                            )
                    
                    graph_context = "\n\n".join(graph_context_parts)
                    
                    print(f"✅ Graph enhanced: {len(graph_results)} results from Neo4j knowledge graph")
                    
                    # Store graph context AND graph result count for metrics
                    self._graph_context = graph_context
                    self._graph_result_count = len(graph_results)
                else:
                    print(f"⚠️  No graph results found for entities: {entities[:3]}")
                    self._graph_context = None
                    self._graph_result_count = 0
            else:
                print(f"⚠️  No entities extracted from query")
                self._graph_context = None
                self._graph_result_count = 0
        else:
            if not self.neo4j_enabled:
                print("⚠️  Neo4j disabled - vector search only")
            self._graph_context = None
            self._graph_result_count = 0
        
        return vector_results
    
    async def ask(self, rag_query: RAGQuery) -> RAGResponse:
        """Answer with hybrid retrieval"""
        
        # Enhanced search with graph
        response = await super().ask(rag_query)
        
        # Add graph context if available
        if hasattr(self, '_graph_context') and self._graph_context:
            # Append graph knowledge to answer
            response.metadata = {
                'graph_entities': self._graph_context,
                'retrieval_type': 'hybrid_vector_graph'
            }
        
        return response
    
    def close(self):
        """Close Neo4j connection (LangChain Best Practice)"""
        if self.graph:
            try:
                # LangChain Neo4jGraph handles connection cleanup internally
                # But we can explicitly close if needed
                if hasattr(self.graph, '_driver') and self.graph._driver:
                    self.graph._driver.close()
                print("✅ Neo4j connection closed")
            except Exception as e:
                print(f"⚠️  Error closing Neo4j: {e}")
