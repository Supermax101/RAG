"""
Hybrid RAG Service: Combines Vector Search (ChromaDB) + Graph Search (Neo4j)
"""

from typing import List, Dict, Any
from neo4j import GraphDatabase
from ..models.documents import SearchQuery, SearchResult, RAGQuery, RAGResponse
from .rag_service import RAGService


class HybridRAGService(RAGService):
    """Enhanced RAG with both vector and graph retrieval"""
    
    def __init__(
        self,
        embedding_provider,
        vector_store,
        llm_provider,
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="yourpassword"
    ):
        super().__init__(embedding_provider, vector_store, llm_provider)
        
        # Initialize Neo4j connection
        try:
            self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            self.neo4j_enabled = True
            print("✅ Neo4j connected for hybrid retrieval")
        except Exception as e:
            print(f"⚠️  Neo4j not available: {e}")
            self.neo4j_driver = None
            self.neo4j_enabled = False
    
    def query_knowledge_graph(self, entities: List[str]) -> List[Dict[str, Any]]:
        """Query Neo4j graph for related entities and relationships"""
        
        if not self.neo4j_enabled:
            return []
        
        graph_results = []
        
        with self.neo4j_driver.session() as session:
            for entity in entities[:3]:  # Limit to top 3 entities
                # Find related entities within 2 hops
                query = """
                MATCH path = (e:Entity {name: $entity})-[*1..2]-(related:Entity)
                RETURN related.name as entity, 
                       type(relationships(path)[0]) as relationship,
                       related.type as entity_type,
                       related.source_doc as source
                LIMIT 10
                """
                
                try:
                    result = session.run(query, entity=entity)
                    for record in result:
                        graph_results.append({
                            'entity': record['entity'],
                            'relationship': record['relationship'],
                            'type': record['entity_type'],
                            'source': record['source']
                        })
                except Exception as e:
                    print(f"Graph query failed for {entity}: {e}")
        
        return graph_results
    
    async def search(self, query: SearchQuery) -> Any:
        """Hybrid search: Vector + Graph"""
        
        # Step 1: Standard vector search
        vector_results = await super().search(query)
        
        # Step 2: If Neo4j enabled, enhance with graph search
        if self.neo4j_enabled and hasattr(self, '_last_extracted_entities'):
            entities = self._last_extracted_entities
            
            if entities:
                print(f"🔍 Graph search for entities: {entities[:3]}")
                graph_results = self.query_knowledge_graph(entities)
                
                if graph_results:
                    # Add graph context to metadata
                    graph_context = "\n".join([
                        f"- {r['entity']} ({r['relationship']}) [{r['type']}]"
                        for r in graph_results[:5]
                    ])
                    
                    print(f"✅ Graph enhanced: {len(graph_results)} related entities found")
                    
                    # Store for context building
                    self._graph_context = graph_context
        
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
        """Close Neo4j connection"""
        if self.neo4j_driver:
            self.neo4j_driver.close()
