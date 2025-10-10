"""
Hybrid RAG Service: Combines Vector Search (ChromaDB) + Graph Search (Neo4j)
Following LangChain Best Practices for Neo4j Integration (2025)

NEW FEATURES (2025):
- Reranking (Cohere/Jina/Embeddings)
- Context Compression
- Query Decomposition  
- Answer Validation & Polish
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

# Advanced RAG Components
try:
    from .advanced_rag_components import (
        AdvancedRAGComponents,
        RerankingConfig,
        CompressionConfig,
        QueryDecompositionConfig,
        ValidationConfig
    )
    ADVANCED_RAG_AVAILABLE = True
except ImportError:
    print("⚠️  Advanced RAG components not available")
    ADVANCED_RAG_AVAILABLE = False
    AdvancedRAGComponents = None
    RerankingConfig = None
    CompressionConfig = None
    QueryDecompositionConfig = None
    ValidationConfig = None

# 2025 Advanced RAG Improvements
try:
    from .advanced_rag_2025 import AdvancedRAG2025, AdvancedRAG2025Config
    ADVANCED_RAG_2025_AVAILABLE = True
except ImportError:
    print("⚠️  Advanced RAG 2025 not available")
    ADVANCED_RAG_2025_AVAILABLE = False
    AdvancedRAG2025 = None
    AdvancedRAG2025Config = None


class HybridRAGService(RAGService):
    """Enhanced RAG with both vector and graph retrieval
    
    Following LangChain Best Practices (2025):
    - Uses Neo4jGraph for connection management
    - Uses GraphCypherQAChain for natural language to Cypher
    - Implements hybrid retrieval (vector + graph)
    - Advanced features: Reranking, Compression, Query Decomposition, Validation
    - Proper error handling and monitoring
    """
    
    def __init__(
        self,
        embedding_provider,
        vector_store,
        llm_provider,
        neo4j_uri: Optional[str] = None,  # DISABLED by default - set to bolt://localhost:7687 to enable
        neo4j_user: str = "neo4j",
        neo4j_password: str = "medicalpass123",
        # Advanced RAG configurations
        enable_reranking: bool = False,
        reranking_provider: str = "embeddings",  # cohere, jina, or embeddings
        enable_compression: bool = False,
        enable_query_decomposition: bool = False,
        enable_validation: bool = False,
        cohere_api_key: Optional[str] = None,
        jina_api_key: Optional[str] = None,
        # 2025 Advanced RAG Improvements
        enable_advanced_2025: bool = True,  # Enable cutting-edge 2025 features
        advanced_2025_config: Optional[Any] = None
    ):
        super().__init__(embedding_provider, vector_store, llm_provider)
        
        # Initialize Neo4j using LangChain's official Neo4jGraph class (BEST PRACTICE)
        self.neo4j_enabled = False
        self.graph = None
        self.cypher_chain = None
        
        # Neo4j is DISABLED by default - only enable if URI is explicitly provided
        if neo4j_uri is None:
            print("ℹ️  Neo4j DISABLED (vector search only)")
            print("   To enable: pass neo4j_uri='bolt://localhost:7687' to HybridRAGService")
        elif not LANGCHAIN_NEO4J_AVAILABLE:
            print("⚠️  LangChain Neo4j integration not available - vector search only")
        else:
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
        
        # Initialize Advanced RAG Components (NEW - LangChain 2025 Best Practices)
        self.advanced_rag = None
        if ADVANCED_RAG_AVAILABLE:
            try:
                # Create configurations
                rerank_config = RerankingConfig(
                    enabled=enable_reranking,
                    provider=reranking_provider,
                    top_n=15,
                    initial_k=50,
                    cohere_api_key=cohere_api_key,
                    jina_api_key=jina_api_key
                )
                
                compression_config = CompressionConfig(
                    enabled=enable_compression,
                    method="embeddings",  # Can be "llm" or "embeddings"
                    similarity_threshold=0.76
                )
                
                decomposition_config = QueryDecompositionConfig(
                    enabled=enable_query_decomposition,
                    num_queries=3,
                    use_fusion=True
                )
                
                validation_config = ValidationConfig(
                    enabled=enable_validation,
                    check_sources=True,
                    polish_answer=True
                )
                
                # Initialize advanced components
                self.advanced_rag = AdvancedRAGComponents(
                    llm_provider=llm_provider,
                    embedding_provider=embedding_provider,
                    rerank_config=rerank_config,
                    compression_config=compression_config,
                    decomposition_config=decomposition_config,
                    validation_config=validation_config
                )
                
            except Exception as e:
                print(f"⚠️  Failed to initialize advanced RAG components: {e}")
                self.advanced_rag = None
        
        # Initialize 2025 Advanced RAG Improvements (NEW - Cutting Edge)
        self.advanced_2025 = None
        if ADVANCED_RAG_2025_AVAILABLE and enable_advanced_2025:
            try:
                config_2025 = advanced_2025_config or AdvancedRAG2025Config()
                
                self.advanced_2025 = AdvancedRAG2025(
                    llm_provider=llm_provider,
                    embedding_provider=embedding_provider,
                    config=config_2025
                )
                
            except Exception as e:
                print(f"⚠️  Failed to initialize 2025 advanced RAG: {e}")
                self.advanced_2025 = None
    
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
        """Hybrid search: BM25 + Vector + Multi-Query + HyDE (LangChain Best Practices 2025)
        
        ENABLED FEATURES (following LangChain):
        1. Multi-Query Retrieval (2-3 variants + original)
        2. BM25 + Vector Hybrid (keyword + semantic)
        3. HyDE (concise hypothetical answers)
        4. RRF Fusion (combine multiple searches)
        5. Automatic Deduplication
        
        DISABLED FEATURES:
        - Cross-Encoder Reranking (was using wrong model)
        - Adaptive Retrieval (use fixed limit instead)
        - Query Decomposition (legacy, not needed)
        """
        
        print(f"🔍 LangChain Advanced RAG (Multi-Query + BM25 + HyDE)")
        
        # Use fixed limit from settings (not adaptive)
        target_limit = query.limit or 10
        
        # STEP 1: Multi-Query Generation (LangChain best practice)
        queries_to_search = [query.query]
        if self.advanced_2025 and self.advanced_2025.config.enable_multi_query:
            try:
                queries_to_search = await self.advanced_2025.multi_query_generation(query.query)
            except Exception as e:
                print(f"⚠️  Multi-query generation failed: {e}")
        
        # STEP 2: HyDE - Add hypothetical answer as additional query
        hyde_query = None
        if self.advanced_2025 and self.advanced_2025.config.enable_hyde:
            try:
                hyde_query = await self.advanced_2025.generate_hyde_hypothesis_concise(query.query)
                if hyde_query:
                    queries_to_search.append(hyde_query)
            except Exception as e:
                print(f"⚠️  HyDE generation failed: {e}")
        
        # STEP 3: Search with all queries (Vector + BM25 for each)
        all_ranked_lists = []  # For RRF fusion
        
        for i, q in enumerate(queries_to_search, 1):
            print(f"  🔎 Query variant {i}/{len(queries_to_search)}: {q[:60]}...")
            
            # 3A: Vector Search (semantic)
            sub_query = SearchQuery(
                query=q,
                limit=target_limit * 2,  # Get more for fusion
                filters=query.filters
            )
            vector_results = await super().search(sub_query)
            
            # 3B: BM25 Search (keyword) - if enabled and available
            bm25_results = []
            if self.advanced_2025 and self.advanced_2025.config.enable_bm25_hybrid:
                try:
                    # Get all chunks for BM25 search
                    all_chunks_query = SearchQuery(query=q, limit=1000, filters=query.filters)
                    all_chunks_response = await super().search(all_chunks_query)
                    bm25_results = await self.advanced_2025.bm25_search(
                        query=q,
                        all_chunks=all_chunks_response.results,
                        top_k=target_limit * 2
                    )
                except Exception as e:
                    print(f"⚠️  BM25 search failed: {e}")
            
            # 3C: Combine vector + BM25 results for this query
            combined_results = list(vector_results.results) + bm25_results
            all_ranked_lists.append(combined_results)
        
        # STEP 4: RRF Fusion - Combine all searches with deduplication
        if self.advanced_2025 and self.advanced_2025.config.enable_rrf and len(all_ranked_lists) > 1:
            try:
                fused_results = await self.advanced_2025.reciprocal_rank_fusion(all_ranked_lists)
                unique_results = fused_results[:target_limit]
                print(f"✅ RRF fused {len(all_ranked_lists)} searches → {len(unique_results)} unique chunks")
            except Exception as e:
                print(f"⚠️  RRF fusion failed: {e}")
                # Fallback to simple deduplication
                seen_content = set()
                unique_results = []
                for ranked_list in all_ranked_lists:
                    for result in ranked_list:
                        content_hash = hash(result.content[:200])
                        if content_hash not in seen_content:
                            seen_content.add(content_hash)
                            unique_results.append(result)
                            if len(unique_results) >= target_limit:
                                break
                    if len(unique_results) >= target_limit:
                        break
        else:
            # Simple case: just use first search results
            unique_results = all_ranked_lists[0][:target_limit] if all_ranked_lists else []
        
        # Create SearchResponse
        from ..models.documents import SearchResponse
        final_response = SearchResponse(
            query=query.query,
            results=unique_results,
            total_results=sum(len(lst) for lst in all_ranked_lists),
            search_time_ms=0,  # TODO: track timing
            model_used="hybrid_bm25_vector_multi_query"
        )
        
        # STEP 5: Neo4j Graph Search (if enabled - disabled by default)
        if self.neo4j_enabled and hasattr(self, '_last_extracted_entities'):
            entities = self._last_extracted_entities
            
            if entities:
                print(f"🔍 Graph search for entities: {entities[:3]}")
                graph_results = self.query_knowledge_graph(entities, query.query)
                
                if graph_results:
                    graph_context_parts = []
                    for r in graph_results[:10]:
                        if r.get('result_type') == 'clinical_section':
                            graph_context_parts.append(
                                f"[Neo4j: {r['source']} - {r['name']}]\n{r.get('content', '')}"
                            )
                        else:
                            graph_context_parts.append(
                                f"- {r['name']} ({r['relationship']}) [{r.get('type', 'unknown')}] from {r.get('source', 'unknown')}"
                            )
                    
                    self._graph_context = "\n\n".join(graph_context_parts)
                    self._graph_result_count = len(graph_results)
                    print(f"✅ Graph enhanced: {len(graph_results)} results from Neo4j")
                else:
                    self._graph_context = None
                    self._graph_result_count = 0
            else:
                self._graph_context = None
                self._graph_result_count = 0
        else:
            self._graph_context = None
            self._graph_result_count = 0
        
        return final_response
    
    async def ask(self, rag_query: RAGQuery) -> RAGResponse:
        """Answer with hybrid retrieval + validation + polish
        
        NEW FEATURES:
        - Answer validation (if enabled)
        - Answer polishing (if enabled)
        """
        
        # Enhanced search with graph
        response = await super().ask(rag_query)
        
        # Add graph context if available
        graph_metadata = {}
        if hasattr(self, '_graph_context') and self._graph_context:
            graph_metadata = {
                'graph_entities': self._graph_context,
                'retrieval_type': 'hybrid_vector_graph'
            }
        
        # STEP 1: Validate answer (if enabled)
        validation_result = None
        if self.advanced_rag and self.advanced_rag.validation_config.enabled:
            try:
                sources = [source.content for source in response.sources[:10]]
                validation_result = await self.advanced_rag.validate_answer(
                    question=rag_query.question,
                    answer=response.answer,
                    sources=sources
                )
                
                print(f"  ✅ Answer validation: {validation_result['validation_status']}")
                graph_metadata['validation'] = validation_result
                
            except Exception as e:
                print(f"⚠️  Validation failed: {e}")
        
        # STEP 2: Polish answer (if enabled and validated)
        if (self.advanced_rag and 
            self.advanced_rag.validation_config.enabled and 
            self.advanced_rag.validation_config.polish_answer):
            
            if validation_result is None or validation_result.get('is_valid', True):
                try:
                    sources = [source.content for source in response.sources[:10]]
                    polished_answer = await self.advanced_rag.polish_answer(
                        question=rag_query.question,
                        answer=response.answer,
                        sources=sources
                    )
                    
                    # Update response with polished answer
                    response.answer = polished_answer
                    
                except Exception as e:
                    print(f"⚠️  Polishing failed: {e}")
        
        # Update metadata
        if graph_metadata:
            response.metadata = graph_metadata
        
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
