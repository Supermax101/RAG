"""
Unified TPN RAG service with enhanced search and ER extraction.
Modern LangChain/LangGraph integration for maximum accuracy.
"""
import time
import asyncio
import json
from typing import List, Optional, Dict, Any, Tuple
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel as LangChainBaseModel, Field as LangChainField
from langgraph.graph import StateGraph, END

from ..models.documents import (
    SearchQuery, SearchResponse, SearchResult, DocumentChunk,
    RAGQuery, RAGResponse
)
from ..interfaces.embeddings import EmbeddingProvider, VectorStore, LLMProvider


class TPNAnswer(LangChainBaseModel):
    """Structured output for TPN clinical answers."""
    answer: str = LangChainField(description="Clinical answer based on TPN guidelines")
    confidence: str = LangChainField(default="medium", description="Confidence level: low, medium, high")
    sources_used: List[int] = LangChainField(default_factory=list, description="List of source indices used")


class RAGService:
    """Modern TPN RAG service with LangChain LCEL chains and LangGraph workflows."""
    
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        llm_provider: LLMProvider
    ):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        
        # Initialize enhanced search capabilities
        self.er_graph = self._build_er_extraction_graph()
        self.neo4j_fallback = None  # Initialize on demand
        
        # Initialize LangChain components
        self.prompt_template = self._build_tpn_prompt_template()
        self.few_shot_examples = self._get_few_shot_examples()
        self.output_parser = StrOutputParser()
    
    async def search(self, query: SearchQuery) -> SearchResponse:
        """Perform enhanced TPN-focused search with ER extraction."""
        
        # Check if enhanced search is requested
        if query.filters.get("enhanced_search", True):
            return await self.enhanced_tpn_search(query)
        else:
            return await self.basic_search(query)
    
    async def basic_search(self, query: SearchQuery) -> SearchResponse:
        """Perform basic vector search (fallback)."""
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = await self.embedding_provider.embed_query(query.query)
        
        # Search vector store
        raw_results = await self.vector_store.search_similar(
            query_embedding,
            limit=query.limit,
            filters=query.filters
        )
        
        # Convert to SearchResult objects
        search_results = []
        for result in raw_results:
            chunk = DocumentChunk(
                chunk_id=result["chunk_id"],
                doc_id=result["doc_id"],
                content=result["content"],
                chunk_type=result.get("chunk_type", "text"),
                page_num=result.get("page_num"),
                section=result.get("section"),
                metadata=result.get("metadata", {})
            )
            
            search_result = SearchResult(
                chunk=chunk,
                score=result["score"],
                document_name=result.get("document_name", "Unknown")
            )
            search_results.append(search_result)
        
        search_time_ms = (time.time() - start_time) * 1000
        
        return SearchResponse(
            query=query,
            results=search_results,
            total_results=len(search_results),
            search_time_ms=search_time_ms,
            model_used=self.embedding_provider.model_name
        )
    
    async def enhanced_tpn_search(self, query: SearchQuery) -> SearchResponse:
        """Enhanced multi-strategy search with ER extraction."""
        
        start_time = time.time()
        
        print(f"🔍 Enhanced TPN Search: {query.query}")
        
        # Step 1: Extract entities and relationships
        er_data = await self.extract_query_er(query.query)
        
        # Step 2: Multi-strategy search
        all_results = []
        
        # Strategy 1: Original query search
        original_results = await self._search_with_strategy(query.query, "original", query.limit)
        all_results.extend(original_results)
        
        # Strategy 2: Enhanced query search
        if er_data.get("enhanced_query") and er_data["enhanced_query"] != query.query:
            enhanced_results = await self._search_with_strategy(er_data["enhanced_query"], "enhanced", query.limit)
            all_results.extend(enhanced_results)
        
        # Strategy 3: Entity-focused searches
        for search_term in er_data.get("search_terms", [])[:2]:  # Limit to top 2
            if search_term != query.query:
                entity_results = await self._search_with_strategy(search_term, "entity_focused", query.limit)
                all_results.extend(entity_results)
        
        # Strategy 4: Semantic expansion
        semantic_results = await self._semantic_expansion_search(query.query, query.limit)
        all_results.extend(semantic_results)
        
        # Step 3: Deduplicate and rank results
        final_results = self._deduplicate_and_rank(all_results, query.limit)
        
        search_time_ms = (time.time() - start_time) * 1000
        
        print(f"✅ Enhanced search complete: {len(final_results)} results in {search_time_ms:.1f}ms")
        
        return SearchResponse(
            query=query,
            results=final_results,
            total_results=len(all_results),
            search_time_ms=search_time_ms,
            model_used=f"enhanced_tpn_search_{self.embedding_provider.model_name}"
        )
    
    async def ask(self, rag_query: RAGQuery) -> RAGResponse:
        """Answer a question using modern LangChain RAG pipeline.
        
        Pipeline: Search → Context Building → Few-Shot Prompting → Generation
        """
        start_time = time.time()
        
        # Step 1: Enhanced search for relevant TPN documents
        search_query = SearchQuery(
            query=rag_query.question,
            limit=rag_query.search_limit
        )
        
        search_start = time.time()
        search_response = await self.search(search_query)
        search_time_ms = (time.time() - search_start) * 1000
        
        if not search_response.results:
            total_time_ms = (time.time() - start_time) * 1000
            return RAGResponse(
                question=rag_query.question,
                answer="I couldn't find relevant TPN clinical guidelines to answer your question. Please rephrase or ask about specific TPN topics (dosing, monitoring, complications, etc.).",
                sources=[],
                search_time_ms=search_time_ms,
                generation_time_ms=0,
                total_time_ms=total_time_ms,
                model_used="no-model"
            )
        
        # Step 2: Build enriched context with source metadata
        context = self._build_context_with_metadata(search_response.results)
        
        # Step 3: Format prompt using LangChain ChatPromptTemplate
        # This includes: system message + few-shot examples + current question
        formatted_messages = self.prompt_template.format_messages(
            context=context,
            question=rag_query.question
        )
        
        # Convert LangChain messages to string for Ollama
        # (Ollama doesn't support chat format directly via our wrapper)
        prompt_str = self._format_messages_for_ollama(formatted_messages)
        
        # Step 4: Generate answer with LLM
        generation_start = time.time()
        answer = await self.llm_provider.generate(
            prompt=prompt_str,
            model=rag_query.model,
            temperature=rag_query.temperature,
            max_tokens=600  # Increased for detailed clinical answers
        )
        generation_time_ms = (time.time() - generation_start) * 1000
        
        total_time_ms = (time.time() - start_time) * 1000
        
        return RAGResponse(
            question=rag_query.question,
            answer=answer.strip(),
            sources=search_response.results,
            search_time_ms=search_time_ms,
            generation_time_ms=generation_time_ms,
            total_time_ms=total_time_ms,
            model_used=rag_query.model or "default"
        )
    
    def _format_messages_for_ollama(self, messages: List[Any]) -> str:
        """Convert LangChain messages to string format for Ollama."""
        formatted_parts = []
        
        for msg in messages:
            role = msg.__class__.__name__.replace("Message", "").upper()
            if role == "SYSTEM":
                formatted_parts.append(f"SYSTEM INSTRUCTIONS:\n{msg.content}\n")
            elif role == "HUMAN":
                formatted_parts.append(f"USER:\n{msg.content}\n")
            elif role == "AI":
                formatted_parts.append(f"ASSISTANT:\n{msg.content}\n")
            else:
                formatted_parts.append(f"{msg.content}\n")
        
        formatted_parts.append("\nASSISTANT:")
        return "\n".join(formatted_parts)
    
    def _get_few_shot_examples(self) -> List[Dict[str, str]]:
        """Get few-shot examples for TPN clinical Q&A."""
        return [
            {
                "context": "[Source 1] Neonatal TPN should start with dextrose 10% at 4-6 mg/kg/min GIR.",
                "question": "What is the recommended starting dextrose concentration for neonatal TPN?",
                "answer": "Based on the ASPEN guidelines, neonatal TPN should start with dextrose 10% at a glucose infusion rate (GIR) of 4-6 mg/kg/min."
            },
            {
                "context": "[Source 1] Potassium levels should be monitored daily in TPN patients, especially during initiation.",
                "question": "How often should potassium be monitored in TPN patients?",
                "answer": "According to the guidelines, potassium levels should be monitored daily in TPN patients, particularly during the initiation phase."
            }
        ]
    
    def _build_tpn_prompt_template(self) -> ChatPromptTemplate:
        """Build modern LangChain prompt template with few-shot examples."""
        
        # Example prompt for few-shot learning
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "Context from TPN guidelines:\n{context}\n\nQuestion: {question}"),
            ("ai", "{answer}")
        ])
        
        # Few-shot prompt
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=self._get_few_shot_examples(),
        )
        
        # Final prompt template with system message, few-shot examples, and current question
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a TPN (Total Parenteral Nutrition) Clinical Specialist with expertise in ASPEN guidelines.

Your role:
- Answer questions based ONLY on the provided TPN clinical guidelines
- Cite specific sources when making recommendations
- If information is insufficient, clearly state what's missing
- Use clinical terminology appropriately
- Provide precise dosing, monitoring, and safety information

Remember: Patient safety depends on accurate information from authoritative sources."""),
            few_shot_prompt,
            ("human", """Based on the following TPN clinical guidelines, answer the question.

{context}

QUESTION: {question}

Provide a clear, evidence-based answer using the guidelines above.""")
        ])
        
        return final_prompt
    
    def _build_context_with_metadata(self, results: List[SearchResult]) -> str:
        """Build enriched context with source metadata for better attribution."""
        context_parts = []
        
        for i, result in enumerate(results, 1):
            # Extract metadata
            doc_name = result.document_name[:60]
            section = result.chunk.section or "General"
            page = f", Page {result.chunk.page_num}" if result.chunk.page_num else ""
            
            # Format with metadata
            context_parts.append(
                f"[Source {i}: {doc_name}{page}]\n"
                f"Section: {section}\n"
                f"{result.content}"
            )
        
        return "\n\n".join(context_parts)
    
    def _build_rag_prompt(self, question: str, context: str) -> str:
        """Build RAG prompt for answer generation (legacy fallback)."""
        return f"""Answer the following question using only the provided context. Be precise and helpful.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer based only on the provided context
- If the context doesn't contain enough information, say so clearly
- Keep the answer focused and concise
- Use specific details from the context when available

ANSWER:"""
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection."""
        return await self.vector_store.get_stats()
    
    async def add_document_chunks(
        self,
        chunks: List[DocumentChunk],
        doc_name: str
    ) -> None:
        """Add new document chunks to the vector store with batch processing."""
        if not chunks:
            return
        
        # Process in batches to avoid memory issues
        batch_size = 50  # Process 50 chunks at a time
        
        print(f"🔄 Processing {len(chunks)} chunks in batches of {batch_size}...")
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_texts = [chunk.content for chunk in batch_chunks]
            
            # Generate embeddings for batch
            print(f"  📊 Generating embeddings for batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
            batch_embeddings = await self.embedding_provider.embed_texts(batch_texts)
            
            # Add batch to vector store
            await self.vector_store.add_chunks(batch_chunks, batch_embeddings, doc_name)
            
        print(f"✅ Successfully processed all {len(chunks)} chunks")
    
    async def remove_document(self, doc_id: str) -> None:
        """Remove a document from the vector store."""
        await self.vector_store.delete_document(doc_id)
    
    # ========== Enhanced Search and ER Extraction Methods ==========
    
    def _build_er_extraction_graph(self) -> StateGraph:
        """Build LangGraph workflow for ER extraction."""
        workflow = StateGraph(dict)  # Use dict as state type
        
        # Add ER extraction nodes
        workflow.add_node("extract_entities", self._extract_tpn_entities)
        workflow.add_node("identify_relationships", self._identify_tpn_relationships)  
        workflow.add_node("enhance_query", self._enhance_query_with_er)
        
        # Add edges
        workflow.add_edge("extract_entities", "identify_relationships")
        workflow.add_edge("identify_relationships", "enhance_query")
        workflow.add_edge("enhance_query", END)
        
        # Set entry point
        workflow.set_entry_point("extract_entities")
        
        return workflow.compile()
    
    async def extract_query_er(self, query: str) -> Dict[str, Any]:
        """Extract entities and relationships from TPN query."""
        
        initial_state = {
            "original_query": query,
            "entities": {},
            "relationships": [],
            "enhanced_query": "",
            "search_terms": []
        }
        
        try:
            # Run ER extraction workflow
            final_state = await self.er_graph.ainvoke(initial_state)
            return final_state
        except Exception as e:
            print(f"⚠️ ER extraction failed: {e}")
            # Fallback to basic processing
            return {
                "original_query": query,
                "entities": {},
                "relationships": [],
                "enhanced_query": query,
                "search_terms": [query]
            }
    
    async def _extract_tpn_entities(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract TPN-specific entities from query."""
        
        query = state["original_query"]
        
        # TPN entity extraction prompt
        er_prompt = f"""Extract TPN clinical entities from this query: "{query}"

Identify these TPN entity types:
- PATIENT: age group (preterm, term, pediatric, adult), weight, clinical condition
- TPN_COMPONENT: amino acids, dextrose, lipids, electrolytes, vitamins, trace elements
- DOSING: mg/kg/day, g/kg/day, ml/kg/day, kcal/kg/day, mg/kg/min
- LAB_VALUE: glucose, electrolytes, liver function, triglycerides
- CLINICAL_CONDITION: sepsis, IFALD, refeeding syndrome, malnutrition
- PROCEDURE: TPN initiation, weaning, monitoring, administration
- SAFETY: contraindications, complications, monitoring requirements

Extract key entities as comma-separated lists.

TPN Entity Analysis:"""
        
        try:
            response = await self.llm_provider.generate(
                prompt=er_prompt,
                temperature=0.1,
                max_tokens=200
            )
            
            # Extract entities (simplified processing)
            entities = {
                "patient": {"age_group": "", "weight": "", "condition": ""},
                "components": [],
                "dosing_units": [],
                "lab_values": [],
                "conditions": [],
                "procedures": [],
                "safety_aspects": []
            }
            
            # Extract key entities from response text
            response_lower = response.lower()
            
            # Patient entities
            if any(term in response_lower for term in ["preterm", "premature"]):
                entities["patient"]["age_group"] = "preterm"
            elif any(term in response_lower for term in ["term", "full-term"]):
                entities["patient"]["age_group"] = "term"
            elif any(term in response_lower for term in ["pediatric", "child"]):
                entities["patient"]["age_group"] = "pediatric"
            elif any(term in response_lower for term in ["adult"]):
                entities["patient"]["age_group"] = "adult"
            
            # TPN components
            tpn_components = ["amino acid", "protein", "dextrose", "glucose", "lipid", "fat", "sodium", "potassium", "phosphorus", "magnesium"]
            entities["components"] = [comp for comp in tpn_components if comp in response_lower]
            
            # Lab values
            lab_values = ["glucose", "triglyceride", "bilirubin", "alt", "ast", "bun", "creatinine", "albumin", "prealbumin"]
            entities["lab_values"] = [lab for lab in lab_values if lab in response_lower]
            
            # Clinical conditions
            conditions = ["sepsis", "ifald", "refeeding", "malnutrition", "cholestasis", "liver disease"]
            entities["conditions"] = [condition for condition in conditions if condition in response_lower]
            
            state["entities"] = entities
            
        except Exception as e:
            print(f"⚠️ Entity extraction failed: {e}")
            state["entities"] = {}
        
        return state
    
    async def _identify_tpn_relationships(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Identify relationships between TPN entities."""
        
        query = state["original_query"]
        entities = state["entities"]
        
        # Common TPN relationships
        relationships = []
        
        # Patient-Component relationships
        age_group = entities.get("patient", {}).get("age_group")
        if age_group and entities.get("components"):
            for component in entities["components"]:
                relationships.append({
                    "type": "REQUIRES",
                    "source": age_group,
                    "target": component,
                    "context": "age_specific_dosing"
                })
        
        # Component-Dosing relationships  
        if "calculate" in query.lower() or "dose" in query.lower():
            for component in entities.get("components", []):
                relationships.append({
                    "type": "CALCULATE_DOSE",
                    "source": component,
                    "target": "dosing_calculation",
                    "context": "tpn_prescription"
                })
        
        # Condition-Safety relationships
        for condition in entities.get("conditions", []):
            relationships.append({
                "type": "CONTRAINDICATED_IN",
                "source": "tpn_component",
                "target": condition,
                "context": "safety_consideration"
            })
        
        state["relationships"] = relationships
        return state
    
    async def _enhance_query_with_er(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance search query using extracted entities and relationships."""
        
        original_query = state["original_query"]
        entities = state["entities"]
        
        # Build enhanced search terms
        search_terms = [original_query]  # Start with original
        
        # Add entity-specific terms
        age_group = entities.get("patient", {}).get("age_group")
        if age_group:
            search_terms.extend([f"{age_group} TPN", f"{age_group} parenteral nutrition"])
        
        # Add component-specific terms
        for component in entities.get("components", []):
            search_terms.extend([f"TPN {component}", f"{component} dosing"])
        
        # Add condition-specific terms
        for condition in entities.get("conditions", []):
            search_terms.extend([f"TPN {condition}", f"{condition} contraindication"])
        
        # Add lab-specific terms
        for lab in entities.get("lab_values", []):
            search_terms.extend([f"TPN {lab} monitoring", f"{lab} normal range"])
        
        # Create enhanced query combining multiple search angles
        enhanced_query_parts = [
            original_query,
            " ".join(entities.get("components", [])),
            age_group or "",
            " ".join(entities.get("conditions", []))
        ]
        
        enhanced_query = " ".join(filter(None, enhanced_query_parts))
        
        state["enhanced_query"] = enhanced_query
        state["search_terms"] = list(set(search_terms))  # Remove duplicates
        
        return state
    
    async def _search_with_strategy(self, search_query: str, strategy: str, limit: int) -> List[SearchResult]:
        """Perform search with specific query and strategy."""
        
        try:
            # Generate embedding for search query
            query_embedding = await self.embedding_provider.embed_query(search_query)
            
            # Search vector store
            raw_results = await self.vector_store.search_similar(
                query_embedding,
                limit=limit,
                filters={}
            )
            
            # Convert to SearchResult objects and tag with strategy
            search_results = []
            for result in raw_results:
                # Include search strategy in metadata at creation time (since DocumentChunk is frozen)
                metadata = result.get("metadata", {}).copy()
                metadata["search_strategy"] = strategy
                
                chunk = DocumentChunk(
                    chunk_id=result["chunk_id"],
                    doc_id=result["doc_id"],
                    content=result["content"],
                    chunk_type=result.get("chunk_type", "text"),
                    page_num=result.get("page_num"),
                    section=result.get("section"),
                    metadata=metadata
                )
                
                search_result = SearchResult(
                    chunk=chunk,
                    score=result["score"],
                    document_name=result.get("document_name", "Unknown")
                )
                search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            print(f"⚠️ Search strategy '{strategy}' failed: {e}")
            return []
    
    async def _semantic_expansion_search(self, query: str, limit: int) -> List[SearchResult]:
        """Perform semantic expansion based on TPN domain knowledge."""
        
        # TPN-specific semantic expansions
        expansions = []
        
        # Age-based expansions
        if "infant" in query.lower() or "baby" in query.lower():
            expansions.extend(["neonate", "preterm", "term infant"])
        
        # Component expansions
        if "protein" in query.lower():
            expansions.extend(["amino acids", "amino acid solution"])
        
        if "sugar" in query.lower() or "carb" in query.lower():
            expansions.extend(["dextrose", "glucose"])
        
        if "fat" in query.lower():
            expansions.extend(["lipid", "fat emulsion", "lipid emulsion"])
        
        # Condition expansions
        if "liver" in query.lower():
            expansions.extend(["IFALD", "cholestasis", "hepatic"])
        
        # Perform searches with expansions
        expansion_results = []
        for expansion in expansions[:2]:  # Limit expansions
            expanded_query = f"{query} {expansion}"
            results = await self._search_with_strategy(expanded_query, "semantic_expansion", limit)
            expansion_results.extend(results)
        
        return expansion_results
    
    def _deduplicate_and_rank(self, all_results: List[SearchResult], limit: int) -> List[SearchResult]:
        """Deduplicate results and rank by relevance and strategy."""
        
        # Remove duplicates based on content hash
        seen_content = set()
        unique_results = []
        
        for result in all_results:
            content_hash = hash(result.content[:200])  # Hash first 200 chars
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
        
        # Enhanced scoring based on multiple factors - create new SearchResults with updated scores
        enhanced_results = []
        for result in unique_results:
            strategy = result.chunk.metadata.get("search_strategy", "unknown")
            base_score = result.score
            
            # Strategy bonuses
            strategy_bonus = {
                "original": 0.0,
                "enhanced": 0.1,
                "entity_focused": 0.05,
                "semantic_expansion": 0.02
            }.get(strategy, 0.0)
            
            # Content type bonuses
            content_type = result.chunk.metadata.get("content_type", "")
            content_bonus = {
                "dosage_recommendation": 0.15,
                "reference_values": 0.12,
                "safety_information": 0.10,
                "clinical_procedure": 0.08
            }.get(content_type, 0.0)
            
            # Document type bonuses
            doc_type = result.chunk.metadata.get("document_type", "")
            doc_bonus = {
                "clinical_guideline": 0.1,
                "nutrition_protocol": 0.08,
                "pediatric_protocol": 0.06
            }.get(doc_type, 0.0)
            
            # TPN relevance bonus
            tpn_keywords = ["tpn", "parenteral", "amino acid", "dextrose", "lipid", "aspen"]
            content_lower = result.content.lower()
            tpn_score = sum(1 for keyword in tpn_keywords if keyword in content_lower)
            tpn_bonus = min(0.1, tpn_score * 0.02)
            
            # Calculate final score and create new SearchResult (since SearchResult is frozen)
            final_score = min(1.0, base_score + strategy_bonus + content_bonus + doc_bonus + tpn_bonus)
            
            enhanced_result = SearchResult(
                chunk=result.chunk,
                score=final_score,
                document_name=result.document_name
            )
            enhanced_results.append(enhanced_result)
        
        # Sort by enhanced score and return top results
        sorted_results = sorted(enhanced_results, key=lambda x: x.score, reverse=True)
        return sorted_results[:limit]
