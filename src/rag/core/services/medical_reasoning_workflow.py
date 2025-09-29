"""
LangGraph workflow for TPN clinical decision-making and multi-step parenteral nutrition analysis.
"""
import json
from typing import List, Dict, Any, TypedDict, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from ..models.documents import RAGQuery, RAGResponse, SearchQuery, SearchResult
from .medical_prompt_templates import MedicalPromptEngine, QuestionType
from .rag_service import RAGService


class TPNReasoningState(TypedDict):
    """State for TPN clinical reasoning workflow."""
    original_question: str
    question_type: QuestionType
    patient_factors: Dict[str, Any]  # Age, weight, clinical condition
    tpn_indication: str
    search_results: List[SearchResult]
    tpn_components: Dict[str, Any]  # Calculated TPN components
    reasoning_steps: List[str]
    conflicts_detected: List[str]
    safety_considerations: List[str]
    monitoring_plan: Dict[str, Any]
    final_recommendation: str
    confidence_score: float
    aspen_compliance: bool
    validation_results: Dict[str, Any]
    response_time_ms: float


class TPNReasoningWorkflow:
    """LangGraph-based workflow for TPN clinical decision-making and recommendations."""
    
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
        self.prompt_engine = MedicalPromptEngine()
        self.graph = self._build_workflow_graph()
    
    def _build_workflow_graph(self) -> StateGraph:
        """Build the LangGraph workflow for medical reasoning."""
        
        workflow = StateGraph(dict)  # Use dict as state type
        
        # Add TPN-specific nodes
        workflow.add_node("analyze_tpn_question", self._analyze_question)
        workflow.add_node("extract_patient_factors", self._extract_patient_factors)
        workflow.add_node("search_tpn_sources", self._search_tpn_sources)
        workflow.add_node("evaluate_tpn_indication", self._evaluate_tpn_indication)
        workflow.add_node("calculate_tpn_components", self._calculate_tpn_components)
        workflow.add_node("assess_safety_considerations", self._assess_safety_considerations)
        workflow.add_node("generate_tpn_recommendation", self._generate_tpn_recommendation)
        workflow.add_node("validate_aspen_compliance", self._validate_aspen_compliance)
        
        # Add edges (workflow flow)
        workflow.add_edge("analyze_tpn_question", "extract_patient_factors")
        workflow.add_edge("extract_patient_factors", "search_tpn_sources")
        workflow.add_edge("search_tpn_sources", "evaluate_tpn_indication")
        workflow.add_edge("evaluate_tpn_indication", "calculate_tpn_components")
        workflow.add_edge("calculate_tpn_components", "assess_safety_considerations")
        workflow.add_edge("assess_safety_considerations", "generate_tpn_recommendation")
        workflow.add_edge("generate_tpn_recommendation", "validate_aspen_compliance")
        workflow.add_edge("validate_aspen_compliance", END)
        
        # Set entry point
        workflow.set_entry_point("analyze_tpn_question")
        
        return workflow.compile()
    
    async def process_tpn_question(self, question: str, **kwargs) -> RAGResponse:
        """Process a TPN clinical question through the specialized workflow."""
        
        # Initialize TPN-specific state
        initial_state = TPNReasoningState(
            original_question=question,
            question_type=QuestionType.BOARD_STYLE,
            patient_factors={},
            tpn_indication="",
            search_results=[],
            tpn_components={},
            reasoning_steps=[],
            conflicts_detected=[],
            safety_considerations=[],
            monitoring_plan={},
            final_recommendation="",
            confidence_score=0.0,
            aspen_compliance=False,
            validation_results={},
            response_time_ms=0.0
        )
        
        # Run TPN-focused workflow
        import time
        start_time = time.time()
        
        final_state = await self.graph.ainvoke(initial_state)
        
        total_time = (time.time() - start_time) * 1000
        
        # Convert to RAGResponse with TPN-specific formatting
        response = RAGResponse(
            question=question,
            answer=final_state["final_recommendation"],
            sources=final_state["search_results"],
            search_time_ms=final_state.get("search_time_ms", 0),
            generation_time_ms=total_time - final_state.get("search_time_ms", 0),
            total_time_ms=total_time,
            model_used=f"tpn_specialist_workflow_{final_state['question_type']}"
        )
        
        return response
    
    async def _analyze_question(self, state: TPNReasoningState) -> TPNReasoningState:
        """Analyze the medical question to determine type and approach."""
        
        question = state["original_question"]
        
        # Detect question type
        question_type = self.prompt_engine._detect_question_type(question)
        state["question_type"] = question_type
        
        # Add reasoning step
        state["reasoning_steps"].append(f"Question classified as: {question_type.value}")
        
        # Determine search strategy based on question type
        if question_type == QuestionType.DOSAGE_CALCULATION:
            state["reasoning_steps"].append("Search strategy: Focus on dosage guidelines and calculations")
        elif question_type == QuestionType.REFERENCE_VALUES:
            state["reasoning_steps"].append("Search strategy: Focus on reference ranges and normal values")
        elif question_type == QuestionType.PROTOCOL_QUESTION:
            state["reasoning_steps"].append("Search strategy: Focus on clinical protocols and procedures")
        else:
            state["reasoning_steps"].append("Search strategy: Comprehensive clinical evidence search")
        
        return state
    
    async def _search_medical_sources(self, state: TPNReasoningState) -> TPNReasoningState:
        """Search for relevant medical sources."""
        
        import time
        search_start = time.time()
        
        question = state["original_question"]
        question_type = state["question_type"]
        
        # Determine search parameters based on question type
        if question_type == QuestionType.DOSAGE_CALCULATION:
            search_limit = 5
            filters = {"content_type": "dosage_recommendation"}
        elif question_type == QuestionType.REFERENCE_VALUES:
            search_limit = 3
            filters = {"content_type": "reference_values"}
        elif question_type == QuestionType.PROTOCOL_QUESTION:
            search_limit = 4
            filters = {"document_type": "clinical_guideline"}
        else:
            search_limit = 6
            filters = {}
        
        # Perform search
        search_query = SearchQuery(
            query=question,
            limit=search_limit,
            filters=filters
        )
        
        search_response = await self.rag_service.search(search_query)
        state["search_results"] = search_response.results
        state["search_time_ms"] = search_response.search_time_ms
        
        # Add reasoning step
        state["reasoning_steps"].append(
            f"Found {len(search_response.results)} relevant sources in {search_response.search_time_ms:.1f}ms"
        )
        
        return state
    
    async def _evaluate_source_quality(self, state: TPNReasoningState) -> TPNReasoningState:
        """Evaluate the quality and relevance of sources."""
        
        sources = state["search_results"]
        
        if not sources:
            state["reasoning_steps"].append("⚠️ No sources found - will indicate insufficient evidence")
            state["confidence_score"] = 0.0
            return state
        
        # Quality scoring based on metadata
        scored_sources = []
        for source in sources:
            quality_score = self._calculate_source_quality(source)
            source.chunk.metadata["quality_score"] = quality_score
            scored_sources.append(source)
        
        # Sort by quality and relevance
        scored_sources.sort(key=lambda x: (x.score * x.chunk.metadata["quality_score"]), reverse=True)
        
        state["search_results"] = scored_sources
        
        # Calculate confidence based on source quality
        avg_quality = sum(s.chunk.metadata["quality_score"] for s in scored_sources) / len(scored_sources)
        avg_relevance = sum(s.score for s in scored_sources) / len(scored_sources)
        state["confidence_score"] = (avg_quality * avg_relevance) * 0.9  # Conservative estimate
        
        state["reasoning_steps"].append(
            f"Source quality evaluation: {len([s for s in scored_sources if s.chunk.metadata['quality_score'] > 0.7])} high-quality sources"
        )
        
        return state
    
    async def _detect_source_conflicts(self, state: TPNReasoningState) -> TPNReasoningState:
        """Detect conflicts between sources."""
        
        sources = state["search_results"]
        conflicts = []
        
        # Simple conflict detection based on content analysis
        for i, source1 in enumerate(sources):
            for j, source2 in enumerate(sources[i+1:], i+1):
                conflict = self._detect_content_conflict(source1, source2)
                if conflict:
                    conflicts.append(conflict)
        
        state["conflicts_detected"] = conflicts
        
        if conflicts:
            state["reasoning_steps"].append(f"⚠️ Detected {len(conflicts)} potential source conflicts")
            state["confidence_score"] *= 0.8  # Reduce confidence due to conflicts
        else:
            state["reasoning_steps"].append("✅ No major conflicts detected between sources")
        
        return state
    
    async def _generate_medical_response(self, state: TPNReasoningState) -> TPNReasoningState:
        """Generate medical response using appropriate prompt template."""
        
        question = state["original_question"]
        sources = state["search_results"]
        question_type = state["question_type"]
        conflicts = state["conflicts_detected"]
        
        # Generate prompt with conflict handling
        custom_instructions = ""
        if conflicts:
            custom_instructions = f"""
IMPORTANT - SOURCE CONFLICTS DETECTED:
{chr(10).join(conflicts)}

When conflicts exist, clearly state the conflict and explain which source to prefer based on:
- Publication date (prefer newer)
- Source authority (prefer official guidelines)
- Specificity to the clinical scenario
"""
        
        prompt = self.prompt_engine.generate_medical_prompt(
            question=question,
            sources=sources,
            question_type=question_type,
            custom_instructions=custom_instructions
        )
        
        # Generate response using LLM
        try:
            response = await self.rag_service.llm_provider.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=800
            )
            state["final_answer"] = response
            
            state["reasoning_steps"].append("✅ Generated structured medical response")
            
        except Exception as e:
            state["final_answer"] = f"Error generating medical response: {str(e)}"
            state["confidence_score"] = 0.0
            state["reasoning_steps"].append(f"❌ Response generation failed: {str(e)}")
        
        return state
    
    async def _validate_medical_response(self, state: TPNReasoningState) -> TPNReasoningState:
        """Validate the medical response format and content."""
        
        response = state["final_answer"]
        question_type = state["question_type"]
        
        # Validate using prompt engine
        validation_results = self.prompt_engine.validate_medical_response(response, question_type)
        state["validation_results"] = validation_results
        
        if not validation_results["is_valid"]:
            state["confidence_score"] *= 0.6  # Reduce confidence for invalid format
            state["reasoning_steps"].append(
                f"⚠️ Response format issues: {', '.join(validation_results['missing_elements'])}"
            )
        else:
            state["reasoning_steps"].append("✅ Response format validated")
        
        return state
    
    async def _refine_response_if_needed(self, state: TPNReasoningState) -> TPNReasoningState:
        """Refine response if validation failed."""
        
        validation_results = state["validation_results"]
        
        if not validation_results["is_valid"] and state["confidence_score"] > 0.3:
            # Attempt to refine the response
            missing_elements = validation_results["missing_elements"]
            current_response = state["final_answer"]
            
            refinement_prompt = f"""
The following medical response is missing required elements: {', '.join(missing_elements)}

Original Response:
{current_response}

Please add the missing elements while maintaining the existing content. Ensure the response follows proper medical format.

Refined Response:"""
            
            try:
                refined_response = await self.rag_service.llm_provider.generate(
                    prompt=refinement_prompt,
                    temperature=0.05,
                    max_tokens=600
                )
                
                state["final_answer"] = refined_response
                state["reasoning_steps"].append("✅ Response refined to address format issues")
                
            except Exception as e:
                state["reasoning_steps"].append(f"⚠️ Refinement failed: {str(e)}")
        
        # Add metadata to response
        metadata_footer = f"""

---
**Reasoning Workflow Metadata:**
- Question Type: {state['question_type'].value}
- Sources Analyzed: {len(state['search_results'])}
- Conflicts Detected: {len(state['conflicts_detected'])}
- Confidence Score: {state['confidence_score']:.2f}
- Processing Steps: {len(state['reasoning_steps'])}
"""
        
        if state["final_answer"] and not state["final_answer"].endswith("Error"):
            state["final_answer"] += metadata_footer
        
        return state
    
    def _calculate_source_quality(self, source: SearchResult) -> float:
        """Calculate quality score for a medical source."""
        
        quality_score = 0.5  # Base score
        
        # Document type scoring
        doc_type = source.chunk.metadata.get('document_type', '')
        if doc_type == 'clinical_guideline':
            quality_score += 0.3
        elif doc_type in ['nutrition_protocol', 'pediatric_protocol']:
            quality_score += 0.2
        
        # Authority scoring
        if 'aspen' in source.document_name.lower():
            quality_score += 0.2
        
        # Content type scoring
        content_type = source.chunk.metadata.get('content_type', '')
        if content_type in ['dosage_recommendation', 'reference_values', 'safety_information']:
            quality_score += 0.2
        
        # Recency scoring
        year = source.chunk.metadata.get('year')
        if year and year >= 2020:
            quality_score += 0.1
        elif year and year >= 2015:
            quality_score += 0.05
        
        return min(quality_score, 1.0)
    
    def _detect_content_conflict(self, source1: SearchResult, source2: SearchResult) -> Optional[str]:
        """Detect potential conflicts between two sources."""
        
        # Simple conflict detection based on contradictory keywords
        content1_lower = source1.content.lower()
        content2_lower = source2.content.lower()
        
        # Look for contradictory statements
        contradictions = [
            ("contraindicated", "recommended"),
            ("avoid", "use"),
            ("not recommended", "recommended"),
            ("increase", "decrease"),
            ("high dose", "low dose")
        ]
        
        for term1, term2 in contradictions:
            if term1 in content1_lower and term2 in content2_lower:
                return f"Potential conflict between '{source1.document_name}' and '{source2.document_name}': {term1} vs {term2}"
            elif term2 in content1_lower and term1 in content2_lower:
                return f"Potential conflict between '{source1.document_name}' and '{source2.document_name}': {term2} vs {term1}"
        
        return None
