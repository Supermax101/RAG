"""
Medical-grade prompt templates for board-style questions and clinical reasoning.
"""
from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel

from ..models.documents import SearchResult


class QuestionType(str, Enum):
    """Types of medical questions."""
    BOARD_STYLE = "board_style"
    CLINICAL_REASONING = "clinical_reasoning"
    DOSAGE_CALCULATION = "dosage_calculation"
    DIFFERENTIAL_DIAGNOSIS = "differential_diagnosis"
    PROTOCOL_QUESTION = "protocol_question"
    REFERENCE_VALUES = "reference_values"
    CONTRAINDICATIONS = "contraindications"


class MedicalPromptTemplate(BaseModel):
    """Template for medical prompts."""
    name: str
    description: str
    template: str
    question_type: QuestionType
    required_sources: int = 3
    max_sources: int = 10


class MedicalPromptEngine:
    """Advanced prompt engineering for medical question answering."""
    
    def __init__(self):
        self.templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[QuestionType, MedicalPromptTemplate]:
        """Initialize medical prompt templates."""
        
        templates = {}
        
        # TPN Specialist Board-Style Question
        templates[QuestionType.BOARD_STYLE] = MedicalPromptTemplate(
            name="TPN Nutrition Specialist",
            description="Specialized template for TPN/parenteral nutrition clinical recommendations",
            question_type=QuestionType.BOARD_STYLE,
            template="""You are a TPN (Total Parenteral Nutrition) Clinical Specialist providing evidence-based recommendations using ONLY the provided ASPEN guidelines and institutional protocols.

TPN CLINICAL QUESTION: {question}

AVAILABLE TPN KNOWLEDGE BASE (USE ONLY THESE SOURCES):
{context}

CRITICAL CONSTRAINTS:
- Base ALL recommendations EXCLUSIVELY on the provided ASPEN documents and TPN protocols above
- DO NOT use external medical knowledge or general clinical experience
- If information is not available in the provided sources, explicitly state "Information not available in current TPN knowledge base"
- Only cite recommendations, dosages, and guidelines that appear in the provided sources
- When calculating TPN components, use ONLY the ranges and formulas from these specific documents

CLINICAL DECISION FRAMEWORK:
- Apply ASPEN evidence-based TPN guidelines from PROVIDED SOURCES as the sole authority
- Consider patient-specific factors ONLY as described in these documents
- Calculate TPN components using ONLY the dosing ranges from these sources
- Reference TPN indications/contraindications ONLY from the provided protocols

TPN RECOMMENDATION FORMAT:

1) **TPN Clinical Recommendation:** (Specific recommendation with rationale)

2) **TPN Component Analysis:** (Evidence-based component breakdown)
   • **Indication Assessment:** [TPN vs alternatives based on ASPEN criteria with source]
   • **Age/Weight Considerations:** [Preterm: <1500g, Term: >2500g, Pediatric: 1-18yr, Adult specifics]
   • **Component Dosing:** [Amino acids: g/kg/day, Dextrose: mg/kg/min, Lipids: g/kg/day, Fluids: mL/kg/day]
   • **Monitoring Protocol:** [Lab parameters, frequencies, clinical markers per ASPEN]
   • **Safety Considerations:** [TPN-related complications, contraindications, precautions]

3) **ASPEN Evidence Base:** [Source, Module/Chapter, Recommendation Level, Year]
   Format: [ASPEN Neonatal PN Guidelines, Energy Requirements, Grade A, 2023]
   Format: [ASPEN Safety Recommendations, Central Line Management, Evidence-based, 2022]

If the provided TPN knowledge base lacks sufficient information: "INSUFFICIENT DATA: The provided TPN documents do not contain adequate information on [specific parameter]. Additional ASPEN guidelines needed."

REMEMBER: Use ONLY the information from the provided TPN knowledge base above. Do not supplement with external medical knowledge.

TPN CLINICAL RECOMMENDATION:""",
            required_sources=3,
            max_sources=6
        )
        
        # Clinical reasoning prompt
        templates[QuestionType.CLINICAL_REASONING] = MedicalPromptTemplate(
            name="Clinical Reasoning",
            description="Template for complex clinical reasoning scenarios",
            question_type=QuestionType.CLINICAL_REASONING,
            template="""You are providing clinical reasoning for a complex medical scenario using evidence-based sources.

CLINICAL SCENARIO: {question}

EVIDENCE BASE:
{context}

CLINICAL REASONING APPROACH:
1. Analyze the clinical scenario systematically
2. Apply evidence-based guidelines from the sources
3. Consider differential diagnoses where applicable
4. Prioritize patient safety and clinical outcomes

RESPONSE FORMAT:

**Clinical Assessment:**
[Primary clinical considerations based on evidence]

**Evidence-Based Recommendations:**
• [Recommendation 1 with source citation]
• [Recommendation 2 with source citation] 
• [Recommendation 3 with source citation]

**Clinical Monitoring:**
[What parameters to monitor and why]

**Contraindications/Precautions:**
[Important safety considerations from sources]

**Source References:**
[List all sources cited in format: Author/Guideline, Section, Year]

CLINICAL ANALYSIS:""",
            required_sources=2,
            max_sources=6
        )
        
        # TPN Dosage Calculation Template
        templates[QuestionType.DOSAGE_CALCULATION] = MedicalPromptTemplate(
            name="TPN Component Calculation",
            description="Template for TPN component dosing and nutritional calculations",
            question_type=QuestionType.DOSAGE_CALCULATION,
            template="""You are a TPN Clinical Pharmacist calculating parenteral nutrition components using ONLY the provided ASPEN guidelines and protocols.

TPN CALCULATION REQUEST: {question}

AVAILABLE TPN DOSING GUIDELINES (USE ONLY THESE):
{context}

STRICT CALCULATION CONSTRAINTS:
- Use EXCLUSIVELY the dosing ranges and formulas from the provided TPN documents above
- DO NOT apply general pharmaceutical calculations or external dosing guidelines
- If specific dosing information is missing from these sources, state "Dosing data not available in provided TPN guidelines"
- All calculations must reference specific values from the provided documents
- Only use age-specific ranges that appear in these exact sources

TPN CALCULATION METHODOLOGY:
- Apply ONLY the weight-based dosing guidelines from the PROVIDED sources by age group
- Use ONLY the units and ranges specified in these documents (g/kg/day, mg/kg/min, mL/kg/day, kcal/kg/day)
- Reference ONLY the safety limits mentioned in these specific protocols
- Account for clinical conditions ONLY as described in the provided sources

TPN CALCULATION FORMAT:

**Patient Parameters:**
[Age group, weight, clinical condition, TPN indication]

**TPN Component Calculations:**

**1. Energy Requirements:**
• Target calories: [kcal/kg/day per age group with ASPEN source]
• Calculation: [Weight × kcal/kg/day = total kcal/day]

**2. Protein (Amino Acids):**
• Target protein: [g/kg/day per ASPEN age recommendations]
• Calculation: [Weight × g/kg/day = total grams/day]
• Verification: [Within ASPEN safe limits with source]

**3. Carbohydrates (Dextrose):**
• Target GIR: [mg/kg/min per age group]
• Calculation: [Weight × GIR × 1440 ÷ 1000 = grams dextrose/day]
• Concentration: [Final dextrose % in TPN]

**4. Lipids (Fat Emulsion):**
• Target lipids: [g/kg/day per ASPEN guidelines]
• Calculation: [Weight × g/kg/day = total lipid grams/day]
• EFAD prevention: [Minimum lipid dose with source]

**5. Fluid Requirements:**
• Age-appropriate fluids: [mL/kg/day with clinical adjustments]
• Total TPN volume calculation
• Concentration adjustments for fluid restrictions

**Clinical Verification:**
• ASPEN compliance: [All components within guideline ranges]
• Safety limits: [Maximum concentrations and infusion rates]
• Monitoring plan: [Required lab parameters and frequencies]

**ASPEN References:**
[Specific ASPEN guidelines with dosing tables/ranges used from the provided sources]

If dosing information is incomplete in provided sources: "CALCULATION INCOMPLETE: Required dosing parameters not available in current TPN knowledge base for [specific component]."

CONSTRAINT REMINDER: All calculations based EXCLUSIVELY on the provided TPN documents above.

TPN PRESCRIPTION CALCULATION:""",
            required_sources=2,
            max_sources=4
        )
        
        # Protocol question prompt
        templates[QuestionType.PROTOCOL_QUESTION] = MedicalPromptTemplate(
            name="Protocol Question", 
            description="Template for clinical protocol and procedure questions",
            question_type=QuestionType.PROTOCOL_QUESTION,
            template="""You are providing clinical protocol guidance based on authoritative medical guidelines.

PROTOCOL QUESTION: {question}

CLINICAL GUIDELINES:
{context}

PROTOCOL RESPONSE REQUIREMENTS:
- Follow evidence-based guidelines strictly
- Include step-by-step procedures where applicable
- Highlight safety considerations
- Note any contraindications
- Reference specific protocol versions/dates

RESPONSE FORMAT:

**Clinical Protocol:**
[Main protocol steps from guidelines]

**Key Steps:**
1. [Step 1 with clinical rationale]
2. [Step 2 with safety considerations]
3. [Step 3 with monitoring requirements]
[Continue as needed]

**Safety Considerations:**
• [Safety point 1 with source]
• [Safety point 2 with source]
• [Contraindications from guidelines]

**Clinical Monitoring:**
[What to monitor and when, per protocols]

**Guideline References:**
[Specific protocol sources with versions/dates]

PROTOCOL GUIDANCE:""",
            required_sources=1,
            max_sources=4
        )
        
        # TPN Reference Values Template
        templates[QuestionType.REFERENCE_VALUES] = MedicalPromptTemplate(
            name="TPN Reference Values & Lab Monitoring",
            description="Template for TPN-related lab values, normal ranges, and monitoring parameters",
            question_type=QuestionType.REFERENCE_VALUES,
            template="""You are a TPN Clinical Specialist providing reference values and monitoring parameters using ONLY the provided ASPEN monitoring guidelines and lab references.

TPN MONITORING QUESTION: {question}

AVAILABLE TPN MONITORING KNOWLEDGE BASE (USE ONLY THESE):
{context}

MONITORING CONSTRAINTS:
- Provide ONLY the reference ranges that appear in the provided TPN documents above
- DO NOT use standard laboratory reference ranges from external sources
- If specific ranges are not mentioned in these documents, state "Reference range not specified in available TPN guidelines"
- Use ONLY the monitoring frequencies specified in these exact sources
- Reference ONLY the patient populations described in the provided documents

TPN MONITORING FRAMEWORK (USING PROVIDED SOURCES ONLY):
- Provide age-specific reference ranges ONLY as stated in these TPN documents
- Include monitoring frequencies ONLY as specified in these sources
- Address parameters affected by TPN ONLY as described in the provided protocols
- Consider patient populations ONLY as categorized in these specific guidelines

TPN MONITORING RESPONSE FORMAT:

**TPN-Related Reference Ranges:**

**Metabolic Parameters:**
• Glucose: [Age-specific ranges in mg/dL with TPN targets]
• BUN/Creatinine: [Normal ranges with TPN considerations]
• Liver function: [ALT, AST, bilirubin ranges affected by TPN]
• Triglycerides: [Lipid tolerance ranges during TPN with ASPEN limits]

**Electrolyte Balance:**
• Sodium: [mEq/L ranges by age with TPN-specific adjustments]
• Potassium: [mEq/L ranges with TPN dosing considerations]
• Phosphorus: [mg/dL ranges affected by TPN dextrose/amino acids]
• Magnesium: [mg/dL ranges for TPN patients]

**Nutritional Status:**
• Albumin/Prealbumin: [Protein status markers during TPN]
• Essential fatty acids: [Triene:tetraene ratios for EFAD monitoring]
• Trace elements: [Zinc, selenium, copper ranges on TPN]

**Age-Specific TPN Ranges:**
• **Preterm (<1500g):** [Specific parameters and frequencies]
• **Term Neonates:** [Age-appropriate ranges and monitoring]
• **Pediatric (1-18 years):** [Growth-adjusted parameters]
• **Adult:** [Standard ranges with TPN modifications]

**TPN Monitoring Schedule:**
• **Daily:** [Parameters requiring daily monitoring on TPN]
• **Weekly:** [Routine TPN monitoring parameters]
• **Monthly:** [Long-term TPN monitoring requirements]

**Critical Values Requiring Intervention:**
[TPN-specific critical values with immediate actions needed]

**ASPEN Monitoring References:**
[ASPEN guidelines with specific monitoring tables/protocols from the provided sources]

If monitoring parameters are not specified in provided sources: "MONITORING DATA INCOMPLETE: Required monitoring parameters not available in current TPN knowledge base."

SOURCE CONSTRAINT: All monitoring recommendations based EXCLUSIVELY on the provided TPN monitoring guidelines above.

TPN MONITORING GUIDANCE:""",
            required_sources=2,
            max_sources=4
        )
        
        return templates
    
    def generate_medical_prompt(
        self,
        question: str,
        sources: List[SearchResult],
        question_type: Optional[QuestionType] = None,
        custom_instructions: Optional[str] = None
    ) -> str:
        """Generate a medical prompt based on question type and sources."""
        
        # Auto-detect question type if not provided
        if question_type is None:
            question_type = self._detect_question_type(question)
        
        # Get appropriate template
        template = self.templates.get(question_type, self.templates[QuestionType.BOARD_STYLE])
        
        # Build context from sources
        context = self._build_medical_context(sources, template.max_sources)
        
        # Generate the prompt
        prompt = template.template.format(
            question=question,
            context=context
        )
        
        # Add custom instructions if provided
        if custom_instructions:
            prompt += f"\n\nADDITIONAL INSTRUCTIONS:\n{custom_instructions}\n"
        
        return prompt
    
    def _detect_question_type(self, question: str) -> QuestionType:
        """Auto-detect the type of TPN/nutrition question."""
        question_lower = question.lower()
        
        # TPN dosage/calculation indicators
        if any(term in question_lower for term in [
            'calculate', 'dose', 'dosage', 'mg/kg', 'ml/kg', 'g/kg', 'kcal/kg',
            'mg/kg/min', 'mg/kg/day', 'tpn calculation', 'amino acid dose',
            'dextrose concentration', 'lipid dose', 'gir', 'glucose infusion rate',
            'how much', 'total calories', 'protein requirement'
        ]):
            return QuestionType.DOSAGE_CALCULATION
        
        # TPN protocol/administration indicators
        elif any(term in question_lower for term in [
            'protocol', 'procedure', 'how to start', 'tpn administration',
            'central line', 'peripheral pn', 'tpn initiation', 'tpn weaning',
            'tpn cycling', 'home tpn', 'tpn transition', 'enteral transition'
        ]):
            return QuestionType.PROTOCOL_QUESTION
        
        # TPN monitoring/lab values indicators
        elif any(term in question_lower for term in [
            'normal range', 'reference range', 'lab values', 'monitor',
            'triglycerides', 'liver function', 'glucose target', 'electrolyte range',
            'bun', 'creatinine', 'albumin', 'prealbumin', 'trace elements',
            'monitoring frequency', 'lab schedule'
        ]):
            return QuestionType.REFERENCE_VALUES
        
        # TPN contraindication/safety indicators
        elif any(term in question_lower for term in [
            'contraindication', 'contraindicated', 'avoid', 'not recommended',
            'tpn complications', 'ifald', 'refeeding syndrome', 'tpn cholestasis',
            'central line infection', 'metabolic complications', 'tpn safety'
        ]):
            return QuestionType.CONTRAINDICATIONS
        
        # TPN clinical reasoning indicators
        elif any(term in question_lower for term in [
            'why', 'mechanism', 'rationale', 'reasoning', 'explain',
            'physiology', 'clinical significance', 'indication', 'when to use tpn'
        ]):
            return QuestionType.CLINICAL_REASONING
        
        # Default to TPN board-style for comprehensive TPN questions
        else:
            return QuestionType.BOARD_STYLE
    
    def _build_medical_context(self, sources: List[SearchResult], max_sources: int) -> str:
        """Build structured TPN context from search results - DOCUMENT-CONSTRAINED."""
        
        if not sources:
            return "[No relevant TPN information found in the available 52 ASPEN/TPN knowledge base documents]"
        
        context_parts = []
        context_parts.append("=== TPN KNOWLEDGE BASE SOURCES (USE ONLY THESE) ===")
        
        # Limit to max sources and prioritize by score
        sorted_sources = sorted(sources, key=lambda x: x.score, reverse=True)
        top_sources = sorted_sources[:max_sources]
        
        for i, result in enumerate(top_sources, 1):
            # Extract metadata for better citation
            doc_type = result.chunk.metadata.get('document_type', 'clinical_text')
            year = result.chunk.metadata.get('year', 'recent')
            section = result.chunk.section or 'General'
            
            # Format source with clinical context
            source_header = f"[SOURCE {i}] {result.document_name}"
            if year != 'recent':
                source_header += f" ({year})"
            source_header += f" - {section}"
            if doc_type in ['clinical_guideline', 'nutrition_protocol']:
                source_header += f" [GUIDELINE]"
            
            source_content = f"{source_header}\n{result.content}\n"
            
            # Add content type context
            content_type = result.chunk.metadata.get('content_type', 'general_clinical')
            if content_type in ['dosage_recommendation', 'reference_values', 'safety_information']:
                source_content += f"[CONTENT TYPE: {content_type.replace('_', ' ').title()}]\n"
            
            context_parts.append(source_content)
        
        # Add constraint footer
        context_parts.append("\n=== END OF AVAILABLE TPN KNOWLEDGE BASE ===")
        context_parts.append("CRITICAL: Base ALL recommendations EXCLUSIVELY on the sources above from your 52 ASPEN/TPN document knowledge base.")
        context_parts.append("DO NOT use external medical knowledge beyond what is provided above.")
        
        return "\n".join(context_parts)
    
    def validate_medical_response(self, response: str, question_type: QuestionType) -> Dict[str, Any]:
        """Validate that medical response follows required format."""
        
        validation_result = {
            "is_valid": True,
            "missing_elements": [],
            "warnings": []
        }
        
        response_lower = response.lower()
        
        # Check for required elements based on question type
        if question_type == QuestionType.BOARD_STYLE:
            required_elements = {
                "final answer": ["final answer", "answer:"],
                "clinical reasoning": ["clinical reasoning", "reasoning:", "bullet"],
                "sources": ["sources:", "references:", "citation"]
            }
        elif question_type == QuestionType.DOSAGE_CALCULATION:
            required_elements = {
                "calculation": ["calculation", "step", "formula"],
                "units": ["mg", "kg", "ml", "hr", "units"],
                "verification": ["range", "normal", "safe"]
            }
        else:
            required_elements = {
                "evidence": ["evidence", "source", "guideline"],
                "clinical": ["clinical", "patient", "safety"]
            }
        
        # Check for missing elements
        for element, indicators in required_elements.items():
            if not any(indicator in response_lower for indicator in indicators):
                validation_result["missing_elements"].append(element)
                validation_result["is_valid"] = False
        
        # Check for quality indicators
        if len(response) < 200:
            validation_result["warnings"].append("Response may be too brief for medical question")
        
        if "insufficient evidence" in response_lower and len(response) < 100:
            validation_result["warnings"].append("Should specify what evidence is missing")
        
        return validation_result
