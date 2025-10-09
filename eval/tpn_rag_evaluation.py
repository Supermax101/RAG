import asyncio
import pandas as pd
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import re
from pydantic import BaseModel, Field

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import RAG system components
from src.rag.infrastructure.embeddings.ollama_embeddings import OllamaEmbeddingProvider
from src.rag.infrastructure.vector_stores.chroma_store import ChromaVectorStore
from src.rag.infrastructure.llm_providers.ollama_provider import OllamaLLMProvider
from src.rag.infrastructure.llm_providers.openai_provider import OpenAILLMProvider
from src.rag.infrastructure.llm_providers.xai_provider import XAILLMProvider
from src.rag.infrastructure.llm_providers.gemini_provider import GeminiLLMProvider
from src.rag.infrastructure.llm_providers.kimi_provider import KimiLLMProvider
from src.rag.core.services.hybrid_rag_service import HybridRAGService
from src.rag.core.models.documents import SearchQuery

# LangChain imports for structured output (Pydantic v2)
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# RAGAS imports (optional for advanced metrics)
# TEMPORARILY DISABLED: RAGAS requires instruction-following models (OpenAI, Claude, Gemini)
# Local LLMs (Ollama) produce unreliable metrics due to poor JSON output formatting
# To re-enable: Change RAGAS_AVAILABLE = True and ensure OpenAI API key is set
RAGAS_AVAILABLE = False  # Explicitly disabled for Ollama-based evaluation

try:
    if RAGAS_AVAILABLE:  # Only import if enabled
        from ragas import evaluate
        from ragas.metrics import (
            answer_correctness,
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from ragas.llms import LangchainLLMWrapper  # RAGAS 0.3.x wrapper for LangChain LLMs
        from ragas.embeddings import LangchainEmbeddingsWrapper  # RAGAS 0.3.x wrapper for embeddings
        from datasets import Dataset
        # LangChain LLM and Embeddings bases
        from langchain_core.language_models.llms import LLM
        from langchain_core.embeddings import Embeddings
    else:
        LLM = None
        Embeddings = None
except ImportError as e:
    print(f"WARNING: RAGAS not fully available ({e}), continuing with basic metrics only")
    RAGAS_AVAILABLE = False
    LLM = None  # Define as None so class definition doesn't fail
    Embeddings = None


class MCQAnswer(BaseModel):
    """Structured output for MCQ answers using Pydantic v2."""
    answer: str = Field(description="Single letter answer (A, B, C, D, E, or F)")
    confidence: Optional[str] = Field(default="medium", description="Confidence level: low, medium, high")


# Only define custom wrappers if RAGAS is available
if RAGAS_AVAILABLE:
    class CustomOllamaRagasLLM(LLM):
        """A custom LLM wrapper for Ragas to use Ollama."""
        model: str
        ollama_provider: Any  # Use Any to avoid Pydantic validation issues
        
        def __init__(self, model: str, ollama_provider: OllamaLLMProvider):
            """Initialize with model and provider."""
            super().__init__(model=model, ollama_provider=ollama_provider)
        
        @property
        def _llm_type(self) -> str:
            return "ollama-ragas-custom"
        
        def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[Any] = None,
            **kwargs: Any,
        ) -> str:
            return asyncio.run(self.ollama_provider.generate(prompt=prompt, model=self.model, **kwargs))
        
        @property
        def _identifying_params(self) -> Dict[str, Any]:
            return {"model": self.model}
        
        class Config:
            arbitrary_types_allowed = True
    
    class CustomOllamaRagasEmbeddings(Embeddings):
        """A custom Embeddings wrapper for Ragas to use Ollama."""
        
        def __init__(self, embedding_provider: OllamaEmbeddingProvider):
            """Initialize with embedding provider."""
            # Embeddings base class doesn't accept kwargs, so call super without args
            super().__init__()
            # Set our custom field as an instance variable
            self.embedding_provider = embedding_provider
        
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            """Embed a list of documents."""
            return asyncio.run(self.embedding_provider.embed_texts(texts))
        
        def embed_query(self, text: str) -> List[float]:
            """Embed a single query."""
            return asyncio.run(self.embedding_provider.embed_query(text))


class TPNRAGEvaluator:
    """Evaluates TPN RAG system using modern LangChain approach with structured output."""
    
    def __init__(self, csv_path: str, selected_model: str = "mistral:7b", provider: str = "ollama"):
        self.csv_path = csv_path
        self.selected_model = selected_model
        self.provider = provider
        self.rag_service = None
        self.evaluation_results = []
        
        # Setup output parser for structured MCQ answers
        self.parser = JsonOutputParser(pydantic_object=MCQAnswer)
        
        # Create few-shot examples for better model guidance
        self.few_shot_examples = [
            {
                "question": "What is the recommended starting dextrose concentration for neonatal TPN?",
                "options": "A) 5%\nB) 10%\nC) 15%\nD) 20%",
                "answer": '{"answer": "B", "confidence": "high"}'
            },
            {
                "question": "Which electrolyte should be monitored most frequently in TPN patients?",
                "options": "A) Sodium\nB) Potassium\nC) Calcium\nD) Magnesium",
                "answer": '{"answer": "B", "confidence": "high"}'
            }
        ]
        
        if RAGAS_AVAILABLE:
            # Create custom Ollama LLM (LangChain-compatible)
            ollama_llm = CustomOllamaRagasLLM(
                model=self.selected_model,
                ollama_provider=OllamaLLMProvider(default_model=self.selected_model)
            )
            
            # Create custom Ollama Embeddings (LangChain-compatible)
            ollama_embeddings = CustomOllamaRagasEmbeddings(
                embedding_provider=OllamaEmbeddingProvider()
            )
            
            # Wrap for RAGAS 0.3.x using official wrappers
            self.ragas_llm = LangchainLLMWrapper(ollama_llm)
            self.ragas_embeddings = LangchainEmbeddingsWrapper(ollama_embeddings)
            
            # RAGAS 0.3.x metrics (pass LLM and embeddings to evaluate())
            self.ragas_metrics = [
                answer_correctness,
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ]
        else:
            self.ragas_metrics = []
            self.ragas_llm = None
            self.ragas_embeddings = None
    
    async def initialize_rag_system(self):
        """Initialize the TPN RAG system with selected model."""
        print(f"Initializing TPN RAG system with {self.provider.upper()} model: {self.selected_model}")
        
        embedding_provider = OllamaEmbeddingProvider()
        vector_store = ChromaVectorStore()
        
        # Select the appropriate LLM provider based on provider type
        if self.provider == "openai":
            llm_provider = OpenAILLMProvider(default_model=self.selected_model)
            if not await llm_provider.check_health():
                raise RuntimeError(
                    "OpenAI API is not accessible. Please check:\n"
                    "  1. OPENAI_API_KEY is set correctly\n"
                    "  2. API key has sufficient credits\n"
                    "  3. Network connectivity"
                )
        elif self.provider == "xai":
            llm_provider = XAILLMProvider(default_model=self.selected_model)
            if not await llm_provider.check_health():
                raise RuntimeError(
                    "xAI API is not accessible. Please check:\n"
                    "  1. XAI_API_KEY is set correctly\n"
                    "  2. API key has sufficient credits\n"
                    "  3. Network connectivity"
                )
        elif self.provider == "gemini":
            llm_provider = GeminiLLMProvider(default_model=self.selected_model)
            if not await llm_provider.check_health():
                raise RuntimeError(
                    "Gemini API is not accessible. Please check:\n"
                    "  1. GEMINI_API_KEY is set correctly\n"
                    "  2. API key is valid\n"
                    "  3. Network connectivity"
                )
        elif self.provider == "kimi":
            llm_provider = KimiLLMProvider(default_model=self.selected_model)
            if not await llm_provider.check_health():
                raise RuntimeError(
                    "Kimi K2 API is not accessible. Please check:\n"
                    "  1. KIMI_API_KEY is set in .env\n"
                    "  2. API key is valid (from https://platform.moonshot.cn)\n"
                    "  3. Network connectivity"
                )
        else:  # ollama
            llm_provider = OllamaLLMProvider(default_model=self.selected_model)
            if not await llm_provider.check_health():
                raise RuntimeError("Ollama is not running. Please start Ollama service.")
        
        # Use RAG service with vector search (ChromaDB) + 2025 Advanced RAG features
        # Neo4j graph database is disabled for simplicity
        
        self.rag_service = HybridRAGService(
            embedding_provider=embedding_provider, 
            vector_store=vector_store, 
            llm_provider=llm_provider,
            # 2025 Advanced RAG Features (ENABLED)
            enable_advanced_2025=True  # Cross-encoder, HyDE, Query Rewriting, Adaptive Retrieval
        )
        
        stats = await self.rag_service.get_collection_stats()
        if stats["total_chunks"] == 0:
            raise RuntimeError("No TPN documents found. Please run 'uv run python main.py init' first.")
        
        print(f"RAG system ready: {stats['total_chunks']} chunks from {stats['total_documents']} documents")
    
    def cleanup(self):
        """Clean up resources."""
        pass  # No cleanup needed when Neo4j is disabled
    
    def load_mcq_questions(self) -> pd.DataFrame:
        """Load MCQ questions from CSV file."""
        print(f"Loading evaluation questions from {self.csv_path}")
        
        # Load CSV with keep_default_na=False to preserve "None" as string
        df = pd.read_csv(self.csv_path, keep_default_na=False, na_values=[''])
        
        print(f"Loaded {len(df)} MCQ questions from {self.csv_path}")
        print(f"Columns: {list(df.columns)}")
        
        # Verify required columns exist
        required_cols = ['ID', 'Question', 'Options', 'Corrrect Option (s)']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        return df
    
    def build_mcq_prompt_template(self) -> ChatPromptTemplate:
        """Build LangChain prompt template with few-shot examples."""
        
        # Example prompt for few-shot learning
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "Context: {{context}}\n\nQuestion: {{question}}\n\nOptions:\n{{options}}"),
            ("ai", "{{answer}}")
        ])
        
        # Few-shot prompt with examples
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=self.few_shot_examples,
        )
        
        # Final prompt template - STRICT: only use retrieved sources
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are evaluating a RAG (Retrieval Augmented Generation) system for TPN (Total Parenteral Nutrition) guidelines.

CRITICAL INSTRUCTIONS:
1. You will receive excerpts from 76 medical documents retrieved via vector search (ChromaDB embeddings)
2. Answer STRICTLY ONLY based on the information in these retrieved sources below
3. DO NOT use any prior medical knowledge from your training data
4. DO NOT guess or infer beyond what the retrieved sources explicitly state
5. If the retrieved sources lack sufficient information, select the best answer based ONLY on what's provided

This is a RAG evaluation - we are testing if retrieval finds the right information, not your medical knowledge.

Output format:
{format_instructions}

Select the correct answer(s) ONLY if supported by the retrieved clinical guidelines below."""),
            few_shot_prompt,
            ("human", """Below are the MOST RELEVANT EXCERPTS retrieved from our RAG system (76 medical documents via vector search):

RETRIEVED CLINICAL GUIDELINES:
{context}

{case_context}

MULTIPLE CHOICE QUESTION: {question}

OPTIONS:
{options}

IMPORTANT: 
- This is a multiple-choice question - select the correct answer letter(s) based on the retrieved guidelines
- Some questions may have ONE correct answer, MULTIPLE correct answers, or NONE correct
- Answer with single letter (e.g., "A") or multiple letters separated by commas (e.g., "A,B,C")
- Base your answer EXCLUSIVELY on the retrieved guidelines above
- Do not use medical knowledge from your training
- Answer in JSON format: {{"answer": "A", "confidence": "high"}}""")
        ])
        
        return final_prompt
    
    def normalize_answer(self, answer: str) -> str:
        """Normalize answer to handle edge cases."""
        answer = answer.strip().upper()
        
        if "ALL OF THE ABOVE" in answer:
            return "ALL"
        if answer in ["NONE", "NONE OF THE ABOVE"]:
            return "NONE"
        
        letters = re.findall(r'\b([A-F])\b', answer)
        if letters:
            return ",".join(sorted(letters))
        
        return answer
    
    def answers_match(self, model_answer: str, correct_answer: str, options_text: str) -> bool:
        """Check if model answer matches correct answer, handling 'All of the above' cases."""
        model_norm = self.normalize_answer(model_answer)
        correct_norm = self.normalize_answer(correct_answer)
        
        # Direct match
        if model_norm == correct_norm:
            return True
        
        # Check if model answered a single letter that represents "All of the above"
        if len(model_norm) == 1 and model_norm in "ABCDEF":
            # Check if this option contains "All of the above"
            option_pattern = rf"{model_norm}\.\s*(.+?)(?=\s*[A-F]\.|$)"
            match = re.search(option_pattern, options_text, re.IGNORECASE | re.DOTALL)
            if match:
                option_text = match.group(1).strip()
                if "ALL OF THE ABOVE" in option_text.upper():
                    # Extract all option letters from the correct answer
                    correct_letters = re.findall(r'\b([A-F])\b', correct_answer.upper())
                    if correct_letters:
                        # Check if correct answer is all available options
                        all_options = re.findall(r'([A-F])\.', options_text)
                        if sorted(correct_letters) == sorted(all_options):
                            return True
        
        return False
    
    async def evaluate_single_question(
        self,
        question_id: str,
        question: str,
        options: str,
        correct_option: str,
        case_context: str = ""
    ) -> Dict[str, Any]:
        """Evaluate a single MCQ question using modern LangChain approach."""
        
        print(f"\nEvaluating Question {question_id}: {question[:70]}...")
        
        try:
            correct_normalized = self.normalize_answer(correct_option)
            
            # STEP 1: Search with clean question
            full_question = f"{case_context}\n\n{question}".strip() if case_context else question
            
            # Reduced from 15 to 5 to prevent information overload
            search_query = SearchQuery(query=full_question, limit=5)
            search_response = await self.rag_service.search(search_query)
            
            if not search_response.results:
                raise RuntimeError("No search results found")
            
            # STEP 2: Build context from retrieved chunks (Vector Search)
            context_parts = []
            
            # Vector search results (ChromaDB)
            for i, result in enumerate(search_response.results, 1):
                doc_name = result.document_name[:50]
                context_parts.append(f"[ChromaDB Vector {i}: {doc_name}]\n{result.content}")
            
            context = "\n\n".join(context_parts)
            
            # STEP 3: Build structured prompt with LangChain
            prompt_template = self.build_mcq_prompt_template()
            
            formatted_prompt = prompt_template.format_messages(
                format_instructions=self.parser.get_format_instructions(),
                context=context,
                case_context=f"CLINICAL CASE:\n{case_context}\n" if case_context else "",
                question=question,
                options=options
            )
            
            # Convert messages to string for Ollama
            prompt_str = "\n\n".join([msg.content for msg in formatted_prompt])
            
            # STEP 4: Generate with structured output
            # Increased max_tokens for reasoning models (Grok, DeepSeek) that need space for <think> tags
            model_response = await self.rag_service.llm_provider.generate(
                prompt=prompt_str,
                temperature=0.0,
                max_tokens=8000,  # Very generous limit to prevent any truncation
                seed=42  # Fixed seed for reproducibility
            )
            
            # STEP 5: Parse structured output
            try:
                parsed_answer = self.parser.parse(model_response)
                model_answer_raw = parsed_answer.get("answer", "PARSE_ERROR").strip().upper()
                confidence = parsed_answer.get("confidence", "unknown")
            except Exception as parse_error:
                print(f"  Warning: JSON parsing failed, trying fallback extraction: {parse_error}")
                # Fallback: extract from text
                model_answer_raw = model_response.strip().upper()
                confidence = "unknown"
                
                # Remove reasoning tags first (for reasoning models like Grok, DeepSeek)
                cleaned_response = re.sub(r'<think>.*?</think>', '', model_response, flags=re.DOTALL | re.IGNORECASE)
                cleaned_response = re.sub(r'<thinking>.*?</thinking>', '', cleaned_response, flags=re.DOTALL | re.IGNORECASE)
                
                # Try to find JSON in cleaned response
                json_match = re.search(r'\{[^}]*"answer"\s*:\s*"([A-F])"[^}]*\}', cleaned_response, re.IGNORECASE)
                if json_match:
                    model_answer_raw = json_match.group(1).upper()
                    # Try to extract confidence too
                    conf_match = re.search(r'"confidence"\s*:\s*"(\w+)"', cleaned_response, re.IGNORECASE)
                    if conf_match:
                        confidence = conf_match.group(1).lower()
                else:
                    # Check for "All of the above" or "None" text responses
                    if "ALL OF THE ABOVE" in model_answer_raw:
                        model_answer_raw = "ALL OF THE ABOVE"
                    elif "NONE OF THE ABOVE" in model_answer_raw or model_answer_raw.strip() == "NONE":
                        model_answer_raw = "NONE"
                    else:
                        # Try to extract letters from malformed output
                        letter_matches = re.findall(r'\b([A-F])\b', model_answer_raw)
                        if letter_matches:
                            # If we got multiple letters (e.g., "A,A,A,B,B,B"), use frequency analysis
                            from collections import Counter
                            letter_counts = Counter(letter_matches)
                            
                            # If only one unique letter, use it
                            if len(letter_counts) == 1:
                                model_answer_raw = letter_matches[0]
                            # If multiple different letters, take the most frequent one
                            elif len(letter_counts) > 1:
                                most_common = letter_counts.most_common(1)[0][0]
                                print(f"  Warning: Multiple letters found {dict(letter_counts)}, using most frequent: {most_common}")
                                model_answer_raw = most_common
                            else:
                                model_answer_raw = letter_matches[0]  # Fallback to first
                        else:
                            model_answer_raw = "PARSE_ERROR"
            
            model_normalized = self.normalize_answer(model_answer_raw)
            is_correct = self.answers_match(model_answer_raw, correct_option, options)
            
            # Build result
            contexts = [source.content for source in search_response.results]
            
            result = {
                "question_id": question_id,
                "question": full_question,
                "options": options,
                "user_input": full_question,
                "response": model_response,
                "reference": correct_option,
                "retrieved_contexts": contexts,
                "correct_option": correct_option,
                "correct_normalized": correct_normalized,
                "model_answer": model_answer_raw,
                "model_normalized": model_normalized,
                "confidence": confidence,
                "is_correct": is_correct,
                "full_rag_answer": model_response,
                "num_sources": len(search_response.results),
                "response_time_ms": search_response.search_time_ms,
                "model_used": self.selected_model,
                "error": None
            }
            
            status = "CORRECT" if is_correct else "WRONG"
            print(f"  Expected: {correct_normalized}, Got: {model_normalized} ({confidence} confidence) - {status}")
            
            return result
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            return {
                "question_id": question_id,
                "question": question,
                "options": options,
                "user_input": question,
                "response": "ERROR",
                "reference": correct_option,
                "retrieved_contexts": [],
                "correct_option": correct_option,
                "correct_normalized": self.normalize_answer(correct_option),
                "model_answer": "ERROR",
                "model_normalized": "ERROR",
                "confidence": "none",
                "is_correct": False,
                "full_rag_answer": f"Error: {str(e)}",
                "num_sources": 0,
                "response_time_ms": 0,
                "model_used": self.selected_model,
                "error": str(e)
            }
    
    async def run_evaluation(self, max_questions: Optional[int] = None) -> Dict[str, Any]:
        """Run complete evaluation on MCQ questions."""
        
        await self.initialize_rag_system()
        mcq_df = self.load_mcq_questions()
        
        if max_questions and max_questions < len(mcq_df):
            mcq_df = mcq_df.head(max_questions)
            print(f"\nLimiting evaluation to first {max_questions} questions")
        
        total_questions = len(mcq_df)
        print(f"\n{'='*60}")
        print(f"Starting LangChain-based evaluation of {total_questions} MCQ questions")
        print(f"Model: {self.selected_model}")
        print(f"Using: Structured output with JSON parsing")
        print(f"{'='*60}\n")
        
        results = []
        for idx, row in mcq_df.iterrows():
            case_context = row.get('Case Context if available', '')
            if pd.isna(case_context):
                case_context = ''
            
            result = await self.evaluate_single_question(
                question_id=row['ID'],
                question=row['Question'],
                options=row['Options'],
                correct_option=row['Corrrect Option (s)'],
                case_context=case_context
            )
            results.append(result)
            
            if (len(results) % 10 == 0):
                correct_so_far = sum(1 for r in results if r['is_correct'])
                accuracy_so_far = (correct_so_far / len(results)) * 100
                print(f"\nProgress: {len(results)}/{total_questions} questions, Accuracy: {accuracy_so_far:.1f}%")
        
        # Calculate metrics
        correct_answers = sum(1 for r in results if r['is_correct'])
        wrong_answers = sum(1 for r in results if not r['is_correct'] and r['error'] is None)
        system_errors = sum(1 for r in results if r['error'] is not None)
        
        accuracy = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
        avg_response_time = sum(r['response_time_ms'] for r in results) / total_questions if total_questions > 0 else 0
        
        print(f"\n{'='*60}")
        print("EVALUATION COMPLETE")
        print(f"{'='*60}")
        print(f"Model: {self.selected_model}")
        print(f"Approach: LangChain structured output with JSON parsing")
        print(f"Total Questions: {total_questions}")
        print(f"Correct Answers: {correct_answers}")
        print(f"Wrong Answers: {wrong_answers}")
        print(f"System Errors: {system_errors}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Average Response Time: {avg_response_time:.1f}ms")
        
        
        # Display Advanced RAG Features Status
        print(f"\n🚀 Advanced RAG Features:")
        if hasattr(self.rag_service, 'advanced_rag') and self.rag_service.advanced_rag:
            advanced = self.rag_service.advanced_rag
            print(f"  - Reranking: {'✅ ENABLED' if advanced.rerank_config.enabled else '❌ disabled'} ({advanced.rerank_config.provider if advanced.rerank_config.enabled else 'N/A'})")
            print(f"  - Context Compression: {'✅ ENABLED' if advanced.compression_config.enabled else '❌ disabled'}")
            print(f"  - Query Decomposition: {'✅ ENABLED' if advanced.decomposition_config.enabled else '❌ disabled'}")
            print(f"  - Answer Validation: {'✅ ENABLED' if advanced.validation_config.enabled else '❌ disabled'}")
        else:
            print(f"  - ⚠️  Advanced RAG components not initialized")
        
        print(f"{'='*60}")
        
        # Get advanced RAG features configuration
        advanced_features_config = {}
        if hasattr(self.rag_service, 'advanced_rag') and self.rag_service.advanced_rag:
            advanced = self.rag_service.advanced_rag
            advanced_features_config = {
                "reranking_enabled": advanced.rerank_config.enabled,
                "reranking_provider": advanced.rerank_config.provider if advanced.rerank_config.enabled else None,
                "compression_enabled": advanced.compression_config.enabled,
                "compression_method": advanced.compression_config.method if advanced.compression_config.enabled else None,
                "query_decomposition_enabled": advanced.decomposition_config.enabled,
                "validation_enabled": advanced.validation_config.enabled
            }
        
        # Build evaluation summary
        evaluation_summary = {
            "model_used": self.selected_model,
            "approach": "rag_chromadb_advanced_2025",
            "total_questions": total_questions,
            "correct_answers": correct_answers,
            "wrong_answers": wrong_answers,
            "system_errors": system_errors,
            "accuracy": accuracy,
            "average_response_time_ms": avg_response_time,
            "advanced_rag_features": advanced_features_config,
            "individual_results": results,
            "ragas_metrics": {}
        }
        
        # RAGAS evaluation (optional)
        if RAGAS_AVAILABLE and results:
            ragas_data = {
                "question": [r["user_input"] for r in results if r["error"] is None],
                "answer": [r["response"] for r in results if r["error"] is None],
                "contexts": [r["retrieved_contexts"] for r in results if r["error"] is None],
                "reference": [r["reference"] for r in results if r["error"] is None]  # RAGAS 0.3.x uses 'reference' not 'ground_truths'
            }
            
            if ragas_data["question"]:
                print(f"\n{'='*60}")
                print("Computing RAGAS metrics...")
                print(f"{'='*60}")
                try:
                    ragas_dataset = Dataset.from_dict(ragas_data)
                    # RAGAS 0.3.x: Pass wrapped Ollama LLM and Embeddings to evaluate()
                    ragas_results = evaluate(
                        ragas_dataset, 
                        metrics=self.ragas_metrics,
                        llm=self.ragas_llm,  # Custom Ollama LLM
                        embeddings=self.ragas_embeddings  # Custom Ollama Embeddings
                    )
                    
                    # RAGAS 0.3.x: Result is a DataFrame, convert to dict
                    # Try multiple ways to extract metrics
                    if hasattr(ragas_results, 'to_pandas'):
                        ragas_df = ragas_results.to_pandas()
                        # Get mean of ONLY numeric metric columns (skip input columns)
                        ragas_metrics_dict = {}
                        # Input columns to exclude (not metrics)
                        input_columns = {'question', 'answer', 'contexts', 'reference', 'user_input', 'retrieved_contexts'}
                        
                        for col in ragas_df.columns:
                            if col not in input_columns:
                                # Only compute mean for numeric columns (actual metrics)
                                try:
                                    mean_value = ragas_df[col].mean()
                                    # Check if it's a valid numeric value
                                    if pd.notna(mean_value):
                                        ragas_metrics_dict[col] = float(mean_value)
                                    else:
                                        # Metric failed for all rows (all NaN)
                                        ragas_metrics_dict[col] = 0.0
                                        print(f"  Warning: Metric '{col}' had no valid values (all NaN)")
                                except (TypeError, ValueError) as e:
                                    # Skip non-numeric columns silently
                                    pass
                        
                        evaluation_summary["ragas_metrics"] = ragas_metrics_dict
                    elif hasattr(ragas_results, '__dict__'):
                        # Fallback: try to get attributes directly
                        evaluation_summary["ragas_metrics"] = {
                            k: v for k, v in ragas_results.__dict__.items() 
                            if not k.startswith('_')
                        }
                    else:
                        # Last resort: try to convert to dict directly
                        evaluation_summary["ragas_metrics"] = dict(ragas_results)
                    
                    print("\nRAGAS Metrics:")
                    for metric_name, score in evaluation_summary["ragas_metrics"].items():
                        print(f"  - {metric_name}: {score:.4f}")
                    print(f"{'='*60}")
                except Exception as e:
                    print(f"WARNING: RAGAS evaluation failed - {e}")
                    import traceback
                    print(f"Traceback: {traceback.format_exc()}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("eval")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_safe_name = self.selected_model.replace(':', '_').replace('/', '_')
        
        json_file = output_dir / f"tpn_eval_langchain_{model_safe_name}_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_summary, f, indent=4)
        print(f"\nDetailed results saved to: {json_file}")
        
        csv_file = output_dir / f"tpn_summary_langchain_{model_safe_name}_{timestamp}.csv"
        pd.DataFrame(results).to_csv(csv_file, index=False)
        print(f"CSV results saved to: {csv_file}")
        
        return evaluation_summary


async def get_available_ollama_models():
    """Get list of available Ollama LLM models (excludes embedding models)."""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                data = response.json()
                all_models = [model["name"] for model in data.get("models", [])]
                
                # Filter out embedding-only models
                embedding_keywords = ["embed", "embedding", "nomic-embed"]
                llm_models = [
                    model for model in all_models 
                    if not any(keyword in model.lower() for keyword in embedding_keywords)
                ]
                
                return llm_models
    except Exception:
        pass
    return []


async def get_available_openai_models():
    """Get list of available OpenAI models (if API key is set)."""
    try:
        from src.rag.config.settings import settings
        
        if not settings.openai_api_key:
            return []
        
        provider = OpenAILLMProvider()
        models = await provider.available_models
        return models
    except Exception as e:
        print(f"Warning: Could not fetch OpenAI models: {e}")
        return []


async def get_available_xai_models():
    """Get list of available xAI models (if API key is set)."""
    try:
        from src.rag.config.settings import settings
        
        if not settings.xai_api_key:
            return []
        
        provider = XAILLMProvider()
        models = await provider.available_models
        return models
    except Exception as e:
        print(f"Warning: Could not fetch xAI models: {e}")
        return []


async def get_available_gemini_models():
    """Get list of available Gemini models (if API key is set)."""
    try:
        from src.rag.config.settings import settings
        
        if not settings.gemini_api_key:
            return []
        
        provider = GeminiLLMProvider()
        models = await provider.available_models
        return models
    except Exception as e:
        print(f"Warning: Could not fetch Gemini models: {e}")
        return []


async def get_available_kimi_models():
    """Get list of available Kimi K2 models (if API key is set)."""
    try:
        from src.rag.config.settings import settings
        
        if not settings.kimi_api_key:
            return []
        
        provider = KimiLLMProvider()
        models = await provider.available_models
        return models
    except Exception as e:
        print(f"Warning: Could not fetch Kimi models: {e}")
        return []


async def get_all_available_models():
    """Get all available models from Ollama, OpenAI, xAI, Gemini, and Kimi K2."""
    ollama_models = await get_available_ollama_models()
    openai_models = await get_available_openai_models()
    xai_models = await get_available_xai_models()
    gemini_models = await get_available_gemini_models()
    kimi_models = await get_available_kimi_models()
    
    # Combine models with provider prefix for clarity
    all_models = []
    
    if ollama_models:
        all_models.extend([("ollama", model) for model in ollama_models])
    
    if openai_models:
        all_models.extend([("openai", model) for model in openai_models])
    
    if xai_models:
        all_models.extend([("xai", model) for model in xai_models])
    
    if gemini_models:
        all_models.extend([("gemini", model) for model in gemini_models])
    
    if kimi_models:
        all_models.extend([("kimi", model) for model in kimi_models])
    
    return all_models


def is_openai_model(model_name: str) -> bool:
    """Check if a model is from OpenAI."""
    openai_indicators = ["gpt-", "gpt4", "gpt5", "o1-", "o3-"]
    return any(indicator in model_name.lower() for indicator in openai_indicators)


def is_gemini_model(model_name: str) -> bool:
    """Check if a model is from Google Gemini."""
    return "gemini" in model_name.lower()


def select_model(available_models):
    """Interactive model selection from available models (Ollama + OpenAI + xAI + Gemini)."""
    if not available_models:
        print("\nERROR: No LLM models found.")
        print("Available options:")
        print("  1. Install Ollama models:")
        print("     ollama pull phi4:latest")
        print("     ollama pull mistral:7b")
        print("  2. Set OpenAI API key:")
        print("     export OPENAI_API_KEY=your_api_key")
        print("  3. Set Gemini API key:")
        print("     export GEMINI_API_KEY=your_api_key")
        print("  4. Set xAI API key:")
        print("     export XAI_API_KEY=your_api_key")
        return None, None
    
    print(f"\nAvailable LLM Models ({len(available_models)} found):")
    
    for i, (provider, model) in enumerate(available_models, 1):
        model_info = ""
        
        # Provider badge
        provider_badge = f"[{provider.upper()}]"
        
        # Model-specific info
        if provider == "ollama":
            size_patterns = {
                "120b": "120B parameters",
                "70b": "70B parameters",
                "27b": "27B parameters",
                "14b": "14B parameters",
                "13b": "13B parameters",
                "8b": "8B parameters",
                "7b": "7B parameters",
                "3b": "3B parameters"
            }
            
            for size, desc in size_patterns.items():
                if size in model.lower():
                    model_info = f" ({desc})"
                    break
            
            if "phi4" in model.lower():
                model_info = " (Phi-4, 14B params)"
            if "gpt-oss" in model.lower():
                model_info = " (GPT-OSS, 120B params)"
        
        elif provider == "openai":
            if "gpt" in model.lower():
                model_info = " (OpenAI)"
        
        elif provider == "xai":
            model_info = " (xAI Grok)"
        
        elif provider == "gemini":
            if "2.5-pro" in model.lower():
                model_info = " (Most capable, 1M context)"
            elif "2.5-flash" in model.lower():
                model_info = " (Fast & efficient, 1M context)"
            else:
                model_info = " (Google Gemini)"
        
        elif provider == "kimi":
            if "k2" in model.lower():
                model_info = " (1T MoE, 32B active, 256K context)"
            else:
                model_info = " (Moonshot AI)"
            
        print(f"  {i}. {provider_badge:<10} {model:<30} {model_info}")
    
    while True:
        try:
            choice = input(f"\nSelect model (1-{len(available_models)}) or press Enter for default: ").strip()
            
            if not choice:
                return available_models[0]
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(available_models):
                provider, model = available_models[choice_num - 1]
                return provider, model
            else:
                print(f"Please enter a number between 1 and {len(available_models)}")
        except ValueError:
            print("Please enter a valid number")


async def benchmark_all_models(max_questions: Optional[int] = None):
    """Benchmark all available models and generate comparison report."""
    
    print("\n" + "="*80)
    print("TPN RAG MULTI-MODEL BENCHMARK")
    print("="*80)
    
    csv_path = "eval/tpn_mcq_cleaned.csv"
    
    # Get all available models (Ollama + OpenAI)
    available_models = await get_all_available_models()
    if not available_models:
        print("ERROR: No models found!")
        return
    
    # Determine actual question count
    actual_question_count = max_questions if max_questions else 48
    
    print(f"\nFound {len(available_models)} models to benchmark:")
    for i, (provider, model) in enumerate(available_models, 1):
        print(f"  {i}. [{provider.upper()}] {model}")
    
    print(f"\nBenchmark settings:")
    print(f"  - Questions: {actual_question_count} MCQ")
    print(f"  - RAGAS Status: {'✅ Available' if RAGAS_AVAILABLE else '❌ Not Available (basic metrics only)'}")
    print(f"  - Metrics: Accuracy, Response Time" + (", RAGAS (faithfulness, relevancy, correctness)" if RAGAS_AVAILABLE else ""))
    print(f"  - System: 4x RTX 4090, 500GB disk")
    
    confirm = input(f"\nProceed with benchmark? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("Benchmark cancelled.")
        return
    
    # Run evaluation for each model
    all_results = []
    start_time = datetime.now()
    
    for i, (provider, model) in enumerate(available_models, 1):
        print(f"\n{'='*80}")
        print(f"BENCHMARKING MODEL {i}/{len(available_models)}: [{provider.upper()}] {model}")
        print(f"{'='*80}")
        
        try:
            evaluator = TPNRAGEvaluator(csv_path, model, provider)
            result = await evaluator.run_evaluation(max_questions)
            
            # Store key metrics
            model_label = f"[{provider.upper()}] {model}"
            all_results.append({
                "model": model_label,
                "provider": provider,
                "accuracy": result.get("accuracy", 0),
                "correct": result.get("correct_answers", 0),
                "wrong": result.get("wrong_answers", 0),
                "errors": result.get("system_errors", 0),
                "avg_time_ms": result.get("average_response_time_ms", 0),
                "ragas_metrics": result.get("ragas_metrics", {})
            })
            
            print(f"\n✅ {model_label} completed: {result.get('accuracy', 0):.2f}% accuracy")
            
        except Exception as e:
            print(f"\n❌ [{provider.upper()}] {model} failed: {e}")
            all_results.append({
                "model": f"[{provider.upper()}] {model}",
                "provider": provider,
                "accuracy": 0,
                "correct": 0,
                "wrong": 0,
                "errors": "BENCHMARK_FAILED",
                "avg_time_ms": 0,
                "ragas_metrics": {}
            })
    
    # Generate comparison report
    total_time = (datetime.now() - start_time).total_seconds()
    
    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE - COMPARISON REPORT")
    print(f"{'='*80}")
    print(f"Total benchmark time: {total_time/60:.1f} minutes")
    print(f"\nRANKING BY ACCURACY:")
    print(f"{'='*80}")
    
    # Sort by accuracy
    sorted_results = sorted(all_results, key=lambda x: x["accuracy"], reverse=True)
    
    print(f"\n{'Rank':<6} {'Model':<35} {'Accuracy':<12} {'Correct':<10} {'Avg Time':<12}")
    print("-" * 80)
    
    for rank, result in enumerate(sorted_results, 1):
        model = result["model"][:33]
        accuracy = f"{result['accuracy']:.2f}%"
        correct = f"{result['correct']}/{actual_question_count}"  # Use actual question count
        avg_time = f"{result['avg_time_ms']:.0f}ms"
        
        # Highlight top 3
        prefix = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        
        print(f"{prefix} {rank:<4} {model:<35} {accuracy:<12} {correct:<10} {avg_time:<12}")
    
    # RAGAS metrics comparison (if available)
    has_ragas_metrics = any(r.get("ragas_metrics") for r in all_results)
    
    if has_ragas_metrics:
        print(f"\n{'='*80}")
        print("RAGAS METRICS COMPARISON:")
        print(f"{'='*80}")
        print(f"\n{'Model':<35} {'Faithful':<12} {'Relevancy':<12} {'Correctness':<12}")
        print("-" * 80)
        
        for result in sorted_results:
            if result.get("ragas_metrics"):
                model = result["model"][:33]
                ragas = result["ragas_metrics"]
                faithful = f"{ragas.get('faithfulness', 0):.3f}"
                relevancy = f"{ragas.get('answer_relevancy', 0):.3f}"
                correctness = f"{ragas.get('answer_correctness', 0):.3f}"
                
                print(f"{model:<35} {faithful:<12} {relevancy:<12} {correctness:<12}")
    else:
        print(f"\n{'='*80}")
        print("RAGAS METRICS: Not Available")
        print(f"{'='*80}")
        print("RAGAS metrics were not computed for this benchmark.")
        print("To enable RAGAS:")
        print("  1. Install: uv pip install ragas datasets langchain-community")
        print("  2. Ensure Ollama models support RAGAS LLM wrapper")
        print("  3. Check eval logs for 'WARNING: RAGAS evaluation failed'")
    
    # Save benchmark report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("eval")
    
    benchmark_file = output_dir / f"benchmark_all_models_{timestamp}.json"
    with open(benchmark_file, 'w', encoding='utf-8') as f:
        json.dump({
            "benchmark_date": timestamp,
            "total_time_seconds": total_time,
            "models_tested": len(available_models),
            "total_questions": actual_question_count,  # Actual count, not limit
            "results": sorted_results
        }, f, indent=4)
    
    print(f"\n{'='*80}")
    print(f"Benchmark report saved: {benchmark_file}")
    print(f"{'='*80}")
    
    # Print recommendation
    if sorted_results:
        best_model = sorted_results[0]
        print(f"\n🏆 RECOMMENDATION: Use '{best_model['model']}' for TPN clinical questions")
        print(f"   - Accuracy: {best_model['accuracy']:.2f}%")
        print(f"   - Speed: {best_model['avg_time_ms']:.0f}ms avg response time")


async def main():
    """Main evaluation function."""
    
    print("TPN RAG System - LangChain Structured Output Evaluation")
    print("============================================================")
    
    csv_path = "eval/tpn_mcq_cleaned.csv"
    
    available_models = await get_all_available_models()
    if not available_models:
        return
    
    # Ask if user wants to benchmark all models
    print(f"\nEvaluation Mode:")
    print(f"  1. Single model evaluation")
    print(f"  2. Benchmark ALL models ({len(available_models)} available)")
    
    mode = input(f"\nSelect mode (1 or 2): ").strip()
    
    if mode == "2":
        # Benchmark all models
        max_questions_input = input(f"\nLimit questions for testing? (default: all, or enter number): ").strip()
        max_questions = None
        if max_questions_input and max_questions_input.lower() != 'all' and max_questions_input.isdigit():
            max_questions = int(max_questions_input)
        
        await benchmark_all_models(max_questions)
        return
    
    # Single model evaluation (original flow)
    selected_provider, selected_model = select_model(available_models)
    if not selected_provider or not selected_model:
        return
    
    print(f"Selected model: [{selected_provider.upper()}] {selected_model}")
    
    max_questions_input = input(f"\nLimit questions for testing? (default: all, or enter number): ").strip()
    max_questions = None
    if max_questions_input and max_questions_input.lower() != 'all' and max_questions_input.isdigit():
        max_questions = int(max_questions_input)
    
    evaluator = None
    try:
        evaluator = TPNRAGEvaluator(csv_path, selected_model, selected_provider)
        await evaluator.run_evaluation(max_questions)
    except RuntimeError as e:
        print(f"\nERROR: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up resources
        if evaluator:
            evaluator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())