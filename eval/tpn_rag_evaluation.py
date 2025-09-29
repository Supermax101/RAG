#!/usr/bin/env python3
"""
TPN RAG System Evaluation using RAGAS

This script evaluates the TPN RAG system against MCQ questions using proper RAGAS metrics.
"""

import asyncio
import pandas as pd
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import re

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import RAG system components
from src.rag.infrastructure.embeddings.ollama_embeddings import OllamaEmbeddingProvider
from src.rag.infrastructure.vector_stores.chroma_store import ChromaVectorStore
from src.rag.infrastructure.llm_providers.ollama_provider import OllamaLLMProvider
from src.rag.core.services.rag_service import RAGService
from src.rag.core.models.documents import RAGQuery, SearchQuery

# RAGAS imports
try:
    from ragas import evaluate, SingleTurnSample
    from ragas.metrics import (
        answer_correctness,
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: RAGAS not available - {e}")
    print("Install RAGAS: pip install ragas")
    RAGAS_AVAILABLE = False
    sys.exit(1)


class TPNRAGEvaluator:
    """Evaluates TPN RAG system using RAGAS metrics for MCQ questions."""
    
    def __init__(self, csv_path: str, selected_model: str = "mistral:7b"):
        self.csv_path = csv_path
        self.selected_model = selected_model
        self.rag_service = None
        self.evaluation_results = []
        
    async def initialize_rag_system(self):
        """Initialize the TPN RAG system with selected model."""
        print(f"Initializing TPN RAG system with model: {self.selected_model}")
        
        # Initialize providers
        embedding_provider = OllamaEmbeddingProvider()
        vector_store = ChromaVectorStore()
        llm_provider = OllamaLLMProvider(default_model=self.selected_model)
        
        # Check Ollama health
        if not await llm_provider.check_health():
            raise RuntimeError("ERROR: Ollama is not running. Please start Ollama service.")
        
        # Create RAG service
        self.rag_service = RAGService(embedding_provider, vector_store, llm_provider)
        
        # Verify we have TPN documents loaded
        stats = await self.rag_service.get_collection_stats()
        if stats["total_chunks"] == 0:
            raise RuntimeError("ERROR: No TPN documents found. Run 'uv run python main.py init' first.")
        
        print(f"RAG system ready: {stats['total_chunks']} chunks from {stats['total_documents']} documents")
    
    def load_mcq_questions(self) -> pd.DataFrame:
        """Load and filter MCQ questions from CSV file."""
        print(f"Loading evaluation questions from {self.csv_path}")
        
        df = pd.read_csv(self.csv_path)
        
        # Filter for MCQ questions only
        mcq_df = df[df['Answer Type'] == 'mcq_single'].copy()
        
        if mcq_df.empty:
            raise ValueError("ERROR: No MCQ questions found in CSV file")
        
        print(f"Loaded {len(mcq_df)} MCQ questions (filtered from {len(df)} total questions)")
        
        # Show breakdown
        question_types = df['Answer Type'].value_counts()
        print("\nQuestion type breakdown:")
        for qtype, count in question_types.items():
            print(f"  - {qtype}: {count} questions")
        
        return mcq_df
    
    def create_mcq_prompt(self, question: str, options: str, case_context: str = "") -> str:
        """Create a concise prompt for MCQ responses to avoid context overflow."""
        
        context_section = f"\nContext: {case_context}\n" if case_context else ""
        
        prompt = f"""Answer this TPN clinical question with ONLY a single letter (A-F). No explanation.
{context_section}
Question: {question}

Options:
{options}

Answer:"""

        return prompt
    
    def normalize_answer(self, answer: str) -> str:
        """Normalize answer to handle edge cases."""
        answer = answer.strip().upper()
        
        # Handle text answers
        if "ALL OF THE ABOVE" in answer:
            return "ALL"
        if answer in ["NONE", "NONE OF THE ABOVE"]:
            return "NONE"
        
        # Extract letters from multi-answer format (e.g., "B and D", "A,B,C,D", "A, D")
        letters = re.findall(r'\b([A-F])\b', answer)
        if letters:
            return ",".join(sorted(letters))  # Normalize as comma-separated sorted list
        
        return answer
    
    async def evaluate_single_question(
        self,
        question_id: str,
        question: str,
        options: str,
        correct_option: str,
        case_context: str = ""
    ) -> Dict[str, Any]:
        """Evaluate a single MCQ question and return results."""
        
        print(f"\nEvaluating Question {question_id}: {question[:70]}...")
        
        try:
            # Normalize correct option
            correct_normalized = self.normalize_answer(correct_option)
            
            # Create MCQ-specific prompt
            full_question = f"{case_context}\n\n{question}".strip() if case_context else question
            mcq_prompt = self.create_mcq_prompt(question, options, case_context)
            
            # Use RAG service ask() method - it handles search + generation
            rag_query = RAGQuery(
                question=full_question,  # Use question + case context for search
                search_limit=5,
                temperature=0.0
            )
            rag_response = await self.rag_service.ask(rag_query)
            
            # Extract and normalize model answer
            model_answer_raw = rag_response.answer.strip().upper()
            
            # Try to extract answer from response
            if len(model_answer_raw) > 1:
                # Look for single letter answer
                match = re.search(r'\b([A-F])\b', model_answer_raw)
                if match:
                    model_answer_raw = match.group(1)
                else:
                    # Check for text answers
                    if "ALL OF THE ABOVE" in model_answer_raw:
                        model_answer_raw = "ALL"
                    elif "NONE" in model_answer_raw:
                        model_answer_raw = "NONE"
                    else:
                        # Take first character if it's a letter
                        model_answer_raw = model_answer_raw[0] if model_answer_raw and model_answer_raw[0].isalpha() else "PARSE_ERROR"
            
            model_normalized = self.normalize_answer(model_answer_raw)
            
            # Check if correct
            is_correct = model_normalized == correct_normalized
            
            # Prepare context strings for RAGAS
            contexts = [source.content for source in rag_response.sources]
            
            result = {
                "question_id": question_id,
                "question": full_question,
                "options": options,
                "user_input": full_question,  # For RAGAS
                "response": rag_response.answer,  # For RAGAS (full answer)
                "reference": correct_option,  # For RAGAS (ground truth)
                "retrieved_contexts": contexts,  # For RAGAS
                "correct_option": correct_option,
                "correct_normalized": correct_normalized,
                "model_answer": model_answer_raw,
                "model_normalized": model_normalized,
                "is_correct": is_correct,
                "full_rag_answer": rag_response.answer,
                "num_sources": len(rag_response.sources),
                "response_time_ms": rag_response.total_time_ms,
                "model_used": self.selected_model,
                "error": None
            }
            
            status = "CORRECT" if is_correct else "WRONG"
            print(f"  Expected: {correct_normalized}, Got: {model_normalized} - {status}")
            
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
                "model_answer": "ERROR",
                "is_correct": False,
                "full_rag_answer": f"Error: {str(e)}",
                "num_sources": 0,
                "response_time_ms": 0,
                "model_used": self.selected_model,
                "error": str(e)
            }
    
    async def run_evaluation(self, max_questions: Optional[int] = None) -> Dict[str, Any]:
        """Run complete evaluation on MCQ questions."""
        
        # Initialize
        await self.initialize_rag_system()
        mcq_df = self.load_mcq_questions()
        
        # Limit questions if requested
        if max_questions and max_questions < len(mcq_df):
            mcq_df = mcq_df.head(max_questions)
            print(f"\nLimiting evaluation to first {max_questions} questions")
        
        total_questions = len(mcq_df)
        print(f"\n{'='*60}")
        print(f"Starting evaluation of {total_questions} MCQ questions")
        print(f"Model: {self.selected_model}")
        print(f"{'='*60}\n")
        
        # Evaluate each question
        results = []
        for idx, row in mcq_df.iterrows():
            # Handle NaN values properly
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
            
            # Progress update every 10 questions
            if (len(results) % 10 == 0):
                correct_so_far = sum(1 for r in results if r['is_correct'])
                accuracy_so_far = (correct_so_far / len(results)) * 100
                print(f"\nProgress: {len(results)}/{total_questions} questions, Accuracy so far: {accuracy_so_far:.1f}%")
        
        self.evaluation_results = results
        return self._generate_summary(results)
    
    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate evaluation summary with basic statistics."""
        
        total = len(results)
        correct = sum(1 for r in results if r['is_correct'])
        errors = sum(1 for r in results if r.get('error'))
        wrong = total - correct - errors
        accuracy = (correct / total * 100) if total > 0 else 0
        avg_time = sum(r['response_time_ms'] for r in results) / total if total > 0 else 0
        
        return {
            "total_questions": total,
            "correct_answers": correct,
            "wrong_answers": wrong,
            "system_errors": errors,
            "accuracy_percent": accuracy,
            "average_response_time_ms": avg_time,
            "model_used": self.selected_model,
            "individual_results": results
        }
    
    def prepare_ragas_dataset(self) -> Dataset:
        """Prepare dataset in RAGAS format."""
        
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run evaluation first.")
        
        # Filter out error cases for RAGAS evaluation
        valid_results = [r for r in self.evaluation_results if not r.get('error')]
        
        if not valid_results:
            raise ValueError("No valid results to evaluate with RAGAS")
        
        # Prepare data for RAGAS
        ragas_data = {
            "user_input": [r["user_input"] for r in valid_results],
            "response": [r["full_rag_answer"] for r in valid_results],
            "reference": [r["correct_option"] for r in valid_results],
            "retrieved_contexts": [r["retrieved_contexts"] for r in valid_results]
        }
        
        return Dataset.from_dict(ragas_data)
    
    def save_results(self, summary: Dict[str, Any], ragas_scores: Optional[Dict] = None, output_dir: str = "eval"):
        """Save evaluation results to JSON and CSV files."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_safe_name = self.selected_model.replace(':', '_').replace('/', '_')
        
        # Save detailed results
        json_file = Path(output_dir) / f"tpn_evaluation_{model_safe_name}_{timestamp}.json"
        json_file.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            "evaluation_summary": summary,
            "ragas_scores": ragas_scores if ragas_scores else {},
            "timestamp": timestamp,
            "model": self.selected_model
        }
        
        with open(json_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nDetailed results saved to: {json_file}")
        
        # Save summary CSV
        csv_file = Path(output_dir) / f"tpn_summary_{model_safe_name}_{timestamp}.csv"
        
        results_df = pd.DataFrame(summary['individual_results'])
        results_df.to_csv(csv_file, index=False)
        
        print(f"CSV results saved to: {csv_file}")
        
        return json_file, csv_file


async def get_available_ollama_models():
    """Get list of available Ollama models."""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
    except Exception:
        pass
    return []


def select_ollama_model(available_models):
    """Interactive model selection from available Ollama models."""
    if not available_models:
        print("\nERROR: No Ollama models found.")
        print("Please pull models first:")
        print("  ollama pull mistral:7b")
        print("  ollama pull llama3:8b")
        return None
    
    print(f"\nAvailable Ollama Models ({len(available_models)} found):")
    for i, model in enumerate(available_models, 1):
        size = ""
        for param_size in ["7b", "8b", "13b", "70b"]:
            if param_size in model.lower():
                size = f" ({param_size.upper()} parameters)"
                break
        print(f"  {i}. {model}{size}")
    
    while True:
        try:
            choice = input(f"\nSelect model (1-{len(available_models)}) or press Enter for default: ").strip()
            
            if not choice:
                return available_models[0]
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(available_models):
                return available_models[choice_num - 1]
            else:
                print(f"ERROR: Please enter a number between 1 and {len(available_models)}")
        except ValueError:
            print("ERROR: Please enter a valid number")


async def main():
    """Main evaluation function."""
    
    print("="*60)
    print("TPN RAG System - RAGAS-Based Clinical Evaluation")
    print("="*60)
    
    csv_path = "eval/tpn_eval_questions.csv"
    
    if not Path(csv_path).exists():
        print(f"\nERROR: CSV file not found: {csv_path}")
        return
    
    # Model selection
    print("\nChecking available Ollama models...")
    available_models = await get_available_ollama_models()
    
    if not available_models:
        print("\nERROR: No Ollama models available.")
        print("Please start Ollama and pull some models:")
        print("  ollama pull mistral:7b")
        print("  ollama pull llama3:8b")
        return
    
    selected_model = select_ollama_model(available_models)
    if not selected_model:
        return
    
    print(f"\nSelected model: {selected_model}")
    
    # Question limit
    max_questions = None
    user_limit = input("\nLimit questions for testing? (default: all, or enter number): ").strip()
    if user_limit and user_limit.lower() != 'all':
        try:
            max_questions = int(user_limit)
        except ValueError:
            print("Invalid number, using all questions")
    
    try:
        # Initialize evaluator
        evaluator = TPNRAGEvaluator(csv_path, selected_model)
        
        # Run evaluation
        print("\n" + "="*60)
        summary = await evaluator.run_evaluation(max_questions)
        
        # Compute RAGAS metrics
        print("\n" + "="*60)
        print("Computing RAGAS metrics...")
        print("="*60)
        
        try:
            ragas_dataset = evaluator.prepare_ragas_dataset()
            
            # Run RAGAS evaluation
            ragas_result = evaluate(
                ragas_dataset,
                metrics=[
                    answer_correctness,
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall
                ]
            )
            
            ragas_scores = ragas_result.to_pandas().to_dict('records')[0] if len(ragas_result) > 0 else {}
            
        except Exception as e:
            print(f"\nWARNING: RAGAS evaluation failed - {e}")
            print("Continuing with basic metrics only")
            ragas_scores = None
        
        # Save results
        evaluator.save_results(summary, ragas_scores)
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print("="*60)
        print(f"Model: {summary['model_used']}")
        print(f"Total Questions: {summary['total_questions']}")
        print(f"Correct Answers: {summary['correct_answers']}")
        print(f"Wrong Answers: {summary['wrong_answers']}")
        print(f"System Errors: {summary['system_errors']}")
        print(f"Accuracy: {summary['accuracy_percent']:.2f}%")
        print(f"Average Response Time: {summary['average_response_time_ms']:.1f}ms")
        
        if ragas_scores:
            print("\nRAGAS Metrics:")
            print("="*60)
            for metric, score in ragas_scores.items():
                if isinstance(score, (int, float)):
                    print(f"  {metric}: {score:.4f}")
            print("\nNote: RAGAS scores range from 0.0 to 1.0 (higher is better)")
            print("Target for production: >0.90 for all metrics")
        
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: Evaluation failed - {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())