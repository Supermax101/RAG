#!/usr/bin/env python3
"""
Baseline Model Evaluation - Direct Model Testing WITHOUT RAG
Tests the same models and questions as the RAG evaluation, but without any knowledge base access.
This provides baseline performance comparison to show RAG system value.
"""

import asyncio
import pandas as pd
import sys
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import httpx
from pydantic import BaseModel, Field

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import LLM providers for direct model access
from src.rag.infrastructure.llm_providers.ollama_provider import OllamaLLMProvider
from src.rag.infrastructure.llm_providers.openai_provider import OpenAILLMProvider
from src.rag.infrastructure.llm_providers.xai_provider import XAILLMProvider


@dataclass
class BaselineResult:
    """Results for a single baseline question evaluation."""
    question_id: str
    question: str
    correct_answer: str
    model_answer: str
    is_correct: bool
    response_time_ms: float
    model_confidence: str
    raw_response: str


class MCQAnswer(BaseModel):
    """Structured output for MCQ answers using Pydantic v2."""
    answer: str = Field(description="Single letter answer (A, B, C, D, E, or F)")
    confidence: Optional[str] = Field(default="medium", description="Confidence level: low, medium, high")


class BaselineModelEvaluator:
    """Evaluates raw model performance on TPN questions WITHOUT any RAG enhancement."""
    
    def __init__(self, csv_path: str, selected_model: str = "mistral:7b", provider: str = "ollama"):
        self.csv_path = csv_path
        self.selected_model = selected_model
        self.provider = provider
        
        # Initialize the appropriate LLM provider
        if provider == "openai":
            self.llm_provider = OpenAILLMProvider(default_model=selected_model)
        elif provider == "xai":
            self.llm_provider = XAILLMProvider(default_model=selected_model)
        else:  # ollama
            self.llm_provider = OllamaLLMProvider(default_model=selected_model)
        
        self.results: List[BaselineResult] = []
        
        # Load questions
        print(f"Loading TPN questions from: {csv_path}")
        self.questions_df = self.load_mcq_questions()
        print(f"Loaded {len(self.questions_df)} MCQ questions for baseline testing")
    
    def load_mcq_questions(self) -> pd.DataFrame:
        """Load MCQ questions from CSV."""
        df = pd.read_csv(self.csv_path)
        
        # Clean data - remove rows with missing critical fields
        df = df.dropna(subset=['Question', 'Options', 'Corrrect Option (s)'])
        
        print(f"\nLoaded {len(df)} MCQ questions from {self.csv_path}")
        print(f"\nQuestion Distribution by Source:")
        question_types = df.groupby('Doc Reference').size().sort_values(ascending=False)
        for doc, count in question_types.head(5).items():
            print(f"  - {doc}: {count} questions")
        
        return df
    
    def create_baseline_prompt(self, question: str, options: str, case_context: str = "") -> str:
        """Create a direct prompt without any RAG context - simple and clear."""
        
        prompt = """You are a medical expert in Total Parenteral Nutrition (TPN) answering MCQ (Multiple Choice Questions).

Answer the MCQ question. Respond in JSON format:
{"answer": "A", "confidence": "high"}

Where answer can be:
- A single letter (A-F) for single answers
- Comma-separated letters (A,B,C) for multiple correct answers
- Special text like "All of the above" or "None" when appropriate

Confidence is low/medium/high.

"""
        
        # Handle case context (could be NaN/float from pandas)
        if case_context and isinstance(case_context, str) and case_context.strip():
            prompt += f"CLINICAL CASE:\n{case_context}\n\n"
        
        prompt += f"""QUESTION: {question}

OPTIONS:
{options}

Select the correct answer(s). Most questions have one answer, but some may have multiple."""
        
        return prompt
    
    def normalize_answer(self, answer: str) -> str:
        """Normalize answer to handle edge cases."""
        answer = answer.strip().upper()
        
        if "ALL OF THE ABOVE" in answer:
            return "ALL"
        if answer in ["NONE", "NONE OF THE ABOVE"]:
            return "NONE"
        
        # Extract letters
        letters = re.findall(r'\b([A-F])\b', answer)
        if letters:
            return ",".join(sorted(letters))  # Return all letters, sorted
        
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
    ) -> BaselineResult:
        """Evaluate a single question using direct model inference (no RAG)."""
        
        print(f"\nQuestion {question_id}: {question[:70]}...")
        
        start_time = time.time()
        
        try:
            # Create baseline prompt (no context from documents)
            prompt = self.create_baseline_prompt(question, options, case_context)
            
            # Get direct model response
            # Increased max_tokens for reasoning models (Grok, DeepSeek) that use <think> tags
            raw_response = await self.llm_provider.generate(
                prompt=prompt,
                model=self.selected_model,
                temperature=0.0,  # Zero temperature for deterministic answers
                max_tokens=2000,  # Extra space to ensure reasoning completes before answer
                seed=42  # Fixed seed for reproducibility
            )
            
            response_time_ms = (time.time() - start_time) * 1000
            
            # Parse JSON response
            model_answer = "UNKNOWN"
            confidence = "low"
            
            try:
                # Remove reasoning tags first (for reasoning models like Grok, DeepSeek)
                cleaned_response = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL | re.IGNORECASE)
                cleaned_response = re.sub(r'<thinking>.*?</thinking>', '', cleaned_response, flags=re.DOTALL | re.IGNORECASE)
                
                # Try to extract JSON from cleaned response
                json_match = re.search(r'\{[^}]*\}', cleaned_response)
                if json_match:
                    response_json = json.loads(json_match.group())
                    model_answer = response_json.get("answer", "UNKNOWN")
                    confidence = response_json.get("confidence", "low")
                else:
                    # Fallback: extract letter from raw text with frequency analysis
                    letters = re.findall(r'\b([A-F])\b', cleaned_response.upper())
                    if letters:
                        from collections import Counter
                        letter_counts = Counter(letters)
                        
                        # If only one unique letter, use it
                        if len(letter_counts) == 1:
                            model_answer = letters[0]
                        # If multiple different letters (malformed output), use most frequent
                        elif len(letter_counts) > 1:
                            most_common = letter_counts.most_common(1)[0][0]
                            print(f"  Warning: Multiple letters found {dict(letter_counts)}, using most frequent: {most_common}")
                            model_answer = most_common
                        else:
                            model_answer = letters[0]
            except (json.JSONDecodeError, KeyError):
                # Last resort: simple regex on cleaned response
                cleaned_response = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL | re.IGNORECASE)
                letters = re.findall(r'\b([A-F])\b', cleaned_response.upper())
                if letters:
                    from collections import Counter
                    letter_counts = Counter(letters)
                    if len(letter_counts) == 1:
                        model_answer = letters[0]
                    elif len(letter_counts) > 1:
                        model_answer = letter_counts.most_common(1)[0][0]
                    else:
                        model_answer = letters[0]
            
            # Normalize answers for comparison
            correct_normalized = self.normalize_answer(correct_option)
            model_normalized = self.normalize_answer(model_answer)
            
            is_correct = self.answers_match(model_answer, correct_option, options)
            
            result = BaselineResult(
                question_id=question_id,
                question=question,
                correct_answer=correct_normalized,
                model_answer=model_normalized,
                is_correct=is_correct,
                response_time_ms=response_time_ms,
                model_confidence=confidence,
                raw_response=raw_response[:500]  # Truncate for storage
            )
            
            status = "CORRECT" if is_correct else "WRONG"
            print(f"   {status}: Expected '{correct_normalized}' -> Got '{model_normalized}' ({response_time_ms:.0f}ms)")
            
            return result
            
        except Exception as e:
            print(f"   WARNING Error: {e}")
            response_time_ms = (time.time() - start_time) * 1000
            
            return BaselineResult(
                question_id=question_id,
                question=question,
                correct_answer=self.normalize_answer(correct_option),
                model_answer="ERROR",
                is_correct=False,
                response_time_ms=response_time_ms,
                model_confidence="low",
                raw_response=f"Error: {str(e)}"
            )
    
    async def run_baseline_evaluation(self, max_questions: Optional[int] = None) -> Dict[str, Any]:
        """Run complete baseline evaluation without RAG system."""
        
        print(f"\nStarting BASELINE evaluation for model: {self.selected_model}")
        print(f"   Testing raw medical knowledge WITHOUT any document access")
        print(f"   Questions: {max_questions or len(self.questions_df)}")
        print("=" * 60)
        
        # Check if model is available (only for Ollama)
        if self.provider == "ollama":
            try:
                available_models = await get_available_ollama_models()
                if self.selected_model not in available_models:
                    print(f"WARNING: Model '{self.selected_model}' not found in available Ollama models")
                    print(f"Available models: {available_models}")
            except Exception as e:
                print(f"WARNING: Could not check available models: {e}")
        
        start_time = time.time()
        
        # Process questions
        questions_to_process = self.questions_df.head(max_questions) if max_questions else self.questions_df
        
        for idx, row in questions_to_process.iterrows():
            # Handle case context - pandas can return NaN as float
            case_context = row.get('Case Context if available', '')
            if not isinstance(case_context, str):
                case_context = ''
            
            result = await self.evaluate_single_question(
                question_id=str(row['ID']),
                question=row['Question'],
                options=row['Options'],
                correct_option=row['Corrrect Option (s)'],
                case_context=case_context
            )
            self.results.append(result)
            
            # Progress update
            if len(self.results) % 5 == 0:
                current_accuracy = sum(r.is_correct for r in self.results) / len(self.results) * 100
                print(f"\nProgress: {len(self.results)}/{len(questions_to_process)} | Accuracy: {current_accuracy:.1f}%")
        
        total_time = time.time() - start_time
        
        # Calculate comprehensive metrics
        evaluation_summary = self.calculate_metrics(total_time)
        
        # Save results
        self.save_results(evaluation_summary)
        
        return evaluation_summary
    
    def calculate_metrics(self, total_time: float) -> Dict[str, Any]:
        """Calculate comprehensive baseline evaluation metrics."""
        
        total_questions = len(self.results)
        correct_answers = sum(1 for r in self.results if r.is_correct)
        
        accuracy = (correct_answers / total_questions * 100) if total_questions > 0 else 0
        avg_response_time = sum(r.response_time_ms for r in self.results) / total_questions
        
        # Confidence distribution
        confidence_dist = {}
        for result in self.results:
            conf = result.model_confidence
            confidence_dist[conf] = confidence_dist.get(conf, 0) + 1
        
        # Performance by confidence level
        confidence_accuracy = {}
        for conf in ['low', 'medium', 'high']:
            conf_results = [r for r in self.results if r.model_confidence == conf]
            if conf_results:
                conf_accuracy = sum(r.is_correct for r in conf_results) / len(conf_results) * 100
                confidence_accuracy[conf] = {
                    'accuracy': conf_accuracy,
                    'count': len(conf_results)
                }
        
        summary = {
            'model_name': self.selected_model,
            'evaluation_type': 'BASELINE_NO_RAG',
            'timestamp': datetime.now().isoformat(),
            'total_questions': total_questions,
            'correct_answers': correct_answers,
            'accuracy_percentage': round(accuracy, 2),
            'avg_response_time_ms': round(avg_response_time, 1),
            'total_evaluation_time_seconds': round(total_time, 1),
            'confidence_distribution': confidence_dist,
            'confidence_accuracy': confidence_accuracy,
            'sample_errors': [
                {
                    'question_id': r.question_id,
                    'question': r.question[:100] + '...',
                    'expected': r.correct_answer,
                    'got': r.model_answer,
                    'confidence': r.model_confidence
                }
                for r in self.results if not r.is_correct
            ][:5]  # First 5 errors
        }
        
        # Print summary
        print("\n" + "="*60)
        print("BASELINE EVALUATION RESULTS (NO RAG)")
        print("="*60)
        print(f"Model: {self.selected_model}")
        print(f"Accuracy: {accuracy:.1f}% ({correct_answers}/{total_questions})")
        print(f"Avg Response Time: {avg_response_time:.0f}ms")
        print(f"Total Time: {total_time:.1f}s")
        
        print(f"\nConfidence Distribution:")
        for conf, count in confidence_dist.items():
            acc_info = confidence_accuracy.get(conf, {})
            acc_pct = acc_info.get('accuracy', 0)
            print(f"  - {conf.title()}: {count} questions ({acc_pct:.1f}% accuracy)")
        
        if summary['sample_errors']:
            print(f"\nSample Errors:")
            for error in summary['sample_errors'][:3]:
                print(f"  - Q{error['question_id']}: Expected '{error['expected']}' -> Got '{error['got']}'")
        
        return summary
    
    def save_results(self, summary: Dict[str, Any]):
        """Save evaluation results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_clean = self.selected_model.replace(":", "_")
        
        # JSON file with complete results
        json_file = f"baseline_evaluation_results_{model_clean}_{timestamp}.json"
        json_path = Path("eval") / json_file
        
        json_data = {
            'summary': summary,
            'detailed_results': [
                {
                    'question_id': r.question_id,
                    'question': r.question,
                    'correct_answer': r.correct_answer,
                    'model_answer': r.model_answer,
                    'is_correct': r.is_correct,
                    'response_time_ms': r.response_time_ms,
                    'confidence': r.model_confidence,
                    'raw_response': r.raw_response
                }
                for r in self.results
            ]
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # CSV file for easy analysis
        csv_file = f"baseline_evaluation_summary_{timestamp}.csv"
        csv_path = Path("eval") / csv_file
        
        df_results = pd.DataFrame([
            {
                'Question_ID': r.question_id,
                'Question': r.question,
                'Correct_Answer': r.correct_answer,
                'Model_Answer': r.model_answer,
                'Is_Correct': r.is_correct,
                'Response_Time_ms': r.response_time_ms,
                'Confidence': r.model_confidence,
                'Model': self.selected_model
            }
            for r in self.results
        ])
        df_results.to_csv(csv_path, index=False)
        
        print(f"\nResults saved:")
        print(f"   JSON: {json_file}")
        print(f"   CSV: {csv_file}")


async def get_available_ollama_models() -> List[str]:
    """Get list of available Ollama LLM models (excludes embedding models)."""
    try:
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
    return ["mistral:7b", "llama3:8b"]  # Fallback defaults


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


async def get_all_available_models():
    """Get all available models from Ollama, OpenAI, and xAI."""
    ollama_models = await get_available_ollama_models()
    openai_models = await get_available_openai_models()
    xai_models = await get_available_xai_models()
    
    # Combine models with provider prefix for clarity
    all_models = []
    
    if ollama_models:
        all_models.extend([("ollama", model) for model in ollama_models])
    
    if openai_models:
        all_models.extend([("openai", model) for model in openai_models])
    
    if xai_models:
        all_models.extend([("xai", model) for model in xai_models])
    
    return all_models


def select_model(available_models):
    """Interactive model selection from available models (Ollama + OpenAI)."""
    if not available_models:
        print("\nERROR: No LLM models found.")
        print("Available options:")
        print("  1. Install Ollama models: ollama pull mistral:7b")
        print("  2. Set OpenAI API key: export OPENAI_API_KEY=your_key")
        return None, None
    
    print(f"\nAvailable LLM Models ({len(available_models)} found):")
    
    for i, (provider, model) in enumerate(available_models, 1):
        provider_badge = f"[{provider.upper()}]"
        model_info = ""
        
        if provider == "openai":
            if "gpt" in model.lower():
                model_info = " (OpenAI)"
        
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
        except KeyboardInterrupt:
            print("\nSelection cancelled.")
            return None, None


async def benchmark_all_baseline_models(max_questions: Optional[int] = None):
    """Benchmark all available models in baseline mode (no RAG)."""
    
    print("\n" + "="*80)
    print("BASELINE MODEL BENCHMARK (NO RAG KNOWLEDGE)")
    print("="*80)
    
    csv_path = "eval/tpn_mcq_questions_clean.csv"
    
    # Get all available models (Ollama + OpenAI)
    available_models = await get_all_available_models()
    if not available_models:
        print("ERROR: No models found!")
        return
    
    actual_question_count = max_questions if max_questions else 48
    
    print(f"\nFound {len(available_models)} models to benchmark in baseline mode:")
    for i, (provider, model) in enumerate(available_models, 1):
        print(f"  {i}. [{provider.upper()}] {model}")
    
    print(f"\nBaseline Benchmark Settings:")
    print(f"  - Questions: {actual_question_count} MCQ")
    print(f"  - Mode: DIRECT MODEL TESTING (No RAG/Document Access)")
    print(f"  - Metrics: Raw Accuracy, Response Time, Confidence Analysis")
    print(f"  - Purpose: Establish baseline performance for RAG comparison")
    
    confirm = input(f"\nProceed with baseline benchmark? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("Benchmark cancelled.")
        return
    
    # Run baseline evaluation for each model
    all_results = []
    start_time = datetime.now()
    
    for i, (provider, model) in enumerate(available_models, 1):
        print(f"\n{'='*60}")
        print(f"Baseline Testing Model {i}/{len(available_models)}: [{provider.upper()}] {model}")
        print(f"{'='*60}")
        
        try:
            evaluator = BaselineModelEvaluator(csv_path, model, provider)
            result = await evaluator.run_baseline_evaluation(actual_question_count)
            all_results.append(result)
            
        except Exception as e:
            print(f"ERROR testing [{provider.upper()}] {model}: {e}")
            continue
    
    # Generate comparison report
    if all_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_file = f"baseline_model_comparison_{timestamp}.json"
        
        comparison_data = {
            'benchmark_type': 'BASELINE_NO_RAG',
            'timestamp': datetime.now().isoformat(),
            'total_models_tested': len(all_results),
            'questions_per_model': actual_question_count,
            'model_results': all_results
        }
        
        with open(Path("eval") / comparison_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        # Print comparison summary
        print(f"\n{'='*80}")
        print("BASELINE MODEL COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        # Sort by accuracy
        sorted_results = sorted(all_results, key=lambda x: x['accuracy_percentage'], reverse=True)
        
        print(f"Model Ranking (by raw accuracy without RAG):")
        for i, result in enumerate(sorted_results, 1):
            accuracy = result['accuracy_percentage']
            time_ms = result['avg_response_time_ms']
            print(f"  {i}. {result['model_name']:<20} {accuracy:>5.1f}% accuracy ({time_ms:>4.0f}ms avg)")
        
        best_model = sorted_results[0]
        print(f"\nBest Baseline Performance:")
        print(f"   - Model: {best_model['model_name']}")
        print(f"   - Raw Accuracy: {best_model['accuracy_percentage']:.1f}% (without any document knowledge)")
        print(f"   - Speed: {best_model['avg_response_time_ms']:.0f}ms avg response time")
        
        print(f"\nComparison report saved: {comparison_file}")


async def main():
    """Main baseline evaluation function."""
    
    print("TPN Baseline Model Evaluation - Raw Model Performance Test")
    print("=" * 65)
    print("Purpose: Test models WITHOUT RAG system to establish baseline")
    print("No document access - pure model medical knowledge only")
    print("=" * 65)
    
    csv_path = "eval/tpn_mcq_questions_clean.csv"
    
    available_models = await get_all_available_models()
    if not available_models:
        print("ERROR: No models found!")
        return
    
    # Ask evaluation mode
    print(f"\nBaseline Evaluation Mode:")
    print(f"  1. Single model baseline test")
    print(f"  2. Benchmark ALL models baseline ({len(available_models)} available)")
    
    mode = input(f"\nSelect mode (1 or 2): ").strip()
    
    if mode == "2":
        # Benchmark all models in baseline mode
        max_questions_input = input(f"\nLimit questions for testing? (default: all, or enter number): ").strip()
        max_questions = None
        if max_questions_input and max_questions_input.lower() != 'all' and max_questions_input.isdigit():
            max_questions = int(max_questions_input)
        
        await benchmark_all_baseline_models(max_questions)
        return
    
    # Single model baseline evaluation
    selected_provider, selected_model = select_model(available_models)
    if not selected_provider or not selected_model:
        return
    
    print(f"Selected model: [{selected_provider.upper()}] {selected_model}")
    
    max_questions_input = input(f"\nLimit questions for testing? (default: all, or enter number): ").strip()
    max_questions = None
    if max_questions_input and max_questions_input.lower() != 'all' and max_questions_input.isdigit():
        max_questions = int(max_questions_input)
    
    try:
        evaluator = BaselineModelEvaluator(csv_path, selected_model, selected_provider)
        await evaluator.run_baseline_evaluation(max_questions)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
