#!/usr/bin/env python3
"""
RAG System Evaluation using BLEU, ROUGE, and other metrics

This script evaluates the quality of RAG-generated answers against reference answers
using multiple metrics including BLEU scores.
"""

import json
import time
import statistics
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for compatibility
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import sacrebleu
    from rouge_score import rouge_scorer
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
except ImportError as e:
    print(f"Missing evaluation dependencies. Please install:")
    print("pip install sacrebleu rouge-score nltk")
    print(f"Error: {e}")
    exit(1)

from test_rag import rag_answer, get_available_ollama_models, select_ollama_model


@dataclass
class EvaluationResult:
    """Single evaluation result for a question."""
    question: str
    reference_answer: str
    generated_answer: str
    bleu_score: float
    rouge_1_f: float
    rouge_2_f: float
    rouge_l_f: float
    response_time: float
    num_sources: int


@dataclass
class EvaluationSummary:
    """Summary of all evaluation results."""
    results: List[EvaluationResult]
    avg_bleu: float
    avg_rouge_1: float
    avg_rouge_2: float
    avg_rouge_l: float
    avg_response_time: float
    total_questions: int
    model_name: str


@dataclass
class ModelComparisonResult:
    """Results comparing multiple models."""
    model_summaries: List[EvaluationSummary]
    comparison_df: pd.DataFrame
    best_model: str
    worst_model: str


class RAGEvaluator:
    """Evaluate RAG system performance using multiple metrics."""
    
    def __init__(self, model_name: str = "mistral:7b"):
        self.model_name = model_name
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction()
        
        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt')
    
    def calculate_bleu(self, reference: str, generated: str) -> float:
        """Calculate BLEU score using sacrebleu."""
        try:
            # Use sacrebleu for standardized BLEU calculation
            bleu = sacrebleu.sentence_bleu(generated, [reference])
            return bleu.score / 100.0  # Convert to 0-1 scale
        except:
            # Fallback to NLTK BLEU
            ref_tokens = nltk.word_tokenize(reference.lower())
            gen_tokens = nltk.word_tokenize(generated.lower())
            return sentence_bleu([ref_tokens], gen_tokens, smoothing_function=self.smoothing.method1)
    
    def calculate_rouge(self, reference: str, generated: str) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        scores = self.rouge_scorer.score(reference, generated)
        return {
            'rouge_1_f': scores['rouge1'].fmeasure,
            'rouge_2_f': scores['rouge2'].fmeasure,
            'rouge_l_f': scores['rougeL'].fmeasure
        }
    
    def evaluate_single_question(self, question: str, reference_answer: str) -> EvaluationResult:
        """Evaluate a single question-answer pair."""
        start_time = time.time()
        
        # Generate answer using RAG system
        rag_result = rag_answer(question, model=self.model_name)
        
        response_time = time.time() - start_time
        generated_answer = rag_result['answer']
        
        # Calculate metrics
        bleu_score = self.calculate_bleu(reference_answer, generated_answer)
        rouge_scores = self.calculate_rouge(reference_answer, generated_answer)
        
        return EvaluationResult(
            question=question,
            reference_answer=reference_answer,
            generated_answer=generated_answer,
            bleu_score=bleu_score,
            rouge_1_f=rouge_scores['rouge_1_f'],
            rouge_2_f=rouge_scores['rouge_2_f'],
            rouge_l_f=rouge_scores['rouge_l_f'],
            response_time=response_time,
            num_sources=len(rag_result.get('sources', []))
        )
    
    def evaluate_dataset(self, test_data: List[Dict[str, str]]) -> EvaluationSummary:
        """Evaluate entire test dataset."""
        print(f"Evaluating {len(test_data)} questions with model: {self.model_name}")
        print("=" * 60)
        
        results = []
        
        for i, item in enumerate(test_data, 1):
            question = item['question']
            reference = item['reference_answer']
            
            print(f"[{i}/{len(test_data)}] Evaluating: {question[:50]}...")
            
            try:
                result = self.evaluate_single_question(question, reference)
                results.append(result)
                
                print(f"  BLEU: {result.bleu_score:.3f} | "
                      f"ROUGE-1: {result.rouge_1_f:.3f} | "
                      f"Time: {result.response_time:.1f}s")
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        # Calculate averages
        if results:
            avg_bleu = statistics.mean(r.bleu_score for r in results)
            avg_rouge_1 = statistics.mean(r.rouge_1_f for r in results)
            avg_rouge_2 = statistics.mean(r.rouge_2_f for r in results)
            avg_rouge_l = statistics.mean(r.rouge_l_f for r in results)
            avg_response_time = statistics.mean(r.response_time for r in results)
        else:
            avg_bleu = avg_rouge_1 = avg_rouge_2 = avg_rouge_l = avg_response_time = 0.0
        
        return EvaluationSummary(
            results=results,
            avg_bleu=avg_bleu,
            avg_rouge_1=avg_rouge_1,
            avg_rouge_2=avg_rouge_2,
            avg_rouge_l=avg_rouge_l,
            avg_response_time=avg_response_time,
            total_questions=len(test_data),
            model_name=self.model_name
        )


def create_sample_test_data() -> List[Dict[str, str]]:
    """Create sample test data for evaluation."""
    return [
        {
            "question": "What is the daily sodium requirement for adults?",
            "reference_answer": "According to ASPEN guidelines, the daily sodium requirement for adults is typically 1-2 mEq/kg/day (23-46 mg/kg/day), with adjustments based on clinical condition and fluid status."
        },
        {
            "question": "How should magnesium be supplemented in parenteral nutrition?",
            "reference_answer": "Magnesium supplementation in parenteral nutrition should provide 8-20 mEq/day for adults, administered as magnesium sulfate or magnesium chloride, with monitoring of serum magnesium levels every 2-3 days."
        },
        {
            "question": "What are the contraindications for enteral nutrition?",
            "reference_answer": "Contraindications for enteral nutrition include severe gastrointestinal bleeding, complete bowel obstruction, severe malabsorption, and hemodynamic instability requiring vasopressor support."
        },
        {
            "question": "What is the recommended protein intake for critically ill patients?",
            "reference_answer": "For critically ill patients, ASPEN recommends 1.2-2.0 g/kg/day of protein, with higher amounts (up to 2.5 g/kg/day) for patients with burns, trauma, or sepsis."
        },
        {
            "question": "How should parenteral nutrition be initiated?",
            "reference_answer": "Parenteral nutrition should be initiated gradually, starting with 50% of estimated needs on day 1, advancing to full nutrition by day 2-3, with careful monitoring of glucose, electrolytes, and fluid balance."
        }
    ]


def load_test_data(file_path: str) -> List[Dict[str, str]]:
    """Load test data from JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Test data file not found: {file_path}")
        print("Using sample test data instead...")
        return create_sample_test_data()


def save_results(summary: EvaluationSummary, output_file: str):
    """Save evaluation results to JSON file."""
    results_data = {
        "evaluation_summary": {
            "total_questions": summary.total_questions,
            "avg_bleu": summary.avg_bleu,
            "avg_rouge_1": summary.avg_rouge_1,
            "avg_rouge_2": summary.avg_rouge_2,
            "avg_rouge_l": summary.avg_rouge_l,
            "avg_response_time": summary.avg_response_time
        },
        "detailed_results": [
            {
                "question": r.question,
                "reference_answer": r.reference_answer,
                "generated_answer": r.generated_answer,
                "bleu_score": r.bleu_score,
                "rouge_1_f": r.rouge_1_f,
                "rouge_2_f": r.rouge_2_f,
                "rouge_l_f": r.rouge_l_f,
                "response_time": r.response_time,
                "num_sources": r.num_sources
            }
            for r in summary.results
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"Results saved to: {output_file}")


def print_summary(summary: EvaluationSummary):
    """Print evaluation summary."""
    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY - {summary.model_name}")
    print("=" * 60)
    print(f"Total Questions: {summary.total_questions}")
    print(f"Successful Evaluations: {len(summary.results)}")
    print()
    print("AVERAGE SCORES:")
    print(f"  BLEU Score:    {summary.avg_bleu:.4f}")
    print(f"  ROUGE-1 F1:    {summary.avg_rouge_1:.4f}")
    print(f"  ROUGE-2 F1:    {summary.avg_rouge_2:.4f}")
    print(f"  ROUGE-L F1:    {summary.avg_rouge_l:.4f}")
    print()
    print(f"Average Response Time: {summary.avg_response_time:.2f}s")
    print("=" * 60)


def evaluate_all_models(test_data: List[Dict[str, str]]) -> ModelComparisonResult:
    """Evaluate all available Ollama models."""
    print("🔍 Discovering available Ollama models...")
    models = get_available_ollama_models()
    
    if not models:
        raise Exception("No Ollama models found!")
    
    print(f"Found {len(models)} models: {[m['name'] for m in models]}")
    print("\n🚀 Starting comprehensive model evaluation...")
    
    model_summaries = []
    
    for i, model_info in enumerate(models, 1):
        model_name = model_info['name']
        print(f"\n[{i}/{len(models)}] Evaluating model: {model_name}")
        print("-" * 60)
        
        try:
            evaluator = RAGEvaluator(model_name=model_name)
            summary = evaluator.evaluate_dataset(test_data)
            model_summaries.append(summary)
            
            print(f"✅ {model_name} completed:")
            print(f"   BLEU: {summary.avg_bleu:.3f} | "
                  f"ROUGE-1: {summary.avg_rouge_1:.3f} | "
                  f"Avg Time: {summary.avg_response_time:.1f}s")
            
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")
            continue
    
    if not model_summaries:
        raise Exception("No models completed evaluation successfully!")
    
    # Create comparison DataFrame
    comparison_data = []
    for summary in model_summaries:
        comparison_data.append({
            'Model': summary.model_name,
            'BLEU': summary.avg_bleu,
            'ROUGE-1': summary.avg_rouge_1,
            'ROUGE-2': summary.avg_rouge_2,
            'ROUGE-L': summary.avg_rouge_l,
            'Response_Time': summary.avg_response_time,
            'Questions_Completed': len(summary.results)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Find best and worst models by BLEU score
    best_model = comparison_df.loc[comparison_df['BLEU'].idxmax(), 'Model']
    worst_model = comparison_df.loc[comparison_df['BLEU'].idxmin(), 'Model']
    
    return ModelComparisonResult(
        model_summaries=model_summaries,
        comparison_df=comparison_df,
        best_model=best_model,
        worst_model=worst_model
    )


def create_comparison_charts(comparison_result: ModelComparisonResult, output_dir: str = "evaluation_charts"):
    """Create comprehensive comparison charts."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    df = comparison_result.comparison_df
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Overall Performance Comparison (Bar Chart)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Ollama Models Performance Comparison', fontsize=16, fontweight='bold')
    
    # BLEU Scores
    axes[0, 0].bar(df['Model'], df['BLEU'], color='skyblue')
    axes[0, 0].set_title('BLEU Scores by Model')
    axes[0, 0].set_ylabel('BLEU Score')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # ROUGE-1 Scores
    axes[0, 1].bar(df['Model'], df['ROUGE-1'], color='lightcoral')
    axes[0, 1].set_title('ROUGE-1 F1 Scores by Model')
    axes[0, 1].set_ylabel('ROUGE-1 F1')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Response Time
    axes[1, 0].bar(df['Model'], df['Response_Time'], color='lightgreen')
    axes[1, 0].set_title('Average Response Time by Model')
    axes[1, 0].set_ylabel('Response Time (seconds)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Success Rate
    axes[1, 1].bar(df['Model'], df['Questions_Completed'], color='gold')
    axes[1, 1].set_title('Questions Completed by Model')
    axes[1, 1].set_ylabel('Number of Questions')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison_bars.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    correlation_cols = ['BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'Response_Time']
    correlation_matrix = df[correlation_cols].corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, fmt='.3f')
    plt.title('Metric Correlations Across Models', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/metric_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Radar Chart for Top 3 Models
    from math import pi
    
    # Select top 3 models by BLEU score
    top_models = df.nlargest(3, 'BLEU')
    
    metrics = ['BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    
    # Normalize metrics to 0-1 scale for radar chart
    normalized_data = top_models[metrics].copy()
    for metric in metrics:
        max_val = normalized_data[metric].max()
        if max_val > 0:
            normalized_data[metric] = normalized_data[metric] / max_val
    
    angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = ['red', 'blue', 'green']
    for i, (idx, model_data) in enumerate(normalized_data.iterrows()):
        values = model_data.tolist()
        values += values[:1]  # Complete the circle
        
        model_name = top_models.iloc[i]['Model']
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title('Top 3 Models - Performance Radar Chart', size=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/top_models_radar.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Performance vs Speed Scatter Plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df['Response_Time'], df['BLEU'], 
                         s=200, alpha=0.7, c=df['ROUGE-1'], 
                         cmap='viridis', edgecolors='black', linewidth=1)
    
    # Add model labels
    for i, model in enumerate(df['Model']):
        plt.annotate(model, (df['Response_Time'].iloc[i], df['BLEU'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.colorbar(scatter, label='ROUGE-1 Score')
    plt.xlabel('Response Time (seconds)')
    plt.ylabel('BLEU Score')
    plt.title('Model Performance vs Speed\n(Bubble color = ROUGE-1 score)', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_vs_speed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Charts saved to '{output_dir}/' directory:")
    print("  • model_comparison_bars.png - Overall performance comparison")
    print("  • metric_correlations.png - Correlation heatmap")
    print("  • top_models_radar.png - Top 3 models radar chart")
    print("  • performance_vs_speed.png - Performance vs Speed scatter plot")


def print_model_comparison_summary(comparison_result: ModelComparisonResult):
    """Print comprehensive model comparison summary."""
    df = comparison_result.comparison_df
    
    print("\n" + "🏆" * 60)
    print("COMPREHENSIVE MODEL COMPARISON RESULTS")
    print("🏆" * 60)
    
    print(f"\n🥇 BEST OVERALL MODEL: {comparison_result.best_model}")
    best_row = df[df['Model'] == comparison_result.best_model].iloc[0]
    print(f"   BLEU: {best_row['BLEU']:.4f} | ROUGE-1: {best_row['ROUGE-1']:.4f} | Time: {best_row['Response_Time']:.1f}s")
    
    print(f"\n🥉 NEEDS IMPROVEMENT: {comparison_result.worst_model}")
    worst_row = df[df['Model'] == comparison_result.worst_model].iloc[0]
    print(f"   BLEU: {worst_row['BLEU']:.4f} | ROUGE-1: {worst_row['ROUGE-1']:.4f} | Time: {worst_row['Response_Time']:.1f}s")
    
    print(f"\n⚡ FASTEST MODEL: {df.loc[df['Response_Time'].idxmin(), 'Model']}")
    print(f"   Response Time: {df['Response_Time'].min():.1f}s")
    
    print(f"\n🎯 HIGHEST ROUGE-1: {df.loc[df['ROUGE-1'].idxmax(), 'Model']}")
    print(f"   ROUGE-1 Score: {df['ROUGE-1'].max():.4f}")
    
    print("\n📋 FULL RANKINGS (by BLEU score):")
    ranked_df = df.sort_values('BLEU', ascending=False)
    for i, (_, row) in enumerate(ranked_df.iterrows(), 1):
        print(f"  {i}. {row['Model']:<20} BLEU: {row['BLEU']:.4f} | Time: {row['Response_Time']:.1f}s")
    
    print("\n" + "🏆" * 60)


def main():
    """Main evaluation function with comprehensive model comparison."""
    print("🚀 COMPREHENSIVE RAG EVALUATION - ALL OLLAMA MODELS")
    print("=" * 70)
    
    # Check if ChromaDB is ready
    from test_rag import check_chromadb_ready
    if not check_chromadb_ready():
        print("ChromaDB not ready. Please run the setup first.")
        return
    
    # Load test data
    test_data = load_test_data("evaluation_data.json")
    print(f"📝 Loaded {len(test_data)} test questions")
    
    # Ask user for evaluation type
    print("\nChoose evaluation mode:")
    print("1. Single model evaluation (choose specific model)")
    print("2. ALL MODELS comparison (evaluate all available models)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        # Single model evaluation (original behavior)
        models = get_available_ollama_models()
        if not models:
            print("No Ollama models available.")
            return
        
        selected_model = select_ollama_model(models)
        if not selected_model:
            return
        
        print(f"\n🔄 Evaluating single model: {selected_model}")
        evaluator = RAGEvaluator(model_name=selected_model)
        summary = evaluator.evaluate_dataset(test_data)
        
        print_summary(summary)
        
        # Save single model results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"evaluation_results_{selected_model.replace(':', '_')}_{timestamp}.json"
        save_results(summary, output_file)
        
        # Print best and worst performing questions
        if summary.results:
            print("\nBEST PERFORMING QUESTIONS (by BLEU):")
            best_results = sorted(summary.results, key=lambda x: x.bleu_score, reverse=True)[:3]
            for i, result in enumerate(best_results, 1):
                print(f"  {i}. BLEU: {result.bleu_score:.3f} - {result.question[:60]}...")
            
            print("\nWORST PERFORMING QUESTIONS (by BLEU):")
            worst_results = sorted(summary.results, key=lambda x: x.bleu_score)[:3]
            for i, result in enumerate(worst_results, 1):
                print(f"  {i}. BLEU: {result.bleu_score:.3f} - {result.question[:60]}...")
    
    elif choice == "2":
        # Comprehensive model comparison
        print(f"\n🎯 COMPREHENSIVE MODEL COMPARISON MODE")
        print("This will evaluate ALL available Ollama models!")
        
        confirm = input("This may take a while. Continue? (y/N): ").strip().lower()
        if confirm != 'y':
            print("Evaluation cancelled.")
            return
        
        try:
            # Run comprehensive evaluation
            comparison_result = evaluate_all_models(test_data)
            
            # Print comprehensive summary
            print_model_comparison_summary(comparison_result)
            
            # Create and save charts
            print(f"\n📊 Creating comparison charts...")
            create_comparison_charts(comparison_result)
            
            # Save comparison data
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            comparison_file = f"model_comparison_{timestamp}.csv"
            comparison_result.comparison_df.to_csv(comparison_file, index=False)
            print(f"📄 Comparison data saved to: {comparison_file}")
            
            # Save detailed results for each model
            detailed_results = {
                "comparison_summary": {
                    "best_model": comparison_result.best_model,
                    "worst_model": comparison_result.worst_model,
                    "total_models_tested": len(comparison_result.model_summaries),
                    "timestamp": timestamp
                },
                "model_results": []
            }
            
            for summary in comparison_result.model_summaries:
                model_data = {
                    "model_name": summary.model_name,
                    "avg_bleu": summary.avg_bleu,
                    "avg_rouge_1": summary.avg_rouge_1,
                    "avg_rouge_2": summary.avg_rouge_2,
                    "avg_rouge_l": summary.avg_rouge_l,
                    "avg_response_time": summary.avg_response_time,
                    "questions_completed": len(summary.results),
                    "individual_results": [
                        {
                            "question": r.question,
                            "bleu_score": r.bleu_score,
                            "rouge_1_f": r.rouge_1_f,
                            "response_time": r.response_time
                        }
                        for r in summary.results
                    ]
                }
                detailed_results["model_results"].append(model_data)
            
            detailed_file = f"detailed_model_comparison_{timestamp}.json"
            with open(detailed_file, 'w') as f:
                json.dump(detailed_results, f, indent=2)
            print(f"📄 Detailed results saved to: {detailed_file}")
            
            print(f"\n✅ COMPREHENSIVE EVALUATION COMPLETE!")
            print(f"🏆 Winner: {comparison_result.best_model}")
            print(f"📊 Charts available in: evaluation_charts/")
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return
    
    else:
        print("Invalid choice. Please run again and select 1 or 2.")
        return


if __name__ == "__main__":
    main()
