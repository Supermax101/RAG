#!/usr/bin/env python3
"""
Quick Model Comparison Script

This is a simplified script that runs the comprehensive model evaluation
without requiring user interaction. Perfect for automated testing.
"""

import sys
from evaluate_rag import evaluate_all_models, load_test_data, create_comparison_charts, print_model_comparison_summary
from test_rag import check_chromadb_ready
import time
import json


def main():
    """Run comprehensive model comparison automatically."""
    print("🚀 AUTOMATED MODEL COMPARISON")
    print("=" * 50)
    
    # Check ChromaDB
    if not check_chromadb_ready():
        print("❌ ChromaDB not ready. Please run setup first.")
        sys.exit(1)
    
    # Load test data
    test_data = load_test_data("evaluation_data.json")
    print(f"📝 Loaded {len(test_data)} test questions")
    
    try:
        # Run evaluation on all models
        print(f"\n🎯 Starting comprehensive evaluation...")
        comparison_result = evaluate_all_models(test_data)
        
        # Print results
        print_model_comparison_summary(comparison_result)
        
        # Create charts
        print(f"\n📊 Creating comparison charts...")
        create_comparison_charts(comparison_result)
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # CSV export
        csv_file = f"model_comparison_{timestamp}.csv"
        comparison_result.comparison_df.to_csv(csv_file, index=False)
        print(f"📄 Results saved to: {csv_file}")
        
        # JSON export
        summary_data = {
            "best_model": comparison_result.best_model,
            "worst_model": comparison_result.worst_model,
            "models_tested": len(comparison_result.model_summaries),
            "timestamp": timestamp,
            "results": comparison_result.comparison_df.to_dict('records')
        }
        
        json_file = f"model_comparison_summary_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"\n✅ EVALUATION COMPLETE!")
        print(f"🏆 Best Model: {comparison_result.best_model}")
        best_row = comparison_result.comparison_df[
            comparison_result.comparison_df['Model'] == comparison_result.best_model
        ].iloc[0]
        print(f"   BLEU: {best_row['BLEU']:.4f}")
        print(f"   ROUGE-1: {best_row['ROUGE-1']:.4f}")
        print(f"   Avg Time: {best_row['Response_Time']:.1f}s")
        
        print(f"\n📊 Charts saved to: evaluation_charts/")
        print(f"📄 Data files: {csv_file}, {json_file}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
