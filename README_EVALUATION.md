# RAG System Evaluation with BLEU & Model Comparison

This evaluation system uses BLEU, ROUGE, and other metrics to assess the quality of your RAG-generated answers. **NEW**: Now includes comprehensive comparison of ALL Ollama models with beautiful charts!

## Setup

1. **Install evaluation dependencies:**
   ```bash
   pip install sacrebleu rouge-score nltk matplotlib seaborn pandas
   ```

2. **Update requirements (if needed):**
   ```bash
   pip install -r requirements.txt
   ```

## Usage Options

### 🚀 NEW: Comprehensive Model Comparison (Recommended!)
**Evaluate ALL your Ollama models at once with charts:**
```bash
# Interactive mode - choose single model or all models
python3 evaluate_rag.py

# Automated all-models comparison
python3 compare_models.py
```

### Single Model Evaluation
```bash
python3 evaluate_rag.py
# Choose option 1 for single model
```

### Custom Test Data
Create your own test data file `evaluation_data.json`:
```json
[
  {
    "question": "Your medical question here",
    "reference_answer": "The gold standard answer from ASPEN guidelines"
  }
]
```

## Metrics Explained

- **BLEU Score (0-1)**: Measures similarity between generated and reference answers
  - 0.0-0.3: Low quality
  - 0.3-0.6: Moderate quality  
  - 0.6-1.0: High quality

- **ROUGE-1/2/L**: Measures overlap of unigrams, bigrams, and longest common subsequence
- **Response Time**: How fast your RAG system generates answers

## Interpreting Results

The evaluation provides:
- Average scores across all test questions
- Individual question performance
- Best/worst performing questions
- Detailed results saved to timestamped JSON files

## Improving Scores

To improve BLEU scores:
1. **Increase token limits** (already done - now 2000 tokens)
2. **Fine-tune prompts** to be more specific
3. **Improve search relevance** by adjusting embedding models
4. **Add more context** by increasing search_limit in rag_answer()

## 📊 What You Get

### Charts Generated (in `evaluation_charts/` folder):
1. **model_comparison_bars.png** - Bar charts comparing BLEU, ROUGE-1, response time, success rate
2. **metric_correlations.png** - Heatmap showing which metrics correlate
3. **top_models_radar.png** - Radar chart of top 3 performing models
4. **performance_vs_speed.png** - Scatter plot of performance vs speed

### Data Files:
- **model_comparison_TIMESTAMP.csv** - Easy-to-analyze spreadsheet
- **detailed_model_comparison_TIMESTAMP.json** - Complete results with individual question scores

## Example Output
```
🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆
COMPREHENSIVE MODEL COMPARISON RESULTS
🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆

🥇 BEST OVERALL MODEL: mistral:7b
   BLEU: 0.4523 | ROUGE-1: 0.6234 | Time: 3.4s

⚡ FASTEST MODEL: phi3:3.8b
   Response Time: 1.2s

🎯 HIGHEST ROUGE-1: llama3.1:8b
   ROUGE-1 Score: 0.6456

📋 FULL RANKINGS (by BLEU score):
  1. mistral:7b          BLEU: 0.4523 | Time: 3.4s
  2. llama3.1:8b         BLEU: 0.4234 | Time: 4.1s
  3. phi3:3.8b           BLEU: 0.3876 | Time: 1.2s
```

## 🎯 Model Selection Guide

**Use this evaluation to pick the best model for your needs:**

- **Best Overall Quality**: Highest BLEU + ROUGE scores
- **Fastest Response**: Lowest response time (good for real-time apps)
- **Balanced**: Good BLEU score with reasonable speed
- **Most Reliable**: Highest success rate (completed most questions)

## Tips for Better Evaluation

1. **Run full comparison first** to see all your options
2. **Add more test questions** for medical specialties you care about
3. **Consider your use case**: Real-time vs batch processing
4. **Re-run after model updates** to track improvements
