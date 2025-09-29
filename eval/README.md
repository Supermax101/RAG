# TPN RAG System Evaluation

This directory contains evaluation tools for the TPN (Total Parenteral Nutrition) RAG system using [RAGAS](https://github.com/explodinggradients/ragas) and clinical MCQ questions.

## Files

- `tpn_eval_questions.csv` - MCQ evaluation dataset from clinical TPN questions
- `tpn_rag_evaluation.py` - Main evaluation script using RAGAS
- `README.md` - This file

## Evaluation Process

The evaluation system:

1. **Loads MCQ Questions**: Filters 48 MCQ questions from 101 total questions (`Answer Type = "mcq_single"`)
2. **Forces Option-Only Responses**: Uses specialized prompts to get only A, B, C, D answers
3. **Tests TPN Clinical Knowledge**: Evaluates against ASPEN TPN guidelines in the RAG system
4. **Measures Accuracy**: Compares model answers against correct options
5. **Uses RAGAS Metrics**: Provides comprehensive evaluation using RAGAS framework
6. **Generates Reports**: Creates JSON and CSV reports with detailed results

## Usage

### Quick Start

```bash
# Make sure you're in the project root and have the TPN RAG system loaded
cd /path/to/your/project

# Install additional dependencies
uv pip install ragas pandas

# Run evaluation (limited to 10 questions for testing)
uv run python eval/tpn_rag_evaluation.py
```

### Full Evaluation

```bash
# Run full evaluation on all 48 MCQ questions
uv run python eval/tpn_rag_evaluation.py
# When prompted, enter 'all' for no question limit
```

### Model Testing

```bash
# Test different models
uv run python eval/tpn_rag_evaluation.py
# When prompted, enter model name like: llama3:8b, mistral:7b, etc.
```

## Output Files

The evaluation generates several output files:

- `tpn_evaluation_results_[model]_[timestamp].json` - Complete evaluation results
- `tpn_evaluation_summary_[timestamp].csv` - Detailed results in CSV format

## Evaluation Metrics

### Primary Metrics
- **Accuracy**: Percentage of correct MCQ answers
- **Response Time**: Average time to generate answers
- **Source Utilization**: Number of TPN documents used per question

### Error Analysis
- **System Errors**: Technical failures (network, parsing, etc.)
- **Wrong Choices**: Incorrect clinical decisions
- **Answer Distribution**: Pattern analysis of chosen options

### RAGAS Integration
While RAGAS is optimized for open-ended QA, we use it for:
- Response quality assessment
- Source relevance evaluation
- Consistency metrics

## Clinical Evaluation Interpretation

### Accuracy Benchmarks
- **>90%**: Excellent clinical knowledge, suitable for clinical support
- **80-90%**: Good performance, may need prompt refinement
- **70-80%**: Moderate performance, requires improvement
- **<70%**: Poor performance, significant issues need addressing

### Common Issues
- **Low accuracy**: May indicate insufficient TPN document coverage or poor prompt engineering
- **High system errors**: Technical issues with RAG pipeline or model connectivity
- **Inconsistent answers**: May suggest unstable model responses or ambiguous prompts

## Prerequisites

1. **TPN RAG System Loaded**: Run `uv run python main.py init` first
2. **Ollama Running**: Ensure Ollama service is active with desired models
3. **Dependencies**: RAGAS and pandas installed

## Customization

### Adding New Questions
- Add questions to `tpn_eval_questions.csv` with format:
  - `Answer Type`: "mcq_single"  
  - `Options`: "A. Option 1 | B. Option 2 | C. Option 3 | D. Option 4"
  - `Corrrect Option (s)`: Single letter (A, B, C, D)

### Modifying Prompts
- Edit the `create_mcq_prompt()` method in `tpn_rag_evaluation.py`
- Test prompt changes with small question sets first

### Custom RAGAS Metrics
- Add custom RAGAS metrics in the `generate_ragas_evaluation()` method
- Refer to [RAGAS documentation](https://docs.ragas.io/) for available metrics

## Example Output

```
🏥 TPN RAG System Clinical Evaluation
📚 Using RAGAS for comprehensive evaluation metrics
============================================================
🤖 Default model: mistral:7b
Enter different model name (or press Enter for default): 
Limit questions for testing? (default: 20, 'all' for no limit): 10

🔧 Initializing TPN RAG system with model: mistral:7b
✅ RAG system ready: 16998 chunks from 81 documents
📄 Loading evaluation questions from eval/tpn_eval_questions.csv
✅ Loaded 48 MCQ questions (out of 101 total questions)
📊 Question breakdown:
   • mcq_single: 48 questions
   • q_and_a: 53 questions
📊 Limiting evaluation to first 10 questions
🧪 Starting evaluation of 10 MCQ questions...
------------------------------------------------------------
🔍 Evaluating Question 1: In which of the following cases should PN be initiated...
   ✅ Expected: F, Got: F
🔍 Evaluating Question 2: Which intervention would be the LEAST likely next access...
   ❌ Expected: C, Got: A
📊 Progress: 5/10 questions, Accuracy: 80.0%
📊 Progress: 10/10 questions, Accuracy: 85.0%

🎊 EVALUATION COMPLETE! 🎊
============================================================
📊 **TPN RAG System Evaluation Results**
🤖 Model Used: mistral:7b
📝 Total Questions: 10
✅ Correct Answers: 8
🎯 **Accuracy: 85.00%**
⏱️  Average Response Time: 1250.5ms

📈 **Error Analysis:**
   • System Errors: 0
   • Wrong Choices: 2
   • Error Rate: 20.00%

📊 **Answer Distribution:**
   • A: 2 (20.0%)
   • B: 1 (10.0%)
   • C: 3 (30.0%)
   • D: 2 (20.0%)
   • F: 2 (20.0%)
```
