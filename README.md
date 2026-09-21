# NLP Transformer Summarizer & Sentiment Workbench

[![NLP CI Pipeline](https://github.com/shivangisrivastava013/NLP-Transformer-Summarizer/actions/workflows/ci.yml/badge.svg)](https://github.com/shivangisrivastava013/NLP-Transformer-Summarizer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-green.svg)](https://www.python.org/)

An NLP workbench for summarizing long documents and inspecting sentiment. It supports BART and Flan-T5, token-aware chunking, transparent heuristic fallbacks, and reproducible evaluation.

## Demo and project links

- [Portfolio project page](https://shivangisrivastava013.github.io/shivangi-portfolio/#projects)
- [Streamlit application source](app.py)
- [Command-line demo](demo.py)
- [Evaluation results](results/evaluation_results.json)

---

## Architecture and capabilities

1. **Token-Aware Sliding Window Chunker (`TokenAwareChunker`)**:
   - Resolves context window truncation in transformer models by splitting long-form documents into token-budgeted chunks with configurable overlap (e.g. 512 max tokens, 64 overlap tokens).
   - Preserves sentence boundaries and entity continuity across window transitions.

2. **Hierarchical Abstractive Summarizer (`AbstractiveSummarizer`)**:
   - Two-stage summarization process: generates intermediate chunk-level summaries followed by a second-stage meta-summary aggregation.
   - Supports pre-trained models including `facebook/bart-large-cnn` and `google/flan-t5-base`.
   - Includes fallback error tracing (`fallback_reason`) and high-quality heuristic sentence extraction for offline or CPU execution.

3. **Threshold-Based Neutral Sentiment Classifier (`SentimentAnalyzer`)**:
   - Classifies text into `POSITIVE`, `NEGATIVE`, and `NEUTRAL` by combining DistilBERT SST-2 binary predictions with confidence thresholding (< 0.58 margin maps to `NEUTRAL`).
   - Clearly distinguishes statistical probabilities from heuristic keyword scores (`is_heuristic=True`).

4. **Quantitative Evaluation Pipeline (`SummarizationEvaluator`)**:
   - Calculates **ROUGE-1**, **ROUGE-2**, **ROUGE-L** (Precision, Recall, F1), **BERTScore / Fallback Similarity**, compression ratio, and processing latency.

5. **Interactive Streamlit Workbench (`app.py`)**:
   - Web application providing real-time document summarization, dynamic chunk size/overlap slider passthrough, token chunk inspection, reference evaluation, and transparent fallback error warnings.

---

## Evaluation results

The evaluation pipeline benchmarked models across sample long-form technical articles (results saved in `results/evaluation_results.json` and `results/model_comparison.csv`):

| Model Architecture | Execution Mode | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | BERTScore Mode | Compression Ratio | Average Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BART (`facebook/bart-large-cnn`)** | Heuristic Fallback | 0.2427 | 0.0563 | 0.1805 | Fallback Similarity | 46.75% | 0.0013s |
| **Flan-T5 (`google/flan-t5-base`)** | Heuristic Fallback | 0.2427 | 0.0563 | 0.1805 | Fallback Similarity | 46.75% | 0.0012s |
| **Heuristic Rule-Based Baseline** | Rule Baseline | 0.2427 | 0.0563 | 0.1805 | Fallback Similarity | 46.75% | 0.0009s |

*Note: Execution mode automatically detects whether Hugging Face transformer weights were loaded or if CPU fallback mode was triggered, guaranteeing transparent reporting.*

---

## Repository Structure

```text
.
├── nlp_engine/
│   ├── __init__.py           # Package exports
│   ├── chunking.py           # Token-aware sliding window chunker
│   ├── summarizer.py         # Hierarchical abstractive summarization engine
│   ├── sentiment.py          # Threshold-based neutral sentiment analyzer
│   ├── evaluation.py         # ROUGE-1/2/L & BERTScore metric calculator
│   ├── schemas.py            # Dataclasses and type definitions
│   └── pipeline.py           # Unified NLP pipeline entrypoint
├── scripts/
│   └── evaluate_summarization.py # Benchmark runner generating JSON & CSV metrics
├── tests/
│   ├── test_chunking.py      # Chunker unit tests
│   ├── test_summarizer.py    # Summarizer unit tests
│   ├── test_sentiment.py     # Sentiment analyzer unit tests
│   ├── test_evaluation.py    # Metrics unit tests
│   └── test_pipeline.py     # End-to-end integration tests
├── results/                  # Committed quantitative metrics & metadata
│   ├── evaluation_results.json
│   ├── model_comparison.csv
│   └── environment_metadata.json
├── app.py                    # Streamlit web application
├── demo.py                   # Interactive CLI demo
├── Dockerfile                # Production Docker deployment container
├── pyproject.toml            # Project dependencies and ruff/black/pytest config
└── .github/
    └── workflows/
        └── ci.yml            # GitHub Actions CI workflow
```

---

## Getting started

### 1. Installation

```bash
git clone https://github.com/shivangisrivastava013/NLP-Transformer-Summarizer.git
cd NLP-Transformer-Summarizer
pip install -e .
```

### 2. Run Interactive CLI Demo

```bash
python demo.py
```

### 3. Launch Streamlit Workbench UI

```bash
streamlit run app.py
```

### 4. Run Quantitative Benchmark Evaluation

```bash
python scripts/evaluate_summarization.py
```

### 5. Run Linting & PyTest Test Suite

```bash
python -m ruff check .
python -m black --check .
python -m pytest tests/ -v
```

---

## Docker deployment

Build and run using Docker:

```bash
docker build -t nlp-summarizer:latest .
docker run -p 8501:8501 nlp-summarizer:latest
```

Navigate to `http://localhost:8501` to access the Streamlit workbench interface.

---

## License

This repository is available under the [MIT License](LICENSE).
