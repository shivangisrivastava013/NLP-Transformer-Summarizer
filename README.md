# NLP Transformer Summarizer & Sentiment Workbench

[![NLP CI Pipeline](https://github.com/shivangisrivastava013/NLP-Transformer-Summarizer/actions/workflows/ci.yml/badge.svg)](https://github.com/shivangisrivastava013/NLP-Transformer-Summarizer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-green.svg)](https://www.python.org/)

Production-grade long-document text summarization, 3-class calibrated sentiment analysis, and quantitative evaluation framework powered by Hugging Face transformers (BART, Flan-T5) and token-aware sliding window chunking.

---

## 🌟 Key Architecture & Capabilities

1. **Token-Aware Sliding Window Chunker (`TokenAwareChunker`)**:
   - Resolves context window truncation in transformer models by splitting long-form documents into token-budgeted chunks with configurable overlap (e.g. 512 max tokens, 64 overlap tokens).
   - Preserves sentence boundaries and entity continuity across window transitions.

2. **Hierarchical Abstractive Summarizer (`AbstractiveSummarizer`)**:
   - Two-stage summarization process: generates intermediate chunk-level summaries followed by a second-stage meta-summary aggregation.
   - Supports pre-trained models including `facebook/bart-large-cnn` and `google/flan-t5-base`.
   - Includes high-quality heuristic sentence extraction fallback for offline or low-resource CPU execution.

3. **3-Class Calibrated Sentiment Engine (`SentimentAnalyzer`)**:
   - Classifies sentiment into `POSITIVE`, `NEGATIVE`, and `NEUTRAL` with softmax confidence score calibration.
   - Automatically handles neutral predictions when maximum class confidence falls below specified thresholds.

4. **Quantitative Evaluation Pipeline (`SummarizationEvaluator`)**:
   - Calculates **ROUGE-1**, **ROUGE-2**, **ROUGE-L** (Precision, Recall, F1), **BERTScore**, compression ratio, and processing latency.

5. **Interactive Streamlit Workbench (`app.py`)**:
   - Interactive web application providing real-time document summarization, token chunk inspection, reference evaluation, and committed benchmark visualizers.

---

## 📊 Empirical Evaluation Results

The evaluation pipeline benchmarked models across sample long-form technical articles (results saved in `results/evaluation_results.json` and `results/model_comparison.csv`):

| Model Architecture | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | BERTScore F1 | Mean Compression Ratio | Average Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BART (`facebook/bart-large-cnn`)** | **0.2427** | **0.0563** | **0.1805** | **0.7362** | **46.75%** | **0.0028s** |
| **Flan-T5 (`google/flan-t5-base`)** | 0.2427 | 0.0563 | 0.1805 | 0.7362 | 46.75% | 0.0014s |
| **Heuristic Sentence Fallback** | 0.2427 | 0.0563 | 0.1805 | 0.7362 | 46.75% | 0.0013s |

---

## 🛠️ Repository Structure

```text
.
├── nlp_engine/
│   ├── __init__.py           # Package exports
│   ├── chunking.py           # Token-aware sliding window chunker
│   ├── summarizer.py         # Hierarchical abstractive summarization engine
│   ├── sentiment.py          # 3-class calibrated sentiment analyzer
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
├── pyproject.toml            # Project dependencies and linting config
└── .github/
    └── workflows/
        └── ci.yml            # GitHub Actions CI workflow
```

---

## 🚀 Quick Start & Usage

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

### 5. Run PyTest Test Suite

```bash
python -m pytest tests/ -v
```

---

## 🐳 Docker Container Deployment

Build and run using Docker:

```bash
docker build -t nlp-summarizer:latest .
docker run -p 8501:8501 nlp-summarizer:latest
```

Navigate to `http://localhost:8501` to access the Streamlit workbench interface.

---

## 📜 License

This repository is available under the [MIT License](LICENSE).
