# NLP Transformer Text Summarization & Sentiment Analysis Engine

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Transformers-HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

Production NLP pipeline performing **abstractive text summarization** and **fine-grained sentiment analysis** on long-form articles, research documents, and customer feedback.

---

## 🌟 Key Capabilities
- 📝 **Abstractive Text Summarization:** Leverages sequence-to-sequence transformer models (BART / T5) to condense long documents into coherent key insight summaries.
- 🎭 **Fine-Grained Sentiment Analysis:** RoBERTa / DistilBERT classification head for sentiment polarity scoring and confidence rating.
- 📊 **Metric Evaluation:** Automated calculation of ROUGE-1, ROUGE-2, ROUGE-L, and BLEU scores.

---

## 📁 Repository Structure
```text
NLP-Transformer-Summarizer/
├── nlp_engine/             # Core NLP Transformer Modules
│   ├── __init__.py
│   ├── summarizer.py       # BART / T5 Abstractive Summarizer
│   └── sentiment.py        # RoBERTa / DistilBERT Sentiment Head
├── demo.py                 # Executable Demonstration Script
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚡ Quick Start
```bash
git clone https://github.com/shivangisrivastava013/NLP-Transformer-Summarizer.git
cd NLP-Transformer-Summarizer

pip install -r requirements.txt
python demo.py
```

---

## 👤 Author
**Shivangi Srivastava**  
MS in Artificial Intelligence @ NJIT  
[LinkedIn Profile](https://www.linkedin.com/in/shivangisrivastava013/) | [Portfolio](https://shivangisrivastava013.github.io/shivangi-portfolio/)
