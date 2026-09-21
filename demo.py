import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from nlp_engine.pipeline import NLPPipeline

SAMPLE_TEXT = """
Artificial Intelligence and machine learning engineering have transitioned from isolated experimental research
into foundational enterprise software infrastructure. Modern industrial applications demand robust, token-aware
architectures capable of processing long-form documents without context truncation or silent memory overflow.
By integrating sliding window chunking with pre-trained transformer models such as BART and Flan-T5, system
architects achieve high compression ratios while preserving critical factual entities and semantic nuance.
Furthermore, combining abstractive text summarization with calibrated three-class sentiment analysis and
reproducible ROUGE evaluation pipelines provides verifiable benchmarks for deploying production-grade
Generative AI solutions across finance, healthcare, and software engineering domains.
"""


def main():
    print("==================================================================")
    print("NLP Transformer Summarizer & Sentiment Workbench - Interactive Demo")
    print("==================================================================\n")

    pipeline = NLPPipeline(
        summarizer_model="facebook/bart-large-cnn",
        sentiment_model="distilbert-base-uncased-finetuned-sst-2-english",
    )

    print("Input Document:")
    print(SAMPLE_TEXT.strip())
    print("\nProcessing document through token chunker, summarizer & sentiment engine...\n")

    result = pipeline.process(
        text=SAMPLE_TEXT,
        reference_summary="Sliding window chunking with BART and Flan-T5 enables high-compression, entity-preserving long document summarization with calibrated sentiment and ROUGE evaluation.",
        max_length=100,
        min_length=25,
    )

    print(f"Summary ({'Fallback' if result.summary.is_fallback else 'Transformer Model'}):")
    print(result.summary.summary_text)

    print(f"\nSentiment Label: {result.sentiment.label} (Score: {result.sentiment.score})")
    print(
        f"Input Tokens: {result.summary.input_token_count} | Summary Tokens: {result.summary.summary_token_count} | Compression: {result.summary.compression_ratio}%"
    )

    if result.evaluation:
        print("\nQuantitative Evaluation Metrics:")
        print(f"  • ROUGE-1 F1: {result.evaluation.rouge1_f1}")
        print(f"  • ROUGE-2 F1: {result.evaluation.rouge2_f1}")
        print(f"  • ROUGE-L F1: {result.evaluation.rougel_f1}")
        print(f"  • BERTScore F1: {result.evaluation.bertscore_f1}")
        print(f"  • Latency: {result.evaluation.latency_seconds}s")

    print("\n==================================================================")


if __name__ == "__main__":
    main()
