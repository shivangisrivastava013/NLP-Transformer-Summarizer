from nlp_engine.summarizer import AbstractiveSummarizer
from nlp_engine.sentiment import SentimentAnalyzer


def main():
    print("[+] Initializing NLP Transformer Summarization & Sentiment Pipeline...")

    sample_article = (
        "Artificial Intelligence and Deep Learning have made phenomenal progress in recent years. "
        "Retrieval-Augmented Generation (RAG) and Transformer architectures like BART, T5, and RoBERTa "
        "have revolutionized natural language understanding. Researchers at major AI labs are deploying "
        "scalable foundation models capable of abstractive text summarization, sentiment analysis, and multi-modal "
        "reasoning with high efficiency and accuracy."
    )

    print("\n[*] INPUT TEXT ARTICLE:")
    print(f"   \"{sample_article}\"\n")

    summarizer = AbstractiveSummarizer()
    sentiment = SentimentAnalyzer()

    summary_result = summarizer.summarize(sample_article)
    sentiment_result = sentiment.analyze(sample_article)

    print("=" * 70)
    print("[*] NLP TRANSFORMER INFERENCE RESULTS")
    print("=" * 70)
    print(f"[*] Abstractive Summary:  \"{summary_result}\"")
    print(f"[+] Sentiment Label:     {sentiment_result['label']} (Confidence: {sentiment_result['score']*100:.1f}%)")
    print("=" * 70)


if __name__ == '__main__':
    main()
