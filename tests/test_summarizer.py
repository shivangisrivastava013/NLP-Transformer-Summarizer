import pytest
from nlp_engine.summarizer import AbstractiveSummarizer


def test_summarizer_empty_text():
    summarizer = AbstractiveSummarizer(model_name="rule-based-fallback")
    res = summarizer.summarize("")
    assert res.summary_text == ""
    assert res.input_token_count == 0


def test_summarizer_heuristic_fallback():
    summarizer = AbstractiveSummarizer(model_name="rule-based-fallback")
    text = (
        "Researchers at NJIT introduced a novel framework for long document text summarization. "
        "The system incorporates sliding window token chunking with transformer architectures. "
        "Empirical evaluations demonstrated high ROUGE-1 scores across multiple benchmark datasets. "
        "In addition, the pipeline handles multi-modal inputs and structured data extractions. "
        "The proposed approach reduces memory footprint while maintaining high factual accuracy."
    )
    res = summarizer.summarize(text, max_length=40, min_length=10)
    assert len(res.summary_text) > 0
    assert res.compression_ratio >= 0.0
    assert res.input_token_count >= res.summary_token_count
