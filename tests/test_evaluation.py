import pytest
from nlp_engine.evaluation import SummarizationEvaluator


def test_evaluation_metrics_exact_match():
    evaluator = SummarizationEvaluator()
    cand = "Researchers developed a novel quantum computing architecture with high gate fidelity."
    ref = "Researchers developed a novel quantum computing architecture with high gate fidelity."
    src = "Full article text describing quantum computing advances."

    metrics = evaluator.evaluate(candidate_summary=cand, reference_summary=ref, source_text=src, latency_seconds=0.1)

    assert metrics.rouge1_f1 == 1.0
    assert metrics.rouge2_f1 == 1.0
    assert metrics.rougel_f1 == 1.0
    assert metrics.bertscore_f1 is not None and metrics.bertscore_f1 >= 0.90


def test_evaluation_metrics_partial_match():
    evaluator = SummarizationEvaluator()
    cand = "Quantum processor achieved high gate fidelity in laboratory testing."
    ref = "Researchers developed a novel 128-qubit quantum processor with high fidelity."
    src = "Full article text describing quantum computing advances."

    metrics = evaluator.evaluate(candidate_summary=cand, reference_summary=ref, source_text=src, latency_seconds=0.2)

    assert 0.0 < metrics.rouge1_f1 < 1.0
    assert metrics.compression_ratio is not None
