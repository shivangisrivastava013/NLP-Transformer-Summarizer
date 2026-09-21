from nlp_engine.pipeline import NLPPipeline


def test_nlp_pipeline_end_to_end():
    pipeline = NLPPipeline(
        summarizer_model="rule-based-fallback",
        sentiment_model="rule-based-fallback",
    )

    text = (
        "Quantum computing systems are advancing rapidly with innovative fault-tolerant topological architectures. "
        "Laboratory benchmarks demonstrated impressive two-qubit gate fidelity surpassing safety thresholds. "
        "Commercial deployment of quantum algorithms for drug discovery is expected to accelerate significantly."
    )
    ref = "Quantum computing systems achieved high gate fidelity accelerating commercial deployment for drug discovery."

    result = pipeline.process(text=text, reference_summary=ref)

    assert result.summary.summary_text != ""
    assert result.sentiment.label in ["POSITIVE", "NEGATIVE", "NEUTRAL"]
    assert result.evaluation is not None
    assert result.evaluation.rouge1_f1 > 0.0
