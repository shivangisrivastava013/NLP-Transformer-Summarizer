import time
from typing import Optional

from nlp_engine.evaluation import SummarizationEvaluator
from nlp_engine.schemas import PipelineResult
from nlp_engine.sentiment import SentimentAnalyzer
from nlp_engine.summarizer import AbstractiveSummarizer


class NLPPipeline:
    """
    Unified end-to-end NLP pipeline for long document summarization, sentiment analysis, and evaluation.
    """

    def __init__(
        self,
        summarizer_model: str = "facebook/bart-large-cnn",
        sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english",
        device: int = -1,
    ):
        self.summarizer = AbstractiveSummarizer(model_name=summarizer_model, device=device)
        self.sentiment_analyzer = SentimentAnalyzer(model_name=sentiment_model)
        self.evaluator = SummarizationEvaluator()

    def process(
        self,
        text: str,
        reference_summary: Optional[str] = None,
        max_length: int = 130,
        min_length: int = 30,
        max_chunk_tokens: int = 512,
        overlap_tokens: int = 64,
    ) -> PipelineResult:
        t0 = time.time()
        summary_res = self.summarizer.summarize(
            text,
            max_length=max_length,
            min_length=min_length,
            max_chunk_tokens=max_chunk_tokens,
            overlap_tokens=overlap_tokens,
        )
        t1 = time.time()

        sentiment_res = self.sentiment_analyzer.analyze(summary_res.summary_text or text)

        eval_res = None
        if reference_summary and reference_summary.strip():
            eval_res = self.evaluator.evaluate(
                candidate_summary=summary_res.summary_text,
                reference_summary=reference_summary,
                source_text=text,
                latency_seconds=t1 - t0,
            )

        return PipelineResult(
            input_text=text,
            summary=summary_res,
            sentiment=sentiment_res,
            evaluation=eval_res,
        )
