import logging
from typing import Optional

from nlp_engine.schemas import SentimentResult

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Threshold-Based Neutral Sentiment Classifier combining DistilBERT SST-2 binary classification
    with confidence score thresholding for neutral sentiment detection.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        neutral_threshold: float = 0.58,
    ):
        self.model_name = model_name
        self.neutral_threshold = neutral_threshold
        self.pipeline = None
        self.fallback_reason: Optional[str] = None
        self._init_pipeline()

    def _init_pipeline(self):
        if self.model_name == "rule-based-fallback":
            self.pipeline = None
            self.fallback_reason = "Explicit rule-based sentiment lexicon selected by user configuration"
            return

        try:
            from transformers import pipeline

            self.pipeline = pipeline("sentiment-analysis", model=self.model_name, top_k=None)
        except Exception as err:
            self.pipeline = None
            self.fallback_reason = f"Transformers initialization error ({type(err).__name__}): {err}"
            logger.warning("Failed to initialize Hugging Face sentiment pipeline: %s", err)

    def _fallback_analyze(self, text: str) -> SentimentResult:
        """
        Heuristic sentiment analysis fallback using keyword counting.
        Note: The score returned in fallback mode represents a heuristic ratio (pos_count / total),
        not a calibrated statistical probability.
        """
        text_lower = text.lower()
        pos_words = [
            "great",
            "excellent",
            "breakthrough",
            "promising",
            "positive",
            "innovative",
            "effective",
            "impressive",
            "success",
            "outstanding",
            "benchmark",
            "robust",
            "high-performing",
            "superior",
            "state-of-the-art",
        ]
        neg_words = [
            "error",
            "failure",
            "risk",
            "delay",
            "poor",
            "negative",
            "issue",
            "flaw",
            "bottleneck",
            "inconsistent",
            "degraded",
            "loss",
            "vulnerability",
            "limitation",
            "drawback",
        ]

        pos_count = sum(1 for w in pos_words if w in text_lower)
        neg_count = sum(1 for w in neg_words if w in text_lower)

        if pos_count == 0 and neg_count == 0:
            return SentimentResult(
                label="NEUTRAL",
                score=None,
                scores_breakdown={"POSITIVE": 0.0, "NEGATIVE": 0.0, "NEUTRAL": 1.0},
                model_name=self.model_name,
                is_fallback=True,
                is_heuristic=True,
                fallback_reason=self.fallback_reason,
            )

        total = pos_count + neg_count
        pos_ratio = pos_count / total
        neg_ratio = neg_count / total

        if abs(pos_ratio - neg_ratio) < 0.2:
            return SentimentResult(
                label="NEUTRAL",
                score=round(float(max(pos_ratio, neg_ratio)), 4),
                scores_breakdown={
                    "POSITIVE": round(pos_ratio * 0.5, 4),
                    "NEGATIVE": round(neg_ratio * 0.5, 4),
                    "NEUTRAL": 0.5,
                },
                model_name=self.model_name,
                is_fallback=True,
                is_heuristic=True,
                fallback_reason=self.fallback_reason,
            )
        elif pos_ratio > neg_ratio:
            return SentimentResult(
                label="POSITIVE",
                score=round(float(pos_ratio), 4),
                scores_breakdown={"POSITIVE": round(pos_ratio, 4), "NEGATIVE": round(neg_ratio, 4)},
                model_name=self.model_name,
                is_fallback=True,
                is_heuristic=True,
                fallback_reason=self.fallback_reason,
            )
        else:
            return SentimentResult(
                label="NEGATIVE",
                score=round(float(neg_ratio), 4),
                scores_breakdown={"POSITIVE": round(pos_ratio, 4), "NEGATIVE": round(neg_ratio, 4)},
                model_name=self.model_name,
                is_fallback=True,
                is_heuristic=True,
                fallback_reason=self.fallback_reason,
            )

    def analyze(self, text: str) -> SentimentResult:
        clean_text = text.strip()
        if not clean_text:
            return SentimentResult(
                label="NEUTRAL",
                score=1.0,
                scores_breakdown={"POSITIVE": 0.0, "NEGATIVE": 0.0, "NEUTRAL": 1.0},
                model_name=self.model_name,
                is_fallback=(self.pipeline is None),
                is_heuristic=(self.pipeline is None),
                fallback_reason=self.fallback_reason,
            )

        if self.pipeline is not None:
            try:
                results = self.pipeline(clean_text[:512])[0]
                breakdown = {}
                for item in results:
                    breakdown[item["label"].upper()] = round(float(item["score"]), 4)

                top_item = max(results, key=lambda x: x["score"])
                top_label = top_item["label"].upper()
                top_score = float(top_item["score"])

                # Threshold-based neutral detection when binary model confidence is low
                if top_score < self.neutral_threshold:
                    final_label = "NEUTRAL"
                    breakdown["NEUTRAL"] = round(1.0 - top_score, 4)
                else:
                    final_label = top_label

                return SentimentResult(
                    label=final_label,
                    score=round(top_score, 4),
                    scores_breakdown=breakdown,
                    model_name=self.model_name,
                    is_fallback=False,
                    is_heuristic=False,
                    fallback_reason=None,
                )
            except Exception as err:
                self.fallback_reason = f"Pipeline execution error ({type(err).__name__}): {err}"
                logger.warning("Sentiment analysis pipeline failed, using heuristic fallback: %s", err)
                return self._fallback_analyze(clean_text)
        else:
            return self._fallback_analyze(clean_text)
