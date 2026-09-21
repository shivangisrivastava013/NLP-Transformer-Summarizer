from typing import Dict, Any, List
from nlp_engine.schemas import SentimentResult


class SentimentAnalyzer:
    """
    3-Class Sentiment Classification Engine (POSITIVE, NEGATIVE, NEUTRAL) with score calibration.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        neutral_threshold: float = 0.58,
    ):
        self.model_name = model_name
        self.neutral_threshold = neutral_threshold
        self.pipeline = None
        self._init_pipeline()

    def _init_pipeline(self):
        try:
            from transformers import pipeline
            self.pipeline = pipeline("sentiment-analysis", model=self.model_name, top_k=None)
        except Exception:
            self.pipeline = None

    def _fallback_analyze(self, text: str) -> SentimentResult:
        text_lower = text.lower()
        pos_words = [
            "great", "excellent", "breakthrough", "promising", "positive",
            "innovative", "effective", "impressive", "success", "outstanding",
            "benchmark", "robust", "high-performing", "superior", "state-of-the-art"
        ]
        neg_words = [
            "error", "failure", "risk", "delay", "poor", "negative",
            "issue", "flaw", "bottleneck", "inconsistent", "degraded",
            "loss", "vulnerability", "limitation", "drawback"
        ]

        pos_count = sum(1 for w in pos_words if w in text_lower)
        neg_count = sum(1 for w in neg_words if w in text_lower)

        if pos_count == 0 and neg_count == 0:
            return SentimentResult(
                label="NEUTRAL",
                score=0.8500,
                scores_breakdown={"POSITIVE": 0.0750, "NEGATIVE": 0.0750, "NEUTRAL": 0.8500},
                model_name=self.model_name,
                is_fallback=True,
            )

        total = pos_count + neg_count
        pos_ratio = pos_count / total
        neg_ratio = neg_count / total

        if abs(pos_ratio - neg_ratio) < 0.2:
            return SentimentResult(
                label="NEUTRAL",
                score=0.7200,
                scores_breakdown={"POSITIVE": round(pos_ratio * 0.4, 4), "NEGATIVE": round(neg_ratio * 0.4, 4), "NEUTRAL": 0.7200},
                model_name=self.model_name,
                is_fallback=True,
            )
        elif pos_ratio > neg_ratio:
            score = round(0.55 + pos_ratio * 0.40, 4)
            return SentimentResult(
                label="POSITIVE",
                score=score,
                scores_breakdown={"POSITIVE": score, "NEGATIVE": round(1.0 - score, 4), "NEUTRAL": 0.0},
                model_name=self.model_name,
                is_fallback=True,
            )
        else:
            score = round(0.55 + neg_ratio * 0.40, 4)
            return SentimentResult(
                label="NEGATIVE",
                score=score,
                scores_breakdown={"POSITIVE": round(1.0 - score, 4), "NEGATIVE": score, "NEUTRAL": 0.0},
                model_name=self.model_name,
                is_fallback=True,
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

                # Calibration for neutral classification on low margin outputs
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
                )
            except Exception:
                return self._fallback_analyze(clean_text)
        else:
            return self._fallback_analyze(clean_text)
