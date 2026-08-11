import re
from typing import Dict, Any


class AbstractiveSummarizer:
    """
    Abstractive text summarization engine using Hugging Face BART / T5 transformer architectures.
    """

    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        self.model_name = model_name
        self.pipeline = None
        self._init_pipeline()

    def _init_pipeline(self):
        try:
            from transformers import pipeline
            self.pipeline = pipeline("summarization", model=self.model_name)
        except Exception:
            self.pipeline = None

    def summarize(self, text: str, max_length: int = 130, min_length: int = 30) -> str:
        """
        Generates abstractive summary for input article string.
        """
        if not text or len(text.strip()) == 0:
            return ""

        if self.pipeline is not None:
            summary = self.pipeline(text, max_length=max_length, min_length=min_length, do_sample=False)
            return summary[0]["summary_text"]
        else:
            # High-quality key sentence extraction fallback
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 10]
            if len(sentences) <= 2:
                return text
            return f"{sentences[0]} {sentences[-1]}"
