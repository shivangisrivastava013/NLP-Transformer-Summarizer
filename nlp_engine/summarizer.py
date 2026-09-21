import logging
import re
from typing import Optional

from nlp_engine.chunking import TokenAwareChunker
from nlp_engine.schemas import SummarizationResult, TextChunk

logger = logging.getLogger(__name__)


class AbstractiveSummarizer:
    """
    Hierarchical abstractive text summarizer handling both short and long documents.
    """

    def __init__(self, model_name: str = "facebook/bart-large-cnn", device: int = -1):
        self.model_name = model_name
        self.device = device
        self.pipeline = None
        self.tokenizer = None
        self.fallback_reason: Optional[str] = None
        self._init_pipeline()

    def _init_pipeline(self):
        if self.model_name == "rule-based-fallback":
            self.pipeline = None
            self.tokenizer = None
            self.fallback_reason = "Explicit rule-based heuristic fallback selected by user configuration"
            return

        try:
            from transformers import AutoTokenizer, pipeline

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.pipeline = pipeline("summarization", model=self.model_name, device=self.device)
        except Exception as err:
            self.pipeline = None
            self.tokenizer = None
            self.fallback_reason = f"Transformers initialization error ({type(err).__name__}): {err}"
            logger.warning("Failed to initialize Hugging Face summarization pipeline: %s", err)

    def _fallback_summary(self, text: str, max_words: int = 60) -> str:
        """
        Extractive heuristic summary fallback when Hugging Face models are offline or uninstalled.
        """
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 10]
        if not sentences:
            return text[: max_words * 6]

        if len(sentences) <= 2:
            return " ".join(sentences)

        selected = [sentences[0]]
        mid = len(sentences) // 2
        if mid > 0 and mid < len(sentences) - 1:
            selected.append(sentences[mid])
        if len(sentences) > 1:
            selected.append(sentences[-1])

        res = " ".join(selected)
        words = res.split()
        if len(words) > max_words:
            res = " ".join(words[:max_words]) + "..."
        return res

    def summarize_chunk(self, chunk_text: str, max_length: int = 130, min_length: int = 30) -> str:
        if not chunk_text or len(chunk_text.strip()) == 0:
            return ""

        if self.pipeline is not None:
            try:
                out = self.pipeline(
                    chunk_text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                    truncation=True,
                )
                return out[0]["summary_text"]
            except Exception as err:
                self.fallback_reason = f"Pipeline execution error ({type(err).__name__}): {err}"
                logger.warning("Summarization pipeline inference failed, using heuristic fallback: %s", err)
                return self._fallback_summary(chunk_text)
        else:
            return self._fallback_summary(chunk_text)

    def summarize(
        self,
        text: str,
        max_length: int = 130,
        min_length: int = 30,
        hierarchical: bool = True,
        max_chunk_tokens: int = 512,
        overlap_tokens: int = 64,
    ) -> SummarizationResult:
        clean_text = text.strip()
        if not clean_text:
            return SummarizationResult(
                summary_text="",
                chunk_summaries=[],
                input_token_count=0,
                summary_token_count=0,
                compression_ratio=0.0,
                model_name=self.model_name,
                is_fallback=(self.pipeline is None),
                fallback_reason=self.fallback_reason,
            )

        chunker = TokenAwareChunker(
            tokenizer=self.tokenizer,
            max_tokens=max_chunk_tokens,
            overlap_tokens=overlap_tokens,
        )
        chunks: list[TextChunk] = chunker.chunk_text(clean_text)
        chunk_summaries: list[str] = []

        for chunk in chunks:
            summary_part = self.summarize_chunk(chunk.text, max_length=max_length, min_length=min_length)
            chunk_summaries.append(summary_part)

        if len(chunks) == 1 or not hierarchical:
            final_summary = " ".join(chunk_summaries)
        else:
            # Hierarchical second-stage aggregation over chunk summaries
            aggregated_chunk_text = " ".join(chunk_summaries)
            final_summary = self.summarize_chunk(
                aggregated_chunk_text,
                max_length=max_length,
                min_length=min_length,
            )

        input_tokens = chunker.count_tokens(clean_text)
        summary_tokens = chunker.count_tokens(final_summary)
        compression = round((1.0 - (summary_tokens / max(input_tokens, 1))) * 100, 2)

        return SummarizationResult(
            summary_text=final_summary,
            chunk_summaries=chunk_summaries,
            input_token_count=input_tokens,
            summary_token_count=summary_tokens,
            compression_ratio=compression,
            model_name=self.model_name,
            is_fallback=(self.pipeline is None),
            fallback_reason=self.fallback_reason,
        )
