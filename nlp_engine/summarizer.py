import re
from typing import List, Optional
from nlp_engine.schemas import SummarizationResult, TextChunk
from nlp_engine.chunking import TokenAwareChunker


class AbstractiveSummarizer:
    """
    Hierarchical abstractive text summarizer handling both short and long documents.
    """

    def __init__(self, model_name: str = "facebook/bart-large-cnn", device: int = -1):
        self.model_name = model_name
        self.device = device
        self.pipeline = None
        self.tokenizer = None
        self._init_pipeline()
        self.chunker = TokenAwareChunker(tokenizer=self.tokenizer, max_tokens=512, overlap_tokens=64)

    def _init_pipeline(self):
        try:
            from transformers import pipeline, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.pipeline = pipeline("summarization", model=self.model_name, device=self.device)
        except Exception:
            self.pipeline = None
            self.tokenizer = None

    def _fallback_summary(self, text: str, max_words: int = 60) -> str:
        """
        Extractive heuristic summary fallback when Hugging Face models are offline or uninstalled.
        """
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 10]
        if not sentences:
            return text[:max_words * 6]

        if len(sentences) <= 2:
            return " ".join(sentences)

        # Select first sentence, highest sentence with numerical/action content, and last sentence
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
            except Exception:
                return self._fallback_summary(chunk_text)
        else:
            return self._fallback_summary(chunk_text)

    def summarize(
        self,
        text: str,
        max_length: int = 130,
        min_length: int = 30,
        hierarchical: bool = True,
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
            )

        chunks: List[TextChunk] = self.chunker.chunk_text(clean_text)
        chunk_summaries: List[str] = []

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

        input_tokens = self.chunker.count_tokens(clean_text)
        summary_tokens = self.chunker.count_tokens(final_summary)
        compression = round((1.0 - (summary_tokens / max(input_tokens, 1))) * 100, 2)

        return SummarizationResult(
            summary_text=final_summary,
            chunk_summaries=chunk_summaries,
            input_token_count=input_tokens,
            summary_token_count=summary_tokens,
            compression_ratio=compression,
            model_name=self.model_name,
            is_fallback=(self.pipeline is None),
        )
