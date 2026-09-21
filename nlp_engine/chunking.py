import re
from typing import List, Optional
from nlp_engine.schemas import TextChunk


class TokenAwareChunker:
    """
    Token-aware sliding-window chunker for handling long documents exceeding model context limits.
    """

    def __init__(self, tokenizer=None, max_tokens: int = 512, overlap_tokens: int = 64):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.overlap_tokens = min(overlap_tokens, max_tokens // 4)

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                pass
        # Fallback whitespace word-based estimation
        return len(text.split())

    def chunk_text(self, text: str) -> List[TextChunk]:
        """
        Splits text into sliding window chunks respecting maximum token limits and sentence boundaries.
        """
        clean_text = text.strip()
        if not clean_text:
            return []

        total_tokens = self.count_tokens(clean_text)
        if total_tokens <= self.max_tokens:
            return [
                TextChunk(
                    chunk_id=0,
                    text=clean_text,
                    token_count=total_tokens,
                    start_char=0,
                    end_char=len(clean_text),
                )
            ]

        # Sentence-boundary split
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_text) if s.strip()]
        if not sentences:
            sentences = [clean_text]

        chunks: List[TextChunk] = []
        current_sentences: List[str] = []
        current_tokens = 0
        chunk_idx = 0
        start_char = 0

        for sentence in sentences:
            sent_tokens = self.count_tokens(sentence)

            # If a single sentence exceeds max_tokens, split it by whitespace
            if sent_tokens > self.max_tokens:
                words = sentence.split()
                w_buf = []
                w_tokens = 0
                for w in words:
                    wt = self.count_tokens(w)
                    if w_tokens + wt > self.max_tokens:
                        c_text = " ".join(w_buf)
                        chunks.append(
                            TextChunk(
                                chunk_id=chunk_idx,
                                text=c_text,
                                token_count=self.count_tokens(c_text),
                                start_char=start_char,
                                end_char=start_char + len(c_text),
                            )
                        )
                        chunk_idx += 1
                        start_char += len(c_text) + 1
                        w_buf = [w]
                        w_tokens = wt
                    else:
                        w_buf.append(w)
                        w_tokens += wt
                if w_buf:
                    c_text = " ".join(w_buf)
                    current_sentences = [c_text]
                    current_tokens = self.count_tokens(c_text)
                continue

            if current_tokens + sent_tokens > self.max_tokens and current_sentences:
                c_text = " ".join(current_sentences)
                chunks.append(
                    TextChunk(
                        chunk_id=chunk_idx,
                        text=c_text,
                        token_count=self.count_tokens(c_text),
                        start_char=start_char,
                        end_char=start_char + len(c_text),
                    )
                )
                chunk_idx += 1
                start_char += len(c_text) + 1

                # Calculate overlap sentences
                overlap_buf: List[str] = []
                overlap_count = 0
                for s in reversed(current_sentences):
                    st = self.count_tokens(s)
                    if overlap_count + st <= self.overlap_tokens:
                        overlap_buf.insert(0, s)
                        overlap_count += st
                    else:
                        break

                current_sentences = overlap_buf + [sentence]
                current_tokens = sum(self.count_tokens(s) for s in current_sentences)
            else:
                current_sentences.append(sentence)
                current_tokens += sent_tokens

        if current_sentences:
            c_text = " ".join(current_sentences)
            chunks.append(
                TextChunk(
                    chunk_id=chunk_idx,
                    text=c_text,
                    token_count=self.count_tokens(c_text),
                    start_char=start_char,
                    end_char=start_char + len(c_text),
                )
            )

        return chunks
