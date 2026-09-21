import pytest
from nlp_engine.chunking import TokenAwareChunker


def test_empty_chunking():
    chunker = TokenAwareChunker(max_tokens=100, overlap_tokens=20)
    chunks = chunker.chunk_text("")
    assert chunks == []


def test_short_text_single_chunk():
    chunker = TokenAwareChunker(max_tokens=100, overlap_tokens=20)
    text = "This is a short article test. It should fit inside a single chunk easily."
    chunks = chunker.chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].token_count > 0


def test_long_text_sliding_window():
    chunker = TokenAwareChunker(max_tokens=30, overlap_tokens=10)
    text = (
        "First sentence provides initial context for testing long text chunking. "
        "Second sentence extends the context with additional domain information. "
        "Third sentence continues adding text so that multiple sliding chunks are triggered. "
        "Fourth sentence provides closing remarks for the chunker unit test verification."
    )
    chunks = chunker.chunk_text(text)
    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= 40
