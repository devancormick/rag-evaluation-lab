"""
Tests for chunking strategies.
"""

import pytest
from src.chunking import (
    FixedSizeChunker,
    TextTilingChunker,
    C99Chunker,
    LayoutAwareChunker
)


def test_fixed_size_chunker():
    """Test fixed-size chunking."""
    text = "This is a test document. " * 50
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk(text)

    assert len(chunks) > 0
    assert all("text" in chunk for chunk in chunks)
    assert all("start_idx" in chunk for chunk in chunks)


def test_texttiling_chunker():
    """Test TextTiling chunking."""
    text = "First topic. " * 20 + "Second topic. " * 20
    chunker = TextTilingChunker(w=10, k=5)
    chunks = chunker.chunk(text)

    assert len(chunks) > 0


def test_c99_chunker():
    """Test C99 chunking."""
    text = "First section. " * 30 + "Second section. " * 30
    chunker = C99Chunker(window_size=50, similarity_threshold=0.5)
    chunks = chunker.chunk(text)

    assert len(chunks) > 0


def test_layout_aware_chunker():
    """Test layout-aware chunking."""
    text = "Header\n\nParagraph one. " * 10
    chunker = LayoutAwareChunker(max_chunk_size=200)
    chunks = chunker.chunk(text)

    assert len(chunks) > 0

