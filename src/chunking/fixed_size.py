"""
Fixed-size window chunking strategy.
"""

from typing import List, Dict, Any
from .base import ChunkingStrategy


class FixedSizeChunker(ChunkingStrategy):
    """Chunk text using fixed-size windows with optional overlap."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separator: str = "\n\n"
    ):
        """
        Initialize fixed-size chunker.

        Args:
            chunk_size: Maximum number of characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
            separator: String to use for splitting (if needed)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Chunk text into fixed-size windows."""
        if metadata is None:
            metadata = {}

        chunks = []
        start_idx = 0
        text_length = len(text)

        while start_idx < text_length:
            end_idx = min(start_idx + self.chunk_size, text_length)
            chunk_text = text[start_idx:end_idx]

            chunks.append({
                "text": chunk_text,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "metadata": {**metadata, "chunk_index": len(chunks)}
            })

            # Move start index forward, accounting for overlap
            start_idx = end_idx - self.chunk_overlap
            if start_idx >= text_length:
                break

        return chunks

    def get_config(self) -> Dict[str, Any]:
        """Return chunker configuration."""
        return {
            "strategy": "fixed_size",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "separator": self.separator
        }

