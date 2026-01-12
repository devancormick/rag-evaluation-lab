"""
C99 algorithm for semantic segmentation.
"""

import re
from typing import List, Dict, Any
import numpy as np
from .base import ChunkingStrategy


class C99Chunker(ChunkingStrategy):
    """Chunk text using the C99 algorithm for semantic segmentation."""

    def __init__(
        self,
        window_size: int = 200,
        similarity_threshold: float = 0.5
    ):
        """
        Initialize C99 chunker.

        Args:
            window_size: Size of the sliding window for similarity computation
            similarity_threshold: Threshold for determining boundaries
        """
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        words = re.findall(r'\b\w+\b', text.lower())
        return words

    def _compute_word_freq(self, words: List[str]) -> Dict[str, int]:
        """Compute word frequency in a sequence."""
        freq = {}
        for word in words:
            freq[word] = freq.get(word, 0) + 1
        return freq

    def _compute_cosine_similarity(self, freq1: Dict[str, int], freq2: Dict[str, int]) -> float:
        """Compute cosine similarity between two frequency dictionaries."""
        all_words = set(freq1.keys()) | set(freq2.keys())
        if not all_words:
            return 0.0

        dot_product = sum(freq1.get(word, 0) * freq2.get(word, 0) for word in all_words)
        norm1 = np.sqrt(sum(freq1.get(word, 0) ** 2 for word in all_words))
        norm2 = np.sqrt(sum(freq2.get(word, 0) ** 2 for word in all_words))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Chunk text using C99 algorithm."""
        if metadata is None:
            metadata = {}

        words = self._tokenize(text)
        if len(words) < self.window_size * 2:
            # Text too short, return as single chunk
            return [{
                "text": text,
                "start_idx": 0,
                "end_idx": len(text),
                "metadata": {**metadata, "chunk_index": 0}
            }]

        # Compute similarity scores for each position
        similarities = []
        positions = []

        for i in range(self.window_size, len(words) - self.window_size):
            window1_words = words[i - self.window_size:i]
            window2_words = words[i:i + self.window_size]

            freq1 = self._compute_word_freq(window1_words)
            freq2 = self._compute_word_freq(window2_words)

            similarity = self._compute_cosine_similarity(freq1, freq2)
            similarities.append(similarity)
            positions.append(i)

        if not similarities:
            return [{
                "text": text,
                "start_idx": 0,
                "end_idx": len(text),
                "metadata": {**metadata, "chunk_index": 0}
            }]

        # Find boundaries where similarity drops below threshold
        boundaries = []
        for i, (pos, sim) in enumerate(zip(positions, similarities)):
            if sim < self.similarity_threshold:
                # Convert word index to approximate character position
                char_pos = len(' '.join(words[:pos]))
                boundaries.append(char_pos)

        # Remove duplicate boundaries that are too close
        min_gap = self.window_size * 5  # Minimum characters between boundaries
        filtered_boundaries = []
        for boundary in boundaries:
            if not filtered_boundaries or boundary - filtered_boundaries[-1] >= min_gap:
                filtered_boundaries.append(boundary)

        # Create chunks based on boundaries
        chunks = []
        start_idx = 0
        for boundary in filtered_boundaries:
            end_idx = min(boundary, len(text))
            chunk_text = text[start_idx:end_idx].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "metadata": {**metadata, "chunk_index": len(chunks)}
                })
            start_idx = end_idx

        # Add final chunk
        if start_idx < len(text):
            chunk_text = text[start_idx:].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "start_idx": start_idx,
                    "end_idx": len(text),
                    "metadata": {**metadata, "chunk_index": len(chunks)}
                })

        return chunks if chunks else [{
            "text": text,
            "start_idx": 0,
            "end_idx": len(text),
            "metadata": {**metadata, "chunk_index": 0}
        }]

    def get_config(self) -> Dict[str, Any]:
        """Return chunker configuration."""
        return {
            "strategy": "c99",
            "window_size": self.window_size,
            "similarity_threshold": self.similarity_threshold
        }

