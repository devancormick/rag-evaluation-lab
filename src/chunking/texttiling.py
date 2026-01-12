"""
TextTiling algorithm for semantic segmentation.
"""

import re
from typing import List, Dict, Any
import numpy as np
from .base import ChunkingStrategy


class TextTilingChunker(ChunkingStrategy):
    """Chunk text using the TextTiling algorithm for semantic segmentation."""

    def __init__(
        self,
        w: int = 20,
        k: int = 10,
        smoothing_width: int = 3,
        smoothing_rounds: int = 1
    ):
        """
        Initialize TextTiling chunker.

        Args:
            w: Size of the pseudo-sentences window
            k: Size of the gap between pseudo-sentences
            smoothing_width: Width of the smoothing window
            smoothing_rounds: Number of smoothing rounds
        """
        self.w = w
        self.k = k
        self.smoothing_width = smoothing_width
        self.smoothing_rounds = smoothing_rounds

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Simple word tokenization
        words = re.findall(r'\b\w+\b', text.lower())
        return words

    def _create_pseudo_sentences(self, words: List[str]) -> List[List[str]]:
        """Create pseudo-sentences of size w."""
        pseudo_sentences = []
        for i in range(0, len(words), self.w):
            pseudo_sentences.append(words[i:i + self.w])
        return pseudo_sentences

    def _compute_similarity(self, seq1: List[str], seq2: List[str]) -> float:
        """Compute similarity between two sequences using word overlap."""
        set1 = set(seq1)
        set2 = set(seq2)
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def _compute_depth_scores(self, pseudo_sentences: List[List[str]]) -> List[float]:
        """Compute depth scores for each gap."""
        depth_scores = []
        for i in range(len(pseudo_sentences) - 1):
            # Compute similarity between sequences before and after gap
            before = []
            after = []
            
            for j in range(max(0, i - self.k + 1), i + 1):
                before.extend(pseudo_sentences[j])
            
            for j in range(i + 1, min(len(pseudo_sentences), i + 1 + self.k)):
                after.extend(pseudo_sentences[j])
            
            similarity = self._compute_similarity(before, after)
            depth_scores.append(1.0 - similarity)  # Lower similarity = higher depth
        
        return depth_scores

    def _smooth_scores(self, scores: List[float]) -> List[float]:
        """Apply smoothing to depth scores."""
        smoothed = scores.copy()
        for _ in range(self.smoothing_rounds):
            new_scores = []
            for i in range(len(smoothed)):
                start = max(0, i - self.smoothing_width)
                end = min(len(smoothed), i + self.smoothing_width + 1)
                new_scores.append(np.mean(smoothed[start:end]))
            smoothed = new_scores
        return smoothed

    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Chunk text using TextTiling algorithm."""
        if metadata is None:
            metadata = {}

        # Split text into sentences (simple approach)
        sentences = re.split(r'[.!?]+\s+', text)
        words = self._tokenize(text)
        
        if len(words) < self.w * 2:
            # Text too short, return as single chunk
            return [{
                "text": text,
                "start_idx": 0,
                "end_idx": len(text),
                "metadata": {**metadata, "chunk_index": 0}
            }]

        pseudo_sentences = self._create_pseudo_sentences(words)
        depth_scores = self._compute_depth_scores(pseudo_sentences)
        smoothed_scores = self._smooth_scores(depth_scores)

        # Find boundaries (peaks in depth scores)
        if not smoothed_scores:
            return [{
                "text": text,
                "start_idx": 0,
                "end_idx": len(text),
                "metadata": {**metadata, "chunk_index": 0}
            }]

        # Use mean + std as threshold for boundaries
        mean_score = np.mean(smoothed_scores)
        std_score = np.std(smoothed_scores)
        threshold = mean_score + 0.5 * std_score

        boundaries = []
        for i, score in enumerate(smoothed_scores):
            if score > threshold:
                # Convert pseudo-sentence index to character position
                char_pos = len(' '.join(words[:i * self.w + self.w]))
                boundaries.append(char_pos)

        # Create chunks based on boundaries
        chunks = []
        start_idx = 0
        for i, boundary in enumerate(boundaries):
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
            "strategy": "texttiling",
            "w": self.w,
            "k": self.k,
            "smoothing_width": self.smoothing_width,
            "smoothing_rounds": self.smoothing_rounds
        }

