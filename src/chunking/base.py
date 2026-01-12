"""
Base class for chunking strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class ChunkingStrategy(ABC):
    """Abstract base class for all chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Chunk a document into segments.

        Args:
            text: The input text to chunk
            metadata: Optional metadata about the document

        Returns:
            List of chunk dictionaries, each containing:
            - 'text': The chunk text
            - 'start_idx': Start character index in original text
            - 'end_idx': End character index in original text
            - 'metadata': Additional chunk metadata
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return the configuration of this chunking strategy."""
        pass

