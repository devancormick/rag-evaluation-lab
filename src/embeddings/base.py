"""
Base class for embedding models.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np


class EmbeddingModel(ABC):
    """Abstract base class for all embedding models."""

    @abstractmethod
    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """
        Encode texts into embeddings.

        Args:
            texts: List of text strings to encode
            **kwargs: Additional encoding parameters

        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Return the dimension of embeddings produced by this model."""
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return the configuration of this embedding model."""
        pass

