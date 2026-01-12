"""
Base class for vector database implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np


class VectorDB(ABC):
    """Abstract base class for all vector database implementations."""

    @abstractmethod
    def create_index(self, dimension: int, **kwargs):
        """Create a new index with the specified dimension."""
        pass

    @abstractmethod
    def add_vectors(
        self,
        vectors: np.ndarray,
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Add vectors to the database.

        Args:
            vectors: numpy array of shape (n_vectors, dimension)
            ids: Optional list of IDs for each vector
            metadata: Optional list of metadata dictionaries
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query vector of shape (dimension,)
            top_k: Number of results to return

        Returns:
            List of result dictionaries, each containing:
            - 'id': Vector ID
            - 'score': Similarity score
            - 'metadata': Associated metadata
        """
        pass

    @abstractmethod
    def clear(self):
        """Clear all vectors from the database."""
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return the configuration of this vector database."""
        pass

