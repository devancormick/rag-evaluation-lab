"""
FAISS vector database implementation.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from .base import VectorDB


class FAISSVectorDB(VectorDB):
    """FAISS vector database implementation with multiple index types."""

    INDEX_TYPES = {
        "flat": "Flat",
        "ivf": "IVF",
        "hnsw": "HNSW"
    }

    def __init__(
        self,
        index_type: str = "flat",
        dimension: int = None,
        metric: str = "cosine",
        **index_kwargs
    ):
        """
        Initialize FAISS vector database.

        Args:
            index_type: Type of index ('flat', 'ivf', 'hnsw')
            dimension: Embedding dimension (required for some index types)
            metric: Distance metric ('cosine', 'l2', 'inner_product')
            **index_kwargs: Additional index-specific parameters
        """
        self.index_type = index_type.lower()
        if self.index_type not in self.INDEX_TYPES:
            raise ValueError(
                f"Unknown index type: {index_type}. "
                f"Available: {list(self.INDEX_TYPES.keys())}"
            )

        self.dimension = dimension
        self.metric = metric
        self.index_kwargs = index_kwargs

        self.index = None
        self.metadata_store = {}  # Store metadata by ID
        self.id_to_index = {}  # Map string IDs to FAISS indices
        self.index_to_id = {}  # Map FAISS indices to string IDs
        self.next_index = 0

        if dimension is not None:
            self.create_index(dimension, **index_kwargs)

    def _create_flat_index(self, dimension: int) -> faiss.Index:
        """Create a flat (brute-force) index."""
        if self.metric == "cosine":
            index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors
        elif self.metric == "l2":
            index = faiss.IndexFlatL2(dimension)
        else:
            raise ValueError(f"Unsupported metric for flat index: {self.metric}")
        return index

    def _create_ivf_index(self, dimension: int, nlist: int = 100) -> faiss.Index:
        """Create an IVF (Inverted File) index."""
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        return index

    def _create_hnsw_index(self, dimension: int, M: int = 32, ef_construction: int = 200) -> faiss.Index:
        """Create an HNSW (Hierarchical Navigable Small World) index."""
        index = faiss.IndexHNSWFlat(dimension, M)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_construction
        return index

    def create_index(self, dimension: int, **kwargs):
        """Create a new index with the specified dimension."""
        self.dimension = dimension
        self.index_kwargs.update(kwargs)

        if self.index_type == "flat":
            self.index = self._create_flat_index(dimension)
        elif self.index_type == "ivf":
            nlist = kwargs.get("nlist", 100)
            self.index = self._create_ivf_index(dimension, nlist)
        elif self.index_type == "hnsw":
            M = kwargs.get("M", 32)
            ef_construction = kwargs.get("ef_construction", 200)
            self.index = self._create_hnsw_index(dimension, M, ef_construction)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        # Reset metadata stores
        self.metadata_store = {}
        self.id_to_index = {}
        self.index_to_id = {}
        self.next_index = 0

    def add_vectors(
        self,
        vectors: np.ndarray,
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ):
        """Add vectors to the FAISS index."""
        if self.index is None:
            if self.dimension is None:
                self.dimension = vectors.shape[1]
            self.create_index(self.dimension, **self.index_kwargs)

        # Normalize for cosine similarity if needed
        if self.metric == "cosine":
            faiss.normalize_L2(vectors)

        n_vectors = vectors.shape[0]

        # Generate IDs if not provided
        if ids is None:
            ids = [f"vec_{self.next_index + i}" for i in range(n_vectors)]

        if len(ids) != n_vectors:
            raise ValueError(f"Number of IDs ({len(ids)}) must match number of vectors ({n_vectors})")

        # Train IVF index if needed
        if self.index_type == "ivf" and not self.index.is_trained:
            self.index.train(vectors)

        # Add vectors
        start_idx = self.next_index
        self.index.add(vectors.astype('float32'))

        # Store metadata and ID mappings
        for i, vec_id in enumerate(ids):
            idx = start_idx + i
            self.id_to_index[vec_id] = idx
            self.index_to_id[idx] = vec_id
            if metadata and i < len(metadata):
                self.metadata_store[vec_id] = metadata[i]
            else:
                self.metadata_store[vec_id] = {}

        self.next_index += n_vectors

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        if self.index is None:
            return []

        # Normalize query vector for cosine similarity
        if self.metric == "cosine":
            query_vector = query_vector.astype('float32')
            faiss.normalize_L2(query_vector.reshape(1, -1))
            query_vector = query_vector.reshape(-1)

        # Ensure query is 2D
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # Search
        distances, indices = self.index.search(query_vector.astype('float32'), top_k)

        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for invalid results
                continue

            vec_id = self.index_to_id.get(idx, f"idx_{idx}")
            metadata = self.metadata_store.get(vec_id, {})

            # Convert distance to similarity score
            if self.metric == "cosine":
                score = float(dist)  # Already inner product for normalized vectors
            elif self.metric == "l2":
                score = float(1.0 / (1.0 + dist))  # Convert distance to similarity
            else:
                score = float(dist)

            results.append({
                "id": vec_id,
                "score": score,
                "metadata": metadata
            })

        return results

    def clear(self):
        """Clear all vectors from the database."""
        if self.index is not None:
            self.index.reset()
        self.metadata_store = {}
        self.id_to_index = {}
        self.index_to_id = {}
        self.next_index = 0

    def get_config(self) -> Dict[str, Any]:
        """Return database configuration."""
        return {
            "backend": "faiss",
            "index_type": self.index_type,
            "dimension": self.dimension,
            "metric": self.metric,
            "index_kwargs": self.index_kwargs,
            "num_vectors": self.next_index
        }

