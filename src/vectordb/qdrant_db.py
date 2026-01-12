"""
Qdrant vector database implementation.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)
from .base import VectorDB


class QdrantVectorDB(VectorDB):
    """Qdrant vector database implementation."""

    def __init__(
        self,
        collection_name: str = "rag_eval",
        dimension: int = 768,
        host: str = "localhost",
        port: int = 6333,
        **kwargs
    ):
        """
        Initialize Qdrant vector database.

        Args:
            collection_name: Name of the Qdrant collection
            dimension: Embedding dimension
            host: Qdrant server host
            port: Qdrant server port
            **kwargs: Additional client parameters
        """
        self.collection_name = collection_name
        self.dimension = dimension
        self.host = host
        self.port = port
        self.client_kwargs = kwargs

        self.client = QdrantClient(host=host, port=port, **kwargs)

    def create_index(self, dimension: int, **kwargs):
        """Create a new collection with the specified dimension."""
        self.dimension = dimension

        # Check if collection exists
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name in collection_names:
            return

        # Create collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE
            )
        )

    def add_vectors(
        self,
        vectors: np.ndarray,
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ):
        """Add vectors to Qdrant."""
        if self.dimension is None:
            self.dimension = vectors.shape[1]

        # Create collection if it doesn't exist
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        if self.collection_name not in collection_names:
            self.create_index(self.dimension)

        n_vectors = vectors.shape[0]
        if ids is None:
            ids = [f"vec_{i}" for i in range(n_vectors)]

        if metadata is None:
            metadata = [{}] * n_vectors

        # Prepare points
        points = []
        for i, (vector, vec_id, meta) in enumerate(zip(vectors, ids, metadata)):
            points.append(
                PointStruct(
                    id=i,  # Qdrant uses integer IDs
                    vector=vector.tolist(),
                    payload={
                        "text_id": vec_id,
                        "metadata": meta
                    }
                )
            )

        # Upsert points
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        # Ensure query is 1D
        if query_vector.ndim > 1:
            query_vector = query_vector[0]

        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k
        )

        # Format results
        formatted_results = []
        for result in results:
            payload = result.payload or {}
            metadata = payload.get("metadata", {})
            text_id = payload.get("text_id", str(result.id))

            formatted_results.append({
                "id": text_id,
                "score": float(result.score),
                "metadata": metadata
            })

        return formatted_results

    def clear(self):
        """Clear all vectors from the database."""
        try:
            self.client.delete_collection(self.collection_name)
        except:
            pass

    def get_config(self) -> Dict[str, Any]:
        """Return database configuration."""
        return {
            "backend": "qdrant",
            "collection_name": self.collection_name,
            "dimension": self.dimension,
            "host": self.host,
            "port": self.port
        }

