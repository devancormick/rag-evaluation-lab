"""
Milvus vector database implementation.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility
)
from .base import VectorDB


class MilvusVectorDB(VectorDB):
    """Milvus vector database implementation."""

    def __init__(
        self,
        collection_name: str = "rag_eval",
        dimension: int = 768,
        host: str = "localhost",
        port: int = 19530,
        **kwargs
    ):
        """
        Initialize Milvus vector database.

        Args:
            collection_name: Name of the Milvus collection
            dimension: Embedding dimension
            host: Milvus server host
            port: Milvus server port
            **kwargs: Additional connection parameters
        """
        self.collection_name = collection_name
        self.dimension = dimension
        self.host = host
        self.port = port
        self.connection_kwargs = kwargs

        self.collection = None
        self._connected = False

    def _ensure_connection(self):
        """Ensure connection to Milvus is established."""
        if not self._connected:
            try:
                connections.connect(
                    alias="default",
                    host=self.host,
                    port=self.port,
                    **self.connection_kwargs
                )
                self._connected = True
            except Exception as e:
                raise ConnectionError(f"Failed to connect to Milvus: {e}")

    def create_index(self, dimension: int, **kwargs):
        """Create a new collection with the specified dimension."""
        self.dimension = dimension
        self._ensure_connection()

        # Check if collection exists
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            return

        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
            FieldSchema(name="text_id", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=2000)
        ]

        schema = CollectionSchema(fields, "RAG evaluation collection")
        self.collection = Collection(self.collection_name, schema)

        # Create index
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200}
        }
        self.collection.create_index("vector", index_params)

    def add_vectors(
        self,
        vectors: np.ndarray,
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ):
        """Add vectors to Milvus."""
        if self.collection is None:
            self.create_index(vectors.shape[1])

        import json

        n_vectors = vectors.shape[0]
        if ids is None:
            ids = [f"vec_{i}" for i in range(n_vectors)]

        if metadata is None:
            metadata = [{}] * n_vectors

        # Prepare data
        vectors_list = vectors.tolist()
        metadata_strs = [json.dumps(m) for m in metadata]

        # Insert data
        data = [
            vectors_list,
            ids,
            metadata_strs
        ]

        self.collection.insert(data)
        self.collection.load()

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        if self.collection is None:
            return []

        import json

        # Ensure collection is loaded
        if not self.collection.has_index():
            return []

        self.collection.load()

        # Prepare query
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        search_params = {"metric_type": "COSINE", "params": {"ef": 200}}

        # Search
        results = self.collection.search(
            data=query_vector.tolist(),
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=["text_id", "metadata"]
        )

        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                metadata_str = hit.entity.get("metadata", "{}")
                try:
                    metadata = json.loads(metadata_str)
                except:
                    metadata = {}

                formatted_results.append({
                    "id": hit.entity.get("text_id", str(hit.id)),
                    "score": float(hit.score),
                    "metadata": metadata
                })

        return formatted_results

    def clear(self):
        """Clear all vectors from the database."""
        if self.collection is not None:
            utility.drop_collection(self.collection_name)
            self.collection = None

    def get_config(self) -> Dict[str, Any]:
        """Return database configuration."""
        return {
            "backend": "milvus",
            "collection_name": self.collection_name,
            "dimension": self.dimension,
            "host": self.host,
            "port": self.port
        }

