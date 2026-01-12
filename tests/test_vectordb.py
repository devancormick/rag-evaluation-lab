"""
Tests for vector databases.
"""

import pytest
import numpy as np
from src.vectordb import FAISSVectorDB


def test_faiss_flat():
    """Test FAISS flat index."""
    db = FAISSVectorDB(index_type="flat", dimension=128, metric="cosine")
    db.create_index(128)

    # Generate random vectors
    vectors = np.random.randn(10, 128).astype('float32')
    ids = [f"vec_{i}" for i in range(10)]
    metadata = [{"index": i} for i in range(10)]

    db.add_vectors(vectors, ids, metadata)

    # Search
    query = np.random.randn(128).astype('float32')
    results = db.search(query, top_k=5)

    assert len(results) == 5
    assert all("id" in r for r in results)
    assert all("score" in r for r in results)

