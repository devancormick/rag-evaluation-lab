"""
Tests for embedding models.
"""

import pytest
import numpy as np
from src.embeddings import E5EmbeddingModel, BGEEmbeddingModel


@pytest.mark.skip(reason="Requires model download")
def test_e5_embedding():
    """Test E5 embedding model."""
    model = E5EmbeddingModel(model_name="e5-small", device="cpu")
    texts = ["This is a test.", "Another test sentence."]
    embeddings = model.encode_passages(texts)

    assert embeddings.shape[0] == len(texts)
    assert embeddings.shape[1] == model.get_embedding_dim()


@pytest.mark.skip(reason="Requires model download")
def test_bge_embedding():
    """Test BGE embedding model."""
    model = BGEEmbeddingModel(model_name="bge-small", device="cpu")
    texts = ["This is a test.", "Another test sentence."]
    embeddings = model.encode(texts)

    assert embeddings.shape[0] == len(texts)
    assert embeddings.shape[1] == model.get_embedding_dim()

