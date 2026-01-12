"""
BGE embedding model family implementations.
"""

from typing import List, Dict, Any
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from .base import EmbeddingModel


class BGEEmbeddingModel(EmbeddingModel):
    """BGE (BAAI General Embedding) model wrapper."""

    MODEL_MAP = {
        "bge-base": "BAAI/bge-base-en-v1.5",
        "bge-large": "BAAI/bge-large-en-v1.5",
        "bge-small": "BAAI/bge-small-en-v1.5",
        "bge-base-zh": "BAAI/bge-base-zh-v1.5",
        "bge-large-zh": "BAAI/bge-large-zh-v1.5",
    }

    def __init__(
        self,
        model_name: str = "bge-base",
        device: str = None,
        normalize: bool = True
    ):
        """
        Initialize BGE embedding model.

        Args:
            model_name: Name of the BGE model variant
            device: Device to run on ('cuda', 'cpu', or None for auto)
            normalize: Whether to normalize embeddings
        """
        if model_name not in self.MODEL_MAP:
            raise ValueError(
                f"Unknown BGE model: {model_name}. "
                f"Available: {list(self.MODEL_MAP.keys())}"
            )

        self.model_name = model_name
        self.hf_model_name = self.MODEL_MAP[model_name]
        self.normalize = normalize

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load model using sentence-transformers
        self.model = SentenceTransformer(self.hf_model_name, device=self.device)
        self._embedding_dim = self.model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        **kwargs
    ) -> np.ndarray:
        """Encode texts into embeddings."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            **kwargs
        )
        return embeddings

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        """Encode queries (BGE models handle queries/passages the same way)."""
        return self.encode(queries, **kwargs)

    def encode_passages(self, passages: List[str], **kwargs) -> np.ndarray:
        """Encode passages."""
        return self.encode(passages, **kwargs)

    def get_embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self._embedding_dim

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration."""
        return {
            "model_family": "bge",
            "model_name": self.model_name,
            "hf_model_name": self.hf_model_name,
            "embedding_dim": self._embedding_dim,
            "device": self.device,
            "normalize": self.normalize
        }

