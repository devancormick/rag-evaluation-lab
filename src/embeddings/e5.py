"""
E5 embedding model family implementations.
"""

from typing import List, Dict, Any
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from .base import EmbeddingModel


class E5EmbeddingModel(EmbeddingModel):
    """E5 embedding model wrapper."""

    MODEL_MAP = {
        "e5-base": "intfloat/e5-base",
        "e5-large": "intfloat/e5-large",
        "e5-base-v2": "intfloat/e5-base-v2",
        "e5-large-v2": "intfloat/e5-large-v2",
        "e5-small": "intfloat/e5-small",
        "e5-small-v2": "intfloat/e5-small-v2",
        "multilingual-e5-base": "intfloat/multilingual-e5-base",
        "multilingual-e5-large": "intfloat/multilingual-e5-large",
    }

    def __init__(
        self,
        model_name: str = "e5-base",
        device: str = None,
        normalize: bool = True
    ):
        """
        Initialize E5 embedding model.

        Args:
            model_name: Name of the E5 model variant
            device: Device to run on ('cuda', 'cpu', or None for auto)
            normalize: Whether to normalize embeddings
        """
        if model_name not in self.MODEL_MAP:
            raise ValueError(
                f"Unknown E5 model: {model_name}. "
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

    def _prepare_texts(self, texts: List[str], instruction: str = None) -> List[str]:
        """Prepare texts with E5 instruction prefix if needed."""
        # E5 models expect instruction prefixes for certain tasks
        # For retrieval, we use "query: " or "passage: "
        if instruction:
            return [f"{instruction} {text}" for text in texts]
        return texts

    def encode(
        self,
        texts: List[str],
        instruction: str = None,
        batch_size: int = 32,
        **kwargs
    ) -> np.ndarray:
        """Encode texts into embeddings."""
        prepared_texts = self._prepare_texts(texts, instruction)
        embeddings = self.model.encode(
            prepared_texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            **kwargs
        )
        return embeddings

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        """Encode queries with 'query: ' instruction."""
        return self.encode(queries, instruction="query:", **kwargs)

    def encode_passages(self, passages: List[str], **kwargs) -> np.ndarray:
        """Encode passages with 'passage: ' instruction."""
        return self.encode(passages, instruction="passage:", **kwargs)

    def get_embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self._embedding_dim

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration."""
        return {
            "model_family": "e5",
            "model_name": self.model_name,
            "hf_model_name": self.hf_model_name,
            "embedding_dim": self._embedding_dim,
            "device": self.device,
            "normalize": self.normalize
        }

