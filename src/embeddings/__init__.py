"""
Embedding model wrappers for various model families.
"""

from .base import EmbeddingModel
from .e5 import E5EmbeddingModel
from .bge import BGEEmbeddingModel

__all__ = [
    "EmbeddingModel",
    "E5EmbeddingModel",
    "BGEEmbeddingModel",
]

