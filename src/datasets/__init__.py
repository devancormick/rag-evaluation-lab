"""
Dataset loaders for evaluation datasets.
"""

from .base import DatasetLoader
from .qasper import QASPERLoader
from .hotpotqa import HotpotQALoader

__all__ = [
    "DatasetLoader",
    "QASPERLoader",
    "HotpotQALoader",
]

