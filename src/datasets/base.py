"""
Base class for dataset loaders.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class DatasetLoader(ABC):
    """Abstract base class for all dataset loaders."""

    @abstractmethod
    def load(self, split: str = "test") -> List[Dict[str, Any]]:
        """
        Load dataset split.

        Args:
            split: Dataset split ('train', 'validation', 'test')

        Returns:
            List of examples, each containing:
            - 'question': The question text
            - 'context': The context/document text
            - 'answer': Ground truth answer(s)
            - 'metadata': Additional example metadata
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this dataset."""
        pass

