"""
QASPER dataset loader for scientific literature question-answering.
"""

from typing import List, Dict, Any
from datasets import load_dataset
from .base import DatasetLoader


class QASPERLoader(DatasetLoader):
    """Loader for QASPER dataset."""

    def __init__(self, cache_dir: str = None):
        """
        Initialize QASPER dataset loader.

        Args:
            cache_dir: Directory to cache downloaded dataset
        """
        self.cache_dir = cache_dir
        self.dataset_name = "allenai/qasper"

    def load(self, split: str = "test") -> List[Dict[str, Any]]:
        """Load QASPER dataset split."""
        try:
            dataset = load_dataset(self.dataset_name, cache_dir=self.cache_dir)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load QASPER dataset: {e}. "
                "Make sure you have internet connection and huggingface-hub installed."
            )

        if split not in dataset:
            available_splits = list(dataset.keys())
            raise ValueError(
                f"Split '{split}' not found. Available splits: {available_splits}"
            )

        split_data = dataset[split]
        examples = []

        for example in split_data:
            # Extract question
            question = example.get("question", "")

            # Extract full text from the paper
            full_text = example.get("full_text", {})
            if isinstance(full_text, dict):
                # QASPER stores text in sections
                sections = full_text.get("sections", [])
                context = "\n\n".join([
                    f"{section.get('section_name', '')}\n{section.get('section_text', '')}"
                    for section in sections
                ])
            else:
                context = str(full_text)

            # Extract answers
            answers = example.get("answers", [])
            if isinstance(answers, list) and len(answers) > 0:
                # Get unanswerable flag
                unanswerable = answers[0].get("unanswerable", False)
                if unanswerable:
                    answer = None
                else:
                    # Extract answer text
                    answer_objects = answers[0].get("answer", {})
                    if isinstance(answer_objects, dict):
                        answer = answer_objects.get("extractive_spans", [])
                        if not answer:
                            answer = answer_objects.get("free_form_answer", "")
                    else:
                        answer = answer_objects
            else:
                answer = None

            examples.append({
                "question": question,
                "context": context,
                "answer": answer,
                "metadata": {
                    "paper_id": example.get("paper_id", ""),
                    "question_id": example.get("question_id", ""),
                    "dataset": "qasper"
                }
            })

        return examples

    def get_name(self) -> str:
        """Return dataset name."""
        return "QASPER"

