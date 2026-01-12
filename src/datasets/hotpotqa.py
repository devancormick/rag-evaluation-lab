"""
HotpotQA dataset loader for multi-hop reasoning tasks.
"""

from typing import List, Dict, Any
from datasets import load_dataset
from .base import DatasetLoader


class HotpotQALoader(DatasetLoader):
    """Loader for HotpotQA dataset."""

    def __init__(self, cache_dir: str = None):
        """
        Initialize HotpotQA dataset loader.

        Args:
            cache_dir: Directory to cache downloaded dataset
        """
        self.cache_dir = cache_dir
        self.dataset_name = "hotpot_qa"

    def load(self, split: str = "test") -> List[Dict[str, Any]]:
        """Load HotpotQA dataset split."""
        try:
            dataset = load_dataset(self.dataset_name, "fullwiki", cache_dir=self.cache_dir)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load HotpotQA dataset: {e}. "
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

            # Extract context from supporting facts
            context_parts = example.get("context", {})
            if isinstance(context_parts, dict):
                # HotpotQA provides context as title-sentence pairs
                context_sentences = []
                for title, sentences in context_parts.items():
                    if isinstance(sentences, list):
                        context_sentences.append(f"{title}\n" + "\n".join(sentences))
                    else:
                        context_sentences.append(f"{title}\n{sentences}")
                context = "\n\n".join(context_sentences)
            else:
                context = str(context_parts)

            # Extract answer
            answer = example.get("answer", "")
            if isinstance(answer, list):
                answer = answer[0] if answer else ""

            # Extract supporting facts
            supporting_facts = example.get("supporting_facts", [])

            examples.append({
                "question": question,
                "context": context,
                "answer": answer,
                "metadata": {
                    "question_id": example.get("_id", ""),
                    "type": example.get("type", ""),
                    "level": example.get("level", ""),
                    "supporting_facts": supporting_facts,
                    "dataset": "hotpotqa"
                }
            })

        return examples

    def get_name(self) -> str:
        """Return dataset name."""
        return "HotpotQA"

