"""
Experiment runner for executing evaluation experiments.
"""

import json
import time
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm

from ..config import ExperimentConfig, load_config
from ..datasets import DatasetLoader, QASPERLoader, HotpotQALoader
from ..evaluation import evaluate_rag_system
from .rag_system import RAGSystem


class ExperimentRunner:
    """Runner for executing RAG evaluation experiments."""

    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment runner.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.rag_system = RAGSystem(config)

    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load dataset based on configuration."""
        dataset_name = self.config.dataset.name
        split = self.config.dataset.split
        limit = self.config.dataset.limit

        if dataset_name == "qasper":
            loader = QASPERLoader()
        elif dataset_name == "hotpotqa":
            loader = HotpotQALoader()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        examples = loader.load(split=split)
        if limit:
            examples = examples[:limit]

        return examples

    def run(self) -> Dict[str, Any]:
        """
        Run the complete experiment.

        Returns:
            Dictionary containing experiment results
        """
        print(f"Running experiment: {self.config.experiment_name}")
        print(f"Configuration: {self.config.json(indent=2)}")

        # Load dataset
        print("Loading dataset...")
        examples = self._load_dataset()
        print(f"Loaded {len(examples)} examples")

        # Index documents
        print("Indexing documents...")
        documents = [ex["context"] for ex in examples]
        document_ids = [ex["metadata"].get("paper_id") or ex["metadata"].get("question_id", f"doc_{i}")
                       for i, ex in enumerate(examples)]
        self.rag_system.index_documents(documents, document_ids)

        # Run evaluation
        print("Running evaluation...")
        predictions = []
        retrieval_results = []
        timings = []

        for example in tqdm(examples, desc="Evaluating"):
            query = example["question"]
            result = self.rag_system.answer(query, top_k=10)

            predictions.append({
                "answer": result["answer"],
                "source_chunks": result["source_chunks"],
                "retrieved_ids": result["retrieved_ids"]
            })

            retrieval_results.append([
                {"id": rid, "score": 1.0}  # Simplified
                for rid in result["retrieved_ids"]
            ])

            timings.append(result["latency"])

        # Prepare ground truth
        ground_truth = []
        for example in examples:
            # For evaluation, we need relevant document IDs
            # In a real scenario, these would be provided by the dataset
            # For now, we'll use a simplified approach
            doc_id = example["metadata"].get("paper_id") or example["metadata"].get("question_id", "")
            ground_truth.append({
                "answer": example["answer"],
                "relevant_ids": [f"{doc_id}_chunk_{i}" for i in range(10)]  # Simplified
            })

        # Compute metrics
        print("Computing metrics...")
        metrics = evaluate_rag_system(
            predictions=predictions,
            ground_truth=ground_truth,
            retrieval_results=retrieval_results,
            timings=timings
        )

        # Prepare results
        results = {
            "experiment_name": self.config.experiment_name,
            "config": self.config.dict(),
            "metrics": metrics,
            "num_examples": len(examples),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Save results
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{self.config.experiment_name}_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {output_file}")
        print("\nMetrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

        return results


def run_experiment(config_path: str) -> Dict[str, Any]:
    """
    Run a single experiment from configuration file.

    Args:
        config_path: Path to experiment configuration file

    Returns:
        Experiment results dictionary
    """
    config = load_config(config_path)
    runner = ExperimentRunner(config)
    return runner.run()

