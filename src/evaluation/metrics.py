"""
Evaluation metrics for RAG systems.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import time
from rouge_score import rouge_scorer
from bert_score import score as bert_score


def compute_precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int
) -> float:
    """
    Compute Precision@K.

    Args:
        retrieved_ids: List of retrieved document IDs (ordered by relevance)
        relevant_ids: List of relevant document IDs
        k: Number of top results to consider

    Returns:
        Precision@K score
    """
    if k == 0:
        return 0.0

    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    retrieved_relevant = sum(1 for doc_id in top_k if doc_id in relevant_set)

    return retrieved_relevant / k


def compute_recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int
) -> float:
    """
    Compute Recall@K.

    Args:
        retrieved_ids: List of retrieved document IDs (ordered by relevance)
        relevant_ids: List of relevant document IDs
        k: Number of top results to consider

    Returns:
        Recall@K score
    """
    if not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    retrieved_relevant = sum(1 for doc_id in top_k if doc_id in relevant_set)

    return retrieved_relevant / len(relevant_ids)


def compute_mrr(
    retrieved_ids: List[str],
    relevant_ids: List[str]
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).

    Args:
        retrieved_ids: List of retrieved document IDs (ordered by relevance)
        relevant_ids: List of relevant document IDs

    Returns:
        MRR score
    """
    if not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank

    return 0.0


def compute_answer_correctness(
    predicted_answer: str,
    ground_truth_answer: Any,
    metric: str = "rouge_l"
) -> Dict[str, float]:
    """
    Compute answer correctness using various metrics.

    Args:
        predicted_answer: The predicted answer text
        ground_truth_answer: Ground truth answer(s)
        metric: Metric to use ('rouge_l', 'rouge_1', 'rouge_2', 'bert_score', 'exact_match')

    Returns:
        Dictionary of metric scores
    """
    # Normalize ground truth answer
    if isinstance(ground_truth_answer, list):
        gt_text = " ".join(str(a) for a in ground_truth_answer)
    else:
        gt_text = str(ground_truth_answer) if ground_truth_answer else ""

    pred_text = str(predicted_answer) if predicted_answer else ""

    scores = {}

    # ROUGE scores
    if metric in ["rouge_l", "rouge_1", "rouge_2", "all"]:
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True
        )
        rouge_scores = scorer.score(gt_text, pred_text)

        if metric in ["rouge_1", "all"]:
            scores["rouge_1"] = rouge_scores["rouge1"].fmeasure
        if metric in ["rouge_2", "all"]:
            scores["rouge_2"] = rouge_scores["rouge2"].fmeasure
        if metric in ["rouge_l", "all"]:
            scores["rouge_l"] = rouge_scores["rougeL"].fmeasure

    # BERTScore
    if metric in ["bert_score", "all"]:
        try:
            P, R, F1 = bert_score([pred_text], [gt_text], lang="en", verbose=False)
            scores["bert_score_precision"] = float(P[0])
            scores["bert_score_recall"] = float(R[0])
            scores["bert_score_f1"] = float(F1[0])
        except Exception as e:
            # Fallback if BERTScore fails
            scores["bert_score_precision"] = 0.0
            scores["bert_score_recall"] = 0.0
            scores["bert_score_f1"] = 0.0

    # Exact match
    if metric in ["exact_match", "all"]:
        pred_normalized = pred_text.lower().strip()
        gt_normalized = gt_text.lower().strip()
        scores["exact_match"] = 1.0 if pred_normalized == gt_normalized else 0.0

    return scores


def compute_faithfulness(
    generated_answer: str,
    source_chunks: List[str],
    metric: str = "rouge_l"
) -> Dict[str, float]:
    """
    Compute faithfulness of generated answer to source material.

    Args:
        generated_answer: The generated answer text
        source_chunks: List of source chunk texts used for generation
        metric: Metric to use for comparison

    Returns:
        Dictionary of faithfulness scores
    """
    if not source_chunks:
        return {"faithfulness": 0.0}

    # Combine source chunks
    source_text = "\n".join(source_chunks)

    # Compute similarity between answer and source
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )
    rouge_scores = scorer.score(source_text, generated_answer)

    return {
        "faithfulness_rouge_1": rouge_scores["rouge1"].fmeasure,
        "faithfulness_rouge_2": rouge_scores["rouge2"].fmeasure,
        "faithfulness_rouge_l": rouge_scores["rougeL"].fmeasure,
        "faithfulness": rouge_scores["rougeL"].fmeasure  # Use ROUGE-L as main metric
    }


def compute_latency_metrics(
    timings: List[float]
) -> Dict[str, float]:
    """
    Compute latency and throughput metrics.

    Args:
        timings: List of timing measurements in seconds

    Returns:
        Dictionary of latency metrics
    """
    if not timings:
        return {
            "mean_latency": 0.0,
            "median_latency": 0.0,
            "p95_latency": 0.0,
            "p99_latency": 0.0,
            "throughput": 0.0
        }

    timings_array = np.array(timings)
    sorted_timings = np.sort(timings_array)

    return {
        "mean_latency": float(np.mean(timings_array)),
        "median_latency": float(np.median(timings_array)),
        "p95_latency": float(np.percentile(sorted_timings, 95)),
        "p99_latency": float(np.percentile(sorted_timings, 99)),
        "min_latency": float(np.min(timings_array)),
        "max_latency": float(np.max(timings_array)),
        "throughput": float(1.0 / np.mean(timings_array))  # Queries per second
    }


def evaluate_retrieval(
    retrieved_results: List[List[Dict[str, Any]]],
    ground_truth_relevant: List[List[str]],
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Evaluate retrieval performance.

    Args:
        retrieved_results: List of retrieval results for each query
        ground_truth_relevant: List of relevant document IDs for each query
        k_values: List of K values to evaluate

    Returns:
        Dictionary of retrieval metrics
    """
    metrics = {}

    for k in k_values:
        precisions = []
        recalls = []
        mrrs = []

        for retrieved, relevant in zip(retrieved_results, ground_truth_relevant):
            retrieved_ids = [r["id"] for r in retrieved]

            precisions.append(compute_precision_at_k(retrieved_ids, relevant, k))
            recalls.append(compute_recall_at_k(retrieved_ids, relevant, k))
            mrrs.append(compute_mrr(retrieved_ids, relevant))

        metrics[f"precision@{k}"] = np.mean(precisions)
        metrics[f"recall@{k}"] = np.mean(recalls)

    metrics["mrr"] = np.mean(mrrs)

    return metrics


def evaluate_rag_system(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    retrieval_results: Optional[List[List[Dict[str, Any]]]] = None,
    timings: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of a RAG system.

    Args:
        predictions: List of prediction dictionaries with 'answer' and optionally 'source_chunks'
        ground_truth: List of ground truth dictionaries with 'answer' and optionally 'relevant_ids'
        retrieval_results: Optional retrieval results for each query
        timings: Optional timing measurements

    Returns:
        Dictionary of all evaluation metrics
    """
    results = {}

    # Answer correctness
    correctness_scores = []
    for pred, gt in zip(predictions, ground_truth):
        answer_scores = compute_answer_correctness(
            pred.get("answer", ""),
            gt.get("answer", ""),
            metric="all"
        )
        correctness_scores.append(answer_scores)

    # Aggregate correctness metrics
    for metric in ["rouge_1", "rouge_2", "rouge_l", "exact_match"]:
        if metric in correctness_scores[0]:
            results[f"answer_{metric}"] = np.mean([s[metric] for s in correctness_scores])

    for metric in ["bert_score_precision", "bert_score_recall", "bert_score_f1"]:
        if metric in correctness_scores[0]:
            results[f"answer_{metric}"] = np.mean([s[metric] for s in correctness_scores])

    # Faithfulness
    faithfulness_scores = []
    for pred in predictions:
        source_chunks = pred.get("source_chunks", [])
        if source_chunks:
            faith_scores = compute_faithfulness(
                pred.get("answer", ""),
                source_chunks
            )
            faithfulness_scores.append(faith_scores)

    if faithfulness_scores:
        for metric in faithfulness_scores[0].keys():
            results[f"faithfulness_{metric}"] = np.mean([s[metric] for s in faithfulness_scores])

    # Retrieval metrics
    if retrieval_results:
        gt_relevant = [gt.get("relevant_ids", []) for gt in ground_truth]
        retrieval_metrics = evaluate_retrieval(retrieval_results, gt_relevant)
        results.update(retrieval_metrics)

    # Latency metrics
    if timings:
        latency_metrics = compute_latency_metrics(timings)
        results.update(latency_metrics)

    return results

