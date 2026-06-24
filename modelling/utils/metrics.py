"""
Link Prediction Evaluation Metrics
====================================
Centralized evaluation script for all link prediction models.
Computes Hits@K, MRR, AUC-ROC, and Average Precision.

Usage:
    from utils.metrics import evaluate_link_prediction
    results = evaluate_link_prediction(pos_scores, neg_scores)
"""

import torch
import numpy as np
import logging
import json
import os
from sklearn.metrics import roc_auc_score, average_precision_score


def compute_hits_at_k(pos_scores: torch.Tensor, neg_scores: torch.Tensor, k: int) -> float:
    """
    Hits@K: For each positive edge, count how many have a score in the
    top-K when ranked against all negative edges.

    For each positive sample, we rank it against ALL negatives. If the
    positive ranks within the top K, it's a 'hit'.
    """
    num_pos = pos_scores.shape[0]
    if num_pos == 0:
        return 0.0

    hits = 0
    for i in range(num_pos):
        # Number of negatives that score higher than this positive
        rank = (neg_scores >= pos_scores[i]).sum().item() + 1  # 1-indexed
        if rank <= k:
            hits += 1

    return hits / num_pos


def compute_mrr(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> float:
    """
    Mean Reciprocal Rank (MRR): For each positive edge, compute its rank
    against all negatives and take the mean of 1/rank.
    """
    num_pos = pos_scores.shape[0]
    if num_pos == 0:
        return 0.0

    reciprocal_ranks = []
    for i in range(num_pos):
        rank = (neg_scores >= pos_scores[i]).sum().item() + 1
        reciprocal_ranks.append(1.0 / rank)

    return float(np.mean(reciprocal_ranks))


def compute_auc(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> float:
    """
    AUC-ROC: Area Under the Receiver Operating Characteristic Curve.
    Measures the probability that a random positive edge is scored
    higher than a random negative edge.
    """
    labels = torch.cat([
        torch.ones(pos_scores.shape[0]),
        torch.zeros(neg_scores.shape[0]),
    ]).numpy()

    scores = torch.cat([pos_scores, neg_scores]).numpy()

    if len(np.unique(labels)) < 2:
        logging.warning("AUC undefined: only one class present. Returning 0.0.")
        return 0.0

    return float(roc_auc_score(labels, scores))


def compute_ap(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> float:
    """
    Average Precision (AP): Summarizes the Precision-Recall curve.
    Especially useful under class imbalance (many more non-edges than edges).
    """
    labels = torch.cat([
        torch.ones(pos_scores.shape[0]),
        torch.zeros(neg_scores.shape[0]),
    ]).numpy()

    scores = torch.cat([pos_scores, neg_scores]).numpy()

    if len(np.unique(labels)) < 2:
        logging.warning("AP undefined: only one class present. Returning 0.0.")
        return 0.0

    return float(average_precision_score(labels, scores))


def evaluate_link_prediction(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    k_list: list = None,
) -> dict:
    """
    Run the full evaluation suite for link prediction.

    Args:
        pos_scores: Tensor of scores for positive (true) edges.
        neg_scores: Tensor of scores for negative (false) edges.
        k_list: List of K values for Hits@K. Defaults to [1, 3, 10].

    Returns:
        Dictionary of metric_name → value.
    """
    if k_list is None:
        k_list = [1, 3, 10]

    # Detach and move to CPU
    pos_scores = pos_scores.detach().cpu()
    neg_scores = neg_scores.detach().cpu()

    results = {}

    # Hits@K
    for k in k_list:
        results[f"Hits@{k}"] = compute_hits_at_k(pos_scores, neg_scores, k)

    # MRR
    results["MRR"] = compute_mrr(pos_scores, neg_scores)

    # AUC-ROC
    results["AUC"] = compute_auc(pos_scores, neg_scores)

    # Average Precision
    results["AP"] = compute_ap(pos_scores, neg_scores)

    # Pairwise accuracy (kept for backward compat with existing logs)
    results["PairwiseAcc"] = (pos_scores > neg_scores).float().mean().item()

    return results


def log_metrics(results: dict, prefix: str = "Test") -> None:
    """Log all metrics in a readable format."""
    logging.info(f"{'-' * 50}")
    logging.info(f"  {prefix} Evaluation Metrics")
    logging.info(f"{'-' * 50}")
    for name, value in results.items():
        logging.info(f"  {name:>15s}: {value:.4f}")
    logging.info(f"{'-' * 50}")


def save_metrics(results: dict, run_dir: str, model_name: str = "model") -> str:
    """Save metrics to a JSON file in the run directory."""
    path = os.path.join(run_dir, f"metrics_{model_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Metrics saved to {path}")
    return path
