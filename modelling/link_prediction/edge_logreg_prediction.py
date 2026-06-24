"""Low-capacity hyperedge link prediction with logistic regression.

This is intended for tiny hypergraph regimes where deep message passing can overfit.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

# Allow running the script directly via path while importing workspace modules.
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from link_prediction.hgnn_prediction import (
    build_membership_map,
    discover_candidates,
    get_node_index_maps,
    load_hypergraph,
    make_negative_membership_typed,
    set_seed,
    signature,
    split_hyperedges_strict_member_disjoint,
)
from utils.metrics import evaluate_link_prediction, log_metrics, save_metrics


class SklearnEdgeScorer:
    """Adapter to score edge vectors with a sklearn classifier in torch-style API."""

    def __init__(self, classifier: LogisticRegression) -> None:
        self.classifier = classifier

    def eval(self) -> "SklearnEdgeScorer":
        return self

    def __call__(self, edge_embeddings: torch.Tensor) -> torch.Tensor:
        arr = edge_embeddings.detach().cpu().numpy()
        probs = self.classifier.predict_proba(arr)[:, 1]
        probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
        logits = np.log(probs / (1.0 - probs))
        return torch.tensor(logits, dtype=torch.float32, device=edge_embeddings.device)


def setup_run_dir(base_dir: str = "link_pred_output/EDGE_LOGREG") -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logging(run_dir: Path) -> None:
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(run_dir / "run.log"),
            logging.StreamHandler(),
        ],
    )


def edge_feature(
    members: list[int],
    node_features: np.ndarray,
    mode: str = "mean_max_std",
) -> np.ndarray:
    unique_members = sorted(set(members))
    if not unique_members:
        feat_dim = node_features.shape[1]
        if mode == "mean":
            return np.zeros(feat_dim, dtype=np.float32)
        return np.zeros(feat_dim * 3, dtype=np.float32)

    arr = node_features[np.array(unique_members, dtype=np.int64)]
    mean_vec = arr.mean(axis=0)

    if mode == "mean":
        return mean_vec.astype(np.float32)

    if mode == "mean_max_std":
        max_vec = arr.max(axis=0)
        std_vec = arr.std(axis=0)
        return np.concatenate([mean_vec, max_vec, std_vec], axis=0).astype(np.float32)

    raise ValueError(f"Unsupported feature mode: {mode}")


def train_and_discover(
    graph_path: str = "data/hypergraph_output/hypergraph_latest.pkl",
    test_ratio: float = 0.1,
    feature_mode: str = "mean_max_std",
    c_value: float = 0.3,
    seed: int = 21,
) -> dict:
    set_seed(seed)
    run_dir = setup_run_dir()
    setup_logging(run_dir)

    data = load_hypergraph(graph_path)
    node_features_tensor: torch.Tensor = data["node_features"]
    node_features = node_features_tensor.detach().cpu().numpy()
    hyperedge_index: torch.Tensor = data["hyperedge_index"]

    by_type, metadata_by_idx, node_types_by_idx = get_node_index_maps(
        data["node_metadata"], data["node_to_idx"]
    )

    memberships = build_membership_map(hyperedge_index)

    train_edges, test_edges, leakage_report = split_hyperedges_strict_member_disjoint(
        memberships=memberships,
        node_types_by_idx=node_types_by_idx,
        test_ratio=test_ratio,
        seed=seed,
    )

    logging.info(
        "Strict split complete | train=%d test=%d overlap_core_nodes=%d",
        leakage_report["num_train_hyperedges"],
        leakage_report["num_test_hyperedges"],
        leakage_report["overlap_core_nodes"],
    )

    train_memberships = {e: memberships[e] for e in train_edges if e in memberships}
    test_memberships = {e: memberships[e] for e in test_edges if e in memberships}

    train_visible_nodes = sorted({n for m in train_memberships.values() for n in m})

    train_neg = make_negative_membership_typed(
        memberships=train_memberships,
        edge_indices=list(train_memberships.keys()),
        candidate_nodes=train_visible_nodes,
        node_types_by_idx=node_types_by_idx,
        num_negatives=1,
        seed=seed,
    )

    test_neg = make_negative_membership_typed(
        memberships=test_memberships,
        edge_indices=list(test_memberships.keys()),
        candidate_nodes=train_visible_nodes,
        node_types_by_idx=node_types_by_idx,
        num_negatives=1,
        seed=seed + 999,
    )

    x_train = np.stack(
        [edge_feature(v, node_features, mode=feature_mode) for v in train_memberships.values()]
        + [edge_feature(v, node_features, mode=feature_mode) for v in train_neg.values()]
    )
    y_train = np.array([1] * len(train_memberships) + [0] * len(train_neg), dtype=np.int64)

    model = LogisticRegression(
        C=c_value,
        max_iter=2000,
        solver="lbfgs",
    )
    model.fit(x_train, y_train)

    x_test_pos = np.stack([edge_feature(v, node_features, mode=feature_mode) for v in test_memberships.values()])
    x_test_neg = np.stack([edge_feature(v, node_features, mode=feature_mode) for v in test_neg.values()])

    pos_scores = torch.tensor(model.predict_proba(x_test_pos)[:, 1], dtype=torch.float32)
    neg_scores = torch.tensor(model.predict_proba(x_test_neg)[:, 1], dtype=torch.float32)

    metrics = evaluate_link_prediction(pos_scores, neg_scores, k_list=[1, 3, 5, 10])
    log_metrics(metrics, prefix="EDGE_LOGREG Test")
    save_metrics(metrics, str(run_dir), model_name="edge_logreg")

    scorer = SklearnEdgeScorer(model)
    node_embeddings = torch.tensor(node_features, dtype=torch.float32)

    known_signatures = {signature(v) for v in memberships.values()}
    candidates, grouped_by_use_case = discover_candidates(
        node_embeddings=node_embeddings,
        scorer=scorer,
        by_type=by_type,
        metadata_by_idx=metadata_by_idx,
        known_signatures=known_signatures,
        top_k=100,
        max_per_task=10,
        edge_pooling=feature_mode,
    )

    output = {
        "run_dir": str(run_dir),
        "model": "EdgeLogReg(mean/max/std features)",
        "train_hyperedges": len(train_edges),
        "test_hyperedges": len(test_edges),
        "leakage_guard": leakage_report,
        "evaluation_metrics": metrics,
        "shortlists_by_use_case": grouped_by_use_case,
        "novel_techniques": candidates,
    }

    with (run_dir / "novel_techniques_edge_logreg.json").open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    with (run_dir / "hyperparameters.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "test_ratio": test_ratio,
                "feature_mode": feature_mode,
                "c_value": c_value,
                "seed": seed,
            },
            f,
            indent=2,
        )

    logging.info("Saved EDGE_LOGREG output to %s", run_dir)
    return output


if __name__ == "__main__":
    os.makedirs("link_pred_output/EDGE_LOGREG", exist_ok=True)
    train_and_discover()
