"""Leakage-safe k-fold evaluation for hyperedge link prediction.

This harness evaluates model stability by averaging AUC across folds instead of
relying on a single train/test split.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

# Allow running the script directly via path while importing workspace modules.
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from link_prediction.edge_logreg_prediction import edge_feature
from link_prediction.hgnn_prediction import (
    aggregate_hyperedge_embeddings,
    build_membership_map,
    filter_hyperedge_index,
    get_node_index_maps,
    load_hypergraph,
    make_encoder,
    make_negative_membership_typed,
    set_seed,
)
from utils.metrics import evaluate_link_prediction


def setup_run_dir(base_dir: str = "link_pred_output/KFOLD") -> Path:
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


def _core_members(
    edge_id: int,
    memberships: dict[int, list[int]],
    node_types_by_idx: dict[int, str],
) -> set[int]:
    protected_types = {
        "AlgorithmicComponent",
        "PromptComponent",
        "DataFlow",
    }
    return {
        n
        for n in memberships.get(edge_id, [])
        if node_types_by_idx.get(n, "") in protected_types
    }


def build_leakage_safe_folds(
    memberships: dict[int, list[int]],
    node_types_by_idx: dict[int, str],
    k_folds: int,
    seed: int,
) -> list[dict]:
    if k_folds < 2:
        raise ValueError("k_folds must be >= 2")

    edge_ids = sorted(memberships.keys())
    if k_folds > len(edge_ids):
        raise ValueError(f"k_folds={k_folds} exceeds number of hyperedges={len(edge_ids)}")

    rng = np.random.default_rng(seed)
    shuffled = edge_ids.copy()
    rng.shuffle(shuffled)

    fold_sizes = [len(shuffled) // k_folds] * k_folds
    for i in range(len(shuffled) % k_folds):
        fold_sizes[i] += 1

    raw_folds: list[list[int]] = []
    cursor = 0
    for fold_size in fold_sizes:
        raw_folds.append(sorted(shuffled[cursor: cursor + fold_size]))
        cursor += fold_size

    fold_plans: list[dict] = []
    for fold_idx in range(k_folds):
        test_edges = raw_folds[fold_idx]
        test_core_nodes: set[int] = set()
        for edge in test_edges:
            test_core_nodes.update(_core_members(edge, memberships, node_types_by_idx))

        candidate_train = [e for i, fold in enumerate(raw_folds) if i != fold_idx for e in fold]
        train_edges = [
            e
            for e in candidate_train
            if _core_members(e, memberships, node_types_by_idx).isdisjoint(test_core_nodes)
        ]

        train_core_nodes: set[int] = set()
        for edge in train_edges:
            train_core_nodes.update(_core_members(edge, memberships, node_types_by_idx))

        overlap = train_core_nodes.intersection(test_core_nodes)
        fold_plans.append(
            {
                "fold_id": fold_idx,
                "train_edges": sorted(train_edges),
                "test_edges": sorted(test_edges),
                "leakage_guard": {
                    "num_train_hyperedges": len(train_edges),
                    "num_test_hyperedges": len(test_edges),
                    "train_core_nodes": len(train_core_nodes),
                    "test_core_nodes": len(test_core_nodes),
                    "overlap_core_nodes": len(overlap),
                },
            }
        )

    return fold_plans


def evaluate_hgnn_fold(
    node_features: torch.Tensor,
    hyperedge_index: torch.Tensor,
    memberships: dict[int, list[int]],
    node_types_by_idx: dict[int, str],
    train_edges: list[int],
    test_edges: list[int],
    fold_seed: int,
    epochs: int,
    lr: float,
    hidden_dim: int,
    out_dim: int,
    dropout: float,
    architecture: str,
    edge_pooling: str,
    num_negatives: int,
    weight_decay: float,
) -> tuple[dict, list[float], list[float]]:
    set_seed(fold_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = node_features.to(device)
    hi = hyperedge_index.to(device)

    train_set = set(train_edges)
    train_hi = filter_hyperedge_index(hi.cpu(), train_set).to(device)

    train_memberships = {e: memberships[e] for e in train_edges if e in memberships}
    test_memberships = {e: memberships[e] for e in test_edges if e in memberships}

    train_visible_nodes = sorted({n for members in train_memberships.values() for n in members})
    if not train_visible_nodes:
        raise RuntimeError("No visible train nodes after leakage-safe filtering.")

    train_neg = make_negative_membership_typed(
        memberships=train_memberships,
        edge_indices=list(train_memberships.keys()),
        candidate_nodes=train_visible_nodes,
        node_types_by_idx=node_types_by_idx,
        num_negatives=num_negatives,
        seed=fold_seed,
    )

    encoder = make_encoder(
        architecture=architecture,
        in_dim=x.shape[1],
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        dropout=dropout,
    ).to(device)

    edge_emb_dim = out_dim if edge_pooling == "mean" else out_dim * 3
    scorer = nn.Sequential(
        nn.Linear(edge_emb_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    ).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(scorer.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    train_edge_indices = sorted(train_memberships.keys())
    train_neg_indices = sorted(train_neg.keys())

    for _ in range(epochs):
        encoder.train()
        scorer.train()
        optimizer.zero_grad()

        z = encoder(x, train_hi)
        pos_vec = aggregate_hyperedge_embeddings(z, train_memberships, train_edge_indices, mode=edge_pooling)
        neg_vec = aggregate_hyperedge_embeddings(z, train_neg, train_neg_indices, mode=edge_pooling)

        pos_logits = scorer(pos_vec).squeeze(-1)
        neg_logits = scorer(neg_vec).squeeze(-1)

        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat(
            [
                torch.ones(pos_logits.shape[0], device=device),
                torch.zeros(neg_logits.shape[0], device=device),
            ]
        )

        bce = F.binary_cross_entropy_with_logits(logits, labels)
        if pos_logits.shape[0] > 0 and neg_logits.shape[0] > 0:
            sampled_neg = neg_logits[: pos_logits.shape[0]]
            rank_loss = F.relu(1.0 - pos_logits + sampled_neg).mean()
        else:
            rank_loss = torch.tensor(0.0, device=device)

        loss = bce + 0.2 * rank_loss
        loss.backward()
        nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(scorer.parameters()), max_norm=1.0)
        optimizer.step()

    test_neg = make_negative_membership_typed(
        memberships=test_memberships,
        edge_indices=list(test_memberships.keys()),
        candidate_nodes=train_visible_nodes,
        node_types_by_idx=node_types_by_idx,
        num_negatives=1,
        seed=fold_seed + 999,
    )

    with torch.no_grad():
        encoder.eval()
        scorer.eval()
        z_all = encoder(x, train_hi)
        test_pos_vec = aggregate_hyperedge_embeddings(z_all, test_memberships, sorted(test_memberships.keys()), mode=edge_pooling)
        test_neg_vec = aggregate_hyperedge_embeddings(z_all, test_neg, sorted(test_neg.keys()), mode=edge_pooling)

        pos_scores = torch.sigmoid(scorer(test_pos_vec).squeeze(-1)).detach().cpu()
        neg_scores = torch.sigmoid(scorer(test_neg_vec).squeeze(-1)).detach().cpu()

    metrics = evaluate_link_prediction(pos_scores, neg_scores, k_list=[1, 3, 5, 10])
    return metrics, pos_scores.tolist(), neg_scores.tolist()


def evaluate_edge_logreg_fold(
    node_features: np.ndarray,
    memberships: dict[int, list[int]],
    node_types_by_idx: dict[int, str],
    train_edges: list[int],
    test_edges: list[int],
    fold_seed: int,
    feature_mode: str,
    c_value: float,
) -> tuple[dict, list[float], list[float]]:
    train_memberships = {e: memberships[e] for e in train_edges if e in memberships}
    test_memberships = {e: memberships[e] for e in test_edges if e in memberships}
    train_visible_nodes = sorted({n for members in train_memberships.values() for n in members})

    train_neg = make_negative_membership_typed(
        memberships=train_memberships,
        edge_indices=list(train_memberships.keys()),
        candidate_nodes=train_visible_nodes,
        node_types_by_idx=node_types_by_idx,
        num_negatives=1,
        seed=fold_seed,
    )
    test_neg = make_negative_membership_typed(
        memberships=test_memberships,
        edge_indices=list(test_memberships.keys()),
        candidate_nodes=train_visible_nodes,
        node_types_by_idx=node_types_by_idx,
        num_negatives=1,
        seed=fold_seed + 999,
    )

    x_train = np.stack(
        [edge_feature(v, node_features, mode=feature_mode) for v in train_memberships.values()]
        + [edge_feature(v, node_features, mode=feature_mode) for v in train_neg.values()]
    )
    y_train = np.array([1] * len(train_memberships) + [0] * len(train_neg), dtype=np.int64)

    model = LogisticRegression(C=c_value, max_iter=2000, solver="lbfgs")
    model.fit(x_train, y_train)

    x_test_pos = np.stack([edge_feature(v, node_features, mode=feature_mode) for v in test_memberships.values()])
    x_test_neg = np.stack([edge_feature(v, node_features, mode=feature_mode) for v in test_neg.values()])

    pos_scores = torch.tensor(model.predict_proba(x_test_pos)[:, 1], dtype=torch.float32)
    neg_scores = torch.tensor(model.predict_proba(x_test_neg)[:, 1], dtype=torch.float32)
    metrics = evaluate_link_prediction(pos_scores, neg_scores, k_list=[1, 3, 5, 10])
    return metrics, pos_scores.tolist(), neg_scores.tolist()


def aggregate_fold_metrics(fold_reports: list[dict]) -> dict:
    valid = [f for f in fold_reports if f.get("status") == "ok"]
    if not valid:
        return {
            "num_valid_folds": 0,
            "AUC_mean": 0.0,
            "AUC_std": 0.0,
            "AUC_min": 0.0,
            "AUC_max": 0.0,
            "AP_mean": 0.0,
            "MRR_mean": 0.0,
            "PairwiseAcc_mean": 0.0,
        }

    aucs = [f["metrics"]["AUC"] for f in valid]
    aps = [f["metrics"]["AP"] for f in valid]
    mrrs = [f["metrics"]["MRR"] for f in valid]
    pairwise = [f["metrics"]["PairwiseAcc"] for f in valid]

    return {
        "num_valid_folds": len(valid),
        "AUC_mean": float(np.mean(aucs)),
        "AUC_std": float(np.std(aucs)),
        "AUC_min": float(np.min(aucs)),
        "AUC_max": float(np.max(aucs)),
        "AP_mean": float(np.mean(aps)),
        "MRR_mean": float(np.mean(mrrs)),
        "PairwiseAcc_mean": float(np.mean(pairwise)),
    }


def run_kfold(
    model_type: str,
    graph_path: str,
    k_folds: int,
    seed: int,
    epochs: int,
    lr: float,
    hidden_dim: int,
    out_dim: int,
    dropout: float,
    architecture: str,
    edge_pooling: str,
    num_negatives: int,
    weight_decay: float,
    feature_mode: str,
    c_value: float,
) -> dict:
    run_dir = setup_run_dir()
    setup_logging(run_dir)
    set_seed(seed)

    data = load_hypergraph(graph_path)
    node_features: torch.Tensor = data["node_features"]
    hyperedge_index: torch.Tensor = data["hyperedge_index"]
    _, _, node_types_by_idx = get_node_index_maps(data["node_metadata"], data["node_to_idx"])
    memberships = build_membership_map(hyperedge_index.cpu())

    fold_plans = build_leakage_safe_folds(
        memberships=memberships,
        node_types_by_idx=node_types_by_idx,
        k_folds=k_folds,
        seed=seed,
    )

    fold_reports: list[dict] = []

    for fold in fold_plans:
        fold_id = int(fold["fold_id"])
        train_edges = fold["train_edges"]
        test_edges = fold["test_edges"]
        leak = fold["leakage_guard"]

        logging.info(
            "Fold %d | train=%d test=%d overlap=%d",
            fold_id,
            leak["num_train_hyperedges"],
            leak["num_test_hyperedges"],
            leak["overlap_core_nodes"],
        )

        if not train_edges:
            fold_reports.append(
                {
                    "fold_id": fold_id,
                    "status": "skipped",
                    "reason": "no_train_hyperedges_after_leakage_filter",
                    "leakage_guard": leak,
                }
            )
            continue

        try:
            fold_seed = seed + fold_id
            if model_type == "hgnn":
                metrics, pos_scores, neg_scores = evaluate_hgnn_fold(
                    node_features=node_features,
                    hyperedge_index=hyperedge_index,
                    memberships=memberships,
                    node_types_by_idx=node_types_by_idx,
                    train_edges=train_edges,
                    test_edges=test_edges,
                    fold_seed=fold_seed,
                    epochs=epochs,
                    lr=lr,
                    hidden_dim=hidden_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    architecture=architecture,
                    edge_pooling=edge_pooling,
                    num_negatives=num_negatives,
                    weight_decay=weight_decay,
                )
            elif model_type == "edge_logreg":
                metrics, pos_scores, neg_scores = evaluate_edge_logreg_fold(
                    node_features=node_features.detach().cpu().numpy(),
                    memberships=memberships,
                    node_types_by_idx=node_types_by_idx,
                    train_edges=train_edges,
                    test_edges=test_edges,
                    fold_seed=fold_seed,
                    feature_mode=feature_mode,
                    c_value=c_value,
                )
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")

            fold_reports.append(
                {
                    "fold_id": fold_id,
                    "status": "ok",
                    "leakage_guard": leak,
                    "metrics": metrics,
                    "num_positive_predictions": len(pos_scores),
                    "num_negative_predictions": len(neg_scores),
                    "positive_scores": pos_scores,
                    "negative_scores": neg_scores,
                }
            )
            logging.info(
                "Fold %d complete | AUC=%.4f AP=%.4f",
                fold_id,
                metrics["AUC"],
                metrics["AP"],
            )
        except Exception as exc:  # pragma: no cover
            fold_reports.append(
                {
                    "fold_id": fold_id,
                    "status": "failed",
                    "error": str(exc),
                    "leakage_guard": leak,
                }
            )
            logging.exception("Fold %d failed", fold_id)

    aggregate = aggregate_fold_metrics(fold_reports)

    result = {
        "run_dir": str(run_dir),
        "model_type": model_type,
        "graph_path": graph_path,
        "k_folds": k_folds,
        "seed": seed,
        "config": {
            "epochs": epochs,
            "lr": lr,
            "hidden_dim": hidden_dim,
            "out_dim": out_dim,
            "dropout": dropout,
            "architecture": architecture,
            "edge_pooling": edge_pooling,
            "num_negatives": num_negatives,
            "weight_decay": weight_decay,
            "feature_mode": feature_mode,
            "c_value": c_value,
        },
        "fold_reports": fold_reports,
        "aggregate": aggregate,
    }

    with (run_dir / "kfold_summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    logging.info(
        "K-fold complete | valid_folds=%d mean_auc=%.4f std_auc=%.4f",
        aggregate["num_valid_folds"],
        aggregate["AUC_mean"],
        aggregate["AUC_std"],
    )

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Leakage-safe k-fold evaluation harness")
    parser.add_argument("--model", type=str, default="edge_logreg", choices=["edge_logreg", "hgnn"])
    parser.add_argument("--graph-path", type=str, default="data/hypergraph_output/hypergraph_latest.pkl")
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=21)

    # HGNN options
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=9e-4)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--out-dim", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--architecture", type=str, default="hgnn")
    parser.add_argument("--edge-pooling", type=str, default="mean", choices=["mean", "mean_max_std"])
    parser.add_argument("--num-negatives", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    # Edge-logreg options
    parser.add_argument("--feature-mode", type=str, default="mean_max_std", choices=["mean", "mean_max_std"])
    parser.add_argument("--c-value", type=float, default=0.3)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    os.makedirs("link_pred_output/KFOLD", exist_ok=True)

    result = run_kfold(
        model_type=args.model,
        graph_path=args.graph_path,
        k_folds=args.k_folds,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        dropout=args.dropout,
        architecture=args.architecture,
        edge_pooling=args.edge_pooling,
        num_negatives=args.num_negatives,
        weight_decay=args.weight_decay,
        feature_mode=args.feature_mode,
        c_value=args.c_value,
    )

    agg = result["aggregate"]
    print("RUN_DIR", result["run_dir"])
    print("VALID_FOLDS", agg["num_valid_folds"])
    print("AUC_MEAN", agg["AUC_mean"])
    print("AUC_STD", agg["AUC_std"])
    print("AUC_MAX", agg["AUC_max"])


if __name__ == "__main__":
    main()
