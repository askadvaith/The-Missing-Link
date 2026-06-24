"""
Multi-Seed Hypergraph Discovery Runner
======================================
Trains a hypergraph model (HGNN or EdgeLogReg) over M random seeds and aggregates
predictions via Rank-Based Fusion (RRF).

Usage:
    python link_prediction/run_multi_seed_hyper_discovery.py
    python link_prediction/run_multi_seed_hyper_discovery.py --model hgnn --seeds 42 7
    python link_prediction/run_multi_seed_hyper_discovery.py --model edge_logreg --seeds 21 123
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import pickle
import random
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

from link_prediction.edge_logreg_prediction import edge_feature, SklearnEdgeScorer
from link_prediction.hgnn_prediction import (
    aggregate_hyperedge_embeddings,
    build_membership_map,
    discover_candidates,
    filter_hyperedge_index,
    get_node_index_maps,
    load_hypergraph,
    make_encoder,
    make_negative_membership_typed,
    set_seed,
    signature,
    split_hyperedges_strict_member_disjoint,
    HyperedgeScorer,
)
from utils.metrics import evaluate_link_prediction, log_metrics, save_metrics


# Default hyperparameters
MODEL_DEFAULTS = {
    "hgnn": {
        "epochs": 200,
        "lr": 1e-3,
        "hidden_dim": 256,
        "out_dim": 128,
        "dropout": 0.2,
        "test_ratio": 0.2,
        "architecture": "hgnn",
        "edge_pooling": "mean_max_std",
        "num_negatives": 3,
        "weight_decay": 1e-4,
        "top_k": 100,
        "max_per_task": 10,
    },
    "edge_logreg": {
        "test_ratio": 0.1,
        "feature_mode": "mean_max_std",
        "c_value": 0.3,
        "top_k": 100,
        "max_per_task": 10,
    }
}


def setup_run_dir(model_type: str, base_dir: str = "link_pred_output") -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"MultiSeed_{model_type.upper()}" / f"run_{ts}"
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


def train_hgnn_one_seed(
    data: dict,
    hparams: dict,
    seed: int,
    seed_idx: int,
    run_dir: Path,
) -> tuple[dict, list[dict]]:
    set_seed(seed)
    logging.info("=" * 60)
    logging.info(f"  HGNN Seed run {seed_idx + 1}  |  seed={seed}")
    logging.info("=" * 60)

    node_features: torch.Tensor = data["node_features"]
    hyperedge_index: torch.Tensor = data["hyperedge_index"]
    by_type, metadata_by_idx, node_types_by_idx = get_node_index_maps(
        data["node_metadata"], data["node_to_idx"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = node_features.to(device)
    hyperedge_index = hyperedge_index.to(device)

    all_memberships = build_membership_map(hyperedge_index.cpu())

    train_edges, test_edges, leakage_report = split_hyperedges_strict_member_disjoint(
        memberships=all_memberships,
        node_types_by_idx=node_types_by_idx,
        test_ratio=hparams["test_ratio"],
        seed=seed,
    )
    if leakage_report["overlap_core_nodes"] != 0:
        raise RuntimeError("Leakage detected in protected node types; aborting training.")

    logging.info(
        "Strict split complete | train=%d test=%d overlap_core_nodes=%d",
        leakage_report["num_train_hyperedges"],
        leakage_report["num_test_hyperedges"],
        leakage_report["overlap_core_nodes"],
    )
    train_set = set(train_edges)
    train_hyperedge_index = filter_hyperedge_index(hyperedge_index.cpu(), train_set).to(device)

    train_memberships = {e: all_memberships[e] for e in train_edges if e in all_memberships}
    train_visible_nodes = sorted({n for members in train_memberships.values() for n in members})
    
    neg_memberships = make_negative_membership_typed(
        memberships=train_memberships,
        edge_indices=list(train_memberships.keys()),
        candidate_nodes=train_visible_nodes,
        node_types_by_idx=node_types_by_idx,
        num_negatives=hparams["num_negatives"],
        seed=seed,
    )

    encoder = make_encoder(
        architecture=hparams["architecture"],
        in_dim=x.shape[1],
        hidden_dim=hparams["hidden_dim"],
        out_dim=hparams["out_dim"],
        dropout=hparams["dropout"],
    ).to(device)

    edge_emb_dim = hparams["out_dim"] if hparams["edge_pooling"] == "mean" else hparams["out_dim"] * 3
    scorer = HyperedgeScorer(emb_dim=edge_emb_dim, hidden_dim=hparams["hidden_dim"]).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(scorer.parameters()),
        lr=hparams["lr"],
        weight_decay=hparams["weight_decay"],
    )

    train_edge_indices = sorted(train_memberships.keys())
    neg_edge_indices = sorted(neg_memberships.keys())

    for epoch in range(1, hparams["epochs"] + 1):
        encoder.train()
        scorer.train()
        optimizer.zero_grad()

        z = encoder(x, train_hyperedge_index)

        pos_vecs = aggregate_hyperedge_embeddings(
            z, train_memberships, train_edge_indices, mode=hparams["edge_pooling"]
        )
        neg_vecs = aggregate_hyperedge_embeddings(
            z, neg_memberships, neg_edge_indices, mode=hparams["edge_pooling"]
        )

        pos_logits = scorer(pos_vecs)
        neg_logits = scorer(neg_vecs)

        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat([
            torch.ones(pos_logits.shape[0], device=device),
            torch.zeros(neg_logits.shape[0], device=device),
        ])

        bce_loss = F.binary_cross_entropy_with_logits(logits, labels)

        if pos_logits.shape[0] > 0 and neg_logits.shape[0] > 0:
            sampled_neg = neg_logits[:pos_logits.shape[0]]
            rank_loss = F.relu(1.0 - pos_logits + sampled_neg).mean()
        else:
            rank_loss = torch.tensor(0.0, device=device)

        loss = bce_loss + 0.2 * rank_loss
        loss.backward()
        nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(scorer.parameters()), max_norm=1.0)
        optimizer.step()

        if epoch % 50 == 0 or epoch == 1:
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                pred = (probs >= 0.5).float()
                acc = (pred == labels).float().mean().item()
            logging.info("Seed %d | Epoch %d | loss=%.4f | train_acc=%.4f", seed, epoch, loss.item(), acc)

    # Save checkpoint
    checkpoint_path = run_dir / f"model_hgnn_seed_{seed}.pt"
    torch.save({
        "encoder_state_dict": encoder.state_dict(),
        "scorer_state_dict": scorer.state_dict(),
    }, checkpoint_path)
    logging.info(f"Saved model checkpoint to {checkpoint_path}")

    # Evaluate
    encoder.eval()
    scorer.eval()
    with torch.no_grad():
        z_all = encoder(x, train_hyperedge_index)

    test_memberships = {e: all_memberships[e] for e in test_edges if e in all_memberships}
    test_neg_memberships = make_negative_membership_typed(
        memberships=test_memberships,
        edge_indices=list(test_memberships.keys()),
        candidate_nodes=sorted({n for m in test_memberships.values() for n in m}),
        node_types_by_idx=node_types_by_idx,
        num_negatives=1,
        seed=seed + 1337,
    )

    test_edge_indices = sorted(test_memberships.keys())
    test_neg_indices = sorted(test_neg_memberships.keys())

    if test_edge_indices and test_neg_indices:
        with torch.no_grad():
            test_pos_vecs = aggregate_hyperedge_embeddings(
                z_all, test_memberships, test_edge_indices, mode=hparams["edge_pooling"]
            )
            test_neg_vecs = aggregate_hyperedge_embeddings(
                z_all, test_neg_memberships, test_neg_indices, mode=hparams["edge_pooling"]
            )
            test_pos_scores = torch.sigmoid(scorer(test_pos_vecs)).detach().cpu()
            test_neg_scores = torch.sigmoid(scorer(test_neg_vecs)).detach().cpu()
        metrics = evaluate_link_prediction(test_pos_scores, test_neg_scores, k_list=[1, 3, 5, 10])
    else:
        metrics = {
            "Hits@1": 0.0, "Hits@3": 0.0, "Hits@5": 0.0, "Hits@10": 0.0,
            "MRR": 0.0, "AUC": 0.0, "AP": 0.0, "PairwiseAcc": 0.0
        }

    log_metrics(metrics, prefix=f"HGNN Seed {seed}")

    # Discover candidates
    known_signatures = {signature(v) for v in all_memberships.values()}
    candidates, _ = discover_candidates(
        node_embeddings=z_all,
        scorer=scorer,
        by_type=by_type,
        metadata_by_idx=metadata_by_idx,
        known_signatures=known_signatures,
        top_k=hparams["top_k"],
        max_per_task=hparams["max_per_task"],
        edge_pooling=hparams["edge_pooling"],
    )

    return metrics, candidates


def train_edge_logreg_one_seed(
    data: dict,
    hparams: dict,
    seed: int,
    seed_idx: int,
    run_dir: Path,
) -> tuple[dict, list[dict]]:
    set_seed(seed)
    logging.info("=" * 60)
    logging.info(f"  EdgeLogReg Seed run {seed_idx + 1}  |  seed={seed}")
    logging.info("=" * 60)

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
        test_ratio=hparams["test_ratio"],
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
        [edge_feature(v, node_features, mode=hparams["feature_mode"]) for v in train_memberships.values()]
        + [edge_feature(v, node_features, mode=hparams["feature_mode"]) for v in train_neg.values()]
    )
    y_train = np.array([1] * len(train_memberships) + [0] * len(train_neg), dtype=np.int64)

    model = LogisticRegression(
        C=hparams["c_value"],
        max_iter=2000,
        solver="lbfgs",
    )
    model.fit(x_train, y_train)

    # Save checkpoint
    checkpoint_path = run_dir / f"model_edge_logreg_seed_{seed}.pkl"
    with open(checkpoint_path, "wb") as f:
        pickle.dump(model, f)
    logging.info(f"Saved model checkpoint to {checkpoint_path}")

    # Evaluate
    x_test_pos = np.stack([edge_feature(v, node_features, mode=hparams["feature_mode"]) for v in test_memberships.values()])
    x_test_neg = np.stack([edge_feature(v, node_features, mode=hparams["feature_mode"]) for v in test_neg.values()])

    pos_scores = torch.tensor(model.predict_proba(x_test_pos)[:, 1], dtype=torch.float32)
    neg_scores = torch.tensor(model.predict_proba(x_test_neg)[:, 1], dtype=torch.float32)

    metrics = evaluate_link_prediction(pos_scores, neg_scores, k_list=[1, 3, 5, 10])
    log_metrics(metrics, prefix=f"EDGE_LOGREG Seed {seed}")

    # Discover candidates
    scorer = SklearnEdgeScorer(model)
    node_embeddings = torch.tensor(node_features, dtype=torch.float32)
    known_signatures = {signature(v) for v in memberships.values()}

    candidates, _ = discover_candidates(
        node_embeddings=node_embeddings,
        scorer=scorer,
        by_type=by_type,
        metadata_by_idx=metadata_by_idx,
        known_signatures=known_signatures,
        top_k=hparams["top_k"],
        max_per_task=hparams["max_per_task"],
        edge_pooling=hparams["feature_mode"],
    )

    return metrics, candidates


def aggregate_scores(candidates_by_seed: list[list[dict]], k: int = 60) -> tuple[list[dict], list[dict]]:
    """
    Perform Reciprocal Rank Fusion (RRF) on the candidates generated across M seed runs.
    
    For each seed, we have a list of candidate blueprints (which are identical in keys/identities
    since they are generated with a fixed seed inside discover_candidates, but have different scores).
    We group candidates by task, rank them within each task, and compute their ensembled RRF scores.
    The ensembled score is normalized so the maximum possible RRF score is 1.0.
    
    Returns:
        flat_ensembled_candidates: list of ensembled candidate blueprints, sorted by ensembled score descending.
        grouped_by_use_case: list of ensembled shortlists grouped by task.
    """
    sig_to_metadata = {}
    sig_to_scores = {}
    
    num_seeds = len(candidates_by_seed)
    
    for seed_idx, candidates in enumerate(candidates_by_seed):
        for c in candidates:
            sig = tuple(sorted(c["member_node_ids"]))
            if sig not in sig_to_metadata:
                sig_to_metadata[sig] = {
                    "task": c["task"],
                    "shortlisted_components": c["shortlisted_components"],
                    "supporting_capabilities": c["supporting_capabilities"],
                    "member_node_ids": c["member_node_ids"],
                }
                sig_to_scores[sig] = []
            sig_to_scores[sig].append(c["score"])

    from collections import defaultdict
    task_to_sigs = defaultdict(list)
    for sig, meta in sig_to_metadata.items():
        task_id = meta["task"]["id"]
        task_to_sigs[task_id].append(sig)
        
    ensembled_scores = {}
    for task_id, sigs in task_to_sigs.items():
        rrf_sums = {sig: 0.0 for sig in sigs}
        
        for seed_idx in range(num_seeds):
            sig_scores = []
            for sig in sigs:
                score_list = sig_to_scores[sig]
                score = score_list[seed_idx] if seed_idx < len(score_list) else 0.0
                sig_scores.append((score, sig))
                
            sig_scores.sort(key=lambda x: x[0], reverse=True)
            
            for rank_idx, (_, sig) in enumerate(sig_scores, start=1):
                rrf_sums[sig] += 1.0 / (k + rank_idx)
                
        max_possible_rrf = num_seeds / (k + 1)
        for sig in sigs:
            ensembled_scores[sig] = rrf_sums[sig] / max_possible_rrf

    flat_ensembled = []
    for sig, meta in sig_to_metadata.items():
        score_list = sig_to_scores[sig]
        avg_score = float(np.mean(score_list)) if score_list else 0.0
        
        flat_ensembled.append({
            "task": meta["task"],
            "score": float(ensembled_scores[sig]),
            "score_avg": avg_score,
            "shortlisted_components": meta["shortlisted_components"],
            "supporting_capabilities": meta["supporting_capabilities"],
            "member_node_ids": meta["member_node_ids"],
        })
        
    flat_ensembled.sort(key=lambda x: x["score"], reverse=True)
    
    by_use_case = {}
    for c in flat_ensembled:
        t_id = c["task"]["id"]
        t_name = c["task"]["name"]
        by_use_case.setdefault(
            t_id,
            {
                "task": {"id": t_id, "name": t_name},
                "shortlisted_technique_blueprints": [],
            },
        )
        by_use_case[t_id]["shortlisted_technique_blueprints"].append(
            {
                "score": c["score"],
                "score_avg": c["score_avg"],
                "shortlisted_components": c["shortlisted_components"],
                "supporting_capabilities": c["supporting_capabilities"],
            }
        )
        
    grouped = sorted(by_use_case.values(), key=lambda x: x["task"]["name"])
    for entry in grouped:
        entry["shortlisted_technique_blueprints"].sort(key=lambda x: x["score"], reverse=True)
        entry["shortlisted_technique_blueprints"] = entry["shortlisted_technique_blueprints"][:5]
        
    return flat_ensembled, grouped


def aggregate_metrics(all_metrics: list[dict]) -> dict:
    keys = list(all_metrics[0].keys())
    summary = {}
    for k in keys:
        vals = [m[k] for m in all_metrics]
        summary[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    summary["per_seed"] = all_metrics
    return summary


def log_aggregated_metrics(summary: dict, seeds: list[int], model_name: str) -> None:
    logging.info("=" * 60)
    logging.info(f"  Multi-Seed Summary ({model_name.upper()}, {len(seeds)} seeds)")
    logging.info("=" * 60)
    for k, v in summary.items():
        if k == "per_seed":
            continue
        logging.info(f"  {k:>15s}:  {v['mean']:.4f}  ±  {v['std']:.4f}  "
                     f"[{v['min']:.4f}, {v['max']:.4f}]")
    logging.info("=" * 60)


def save_aggregated_metrics(summary: dict, run_dir: Path, model_name: str) -> Path:
    path = run_dir / f"multi_seed_metrics_{model_name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Aggregated metrics saved to {path}")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed HGNN and Baseline training + RRF ensemble discovery"
    )
    parser.add_argument(
        "--model",
        choices=["hgnn", "edge_logreg"],
        default="hgnn",
        help="Hypergraph model type to ensemble (default: hgnn)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 7, 123, 999, 2024],
        help="Random seeds to run (default: 42 7 123 999 2024)",
    )
    parser.add_argument(
        "--graph-path",
        type=str,
        default="data/hypergraph_output/hypergraph_latest.pkl",
        help="Path to a saved hypergraph .pkl file",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--out-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--edge-pooling", type=str, default=None)
    
    args = parser.parse_args()
    model_type = args.model
    seeds = args.seeds
    graph_path = args.graph_path

    hparams = dict(MODEL_DEFAULTS[model_type])
    
    if args.epochs is not None:
        hparams["epochs"] = args.epochs
    if args.lr is not None:
        hparams["lr"] = args.lr
    if args.hidden_dim is not None:
        hparams["hidden_dim"] = args.hidden_dim
    if args.out_dim is not None:
        hparams["out_dim"] = args.out_dim
    if args.dropout is not None:
        hparams["dropout"] = args.dropout
    if args.edge_pooling is not None:
        hparams["edge_pooling"] = args.edge_pooling

    run_dir = setup_run_dir(model_type=model_type)
    setup_logging(run_dir)

    logging.info(f"Multi-Seed Hypergraph Discovery Runner — {model_type}")
    logging.info(f"Seeds: {seeds}")
    logging.info(f"Hypergraph path: {graph_path}")
    logging.info(f"Run dir: {run_dir}")

    logging.info("Loading hypergraph data...")
    data = load_hypergraph(graph_path)

    hparams["seeds"] = seeds
    hparams["num_seeds"] = len(seeds)
    hparams["graph_path"] = graph_path
    
    env_info = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
    }
    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "environment": env_info,
        "hyperparameters": hparams,
    }
    with open(run_dir / "hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)

    all_metrics = []
    all_candidates = []

    for seed_idx, seed in enumerate(seeds):
        if model_type == "hgnn":
            metrics, candidates = train_hgnn_one_seed(
                data=data,
                hparams=hparams,
                seed=seed,
                seed_idx=seed_idx,
                run_dir=run_dir,
            )
        elif model_type == "edge_logreg":
            metrics, candidates = train_edge_logreg_one_seed(
                data=data,
                hparams=hparams,
                seed=seed,
                seed_idx=seed_idx,
                run_dir=run_dir,
            )
        else:
            raise ValueError(f"Unknown model: {model_type}")

        all_metrics.append(metrics)
        all_candidates.append(candidates)

    summary = aggregate_metrics(all_metrics)
    log_aggregated_metrics(summary, seeds, model_type)
    save_aggregated_metrics(summary, run_dir, model_name=model_type)

    logging.info("Aggregating candidate blueprint rankings using Rank-Based Fusion (RRF)...")
    flat_ensembled, grouped_by_use_case = aggregate_scores(all_candidates)

    output = {
        "run_dir": str(run_dir),
        "model": f"{model_type}_ensemble",
        "seeds": seeds,
        "num_seeds": len(seeds),
        "evaluation_metrics": summary,
        "shortlists_by_use_case": grouped_by_use_case,
        "novel_techniques": flat_ensembled,
    }

    output_path = run_dir / f"novel_techniques_{model_type}_ensembled.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logging.info(f"Ensembled discovery complete. Saved results to {output_path}")
    logging.info(f"Multi-seed {model_type} run completed successfully. Results in: {run_dir}")


if __name__ == "__main__":
    main()
