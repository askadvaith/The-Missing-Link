"""
Multi-Seed Discovery Runner
============================
Trains a GNN model over M random seeds and aggregates predictions via
Mean Score Aggregation (Soft-Voting Ensemble).

For each seed:
  - Trains the model with `create_technique_wise_split`
  - Evaluates deterministically (no random negative sampling)
  - Computes per-(task, component) scores for the ensemble

After all seeds:
  - Averages scores element-wise across seeds (soft vote)
  - Applies Dynamic Relative Thresholding on the ensembled scores
  - Reports mean ± std for all test metrics across seeds

Usage:
    python link_prediction/run_multi_seed_discovery.py
    python link_prediction/run_multi_seed_discovery.py --model GCN --seeds 5
    python link_prediction/run_multi_seed_discovery.py --model GAT --seeds 42 7 123 999 2024

Output: link_pred_output/MultiSeed_<MODEL>/run_<timestamp>/
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import logging
import numpy as np
import torch

from link_prediction.utils import (
    set_seed,
    select_graph_run,
    load_graph_data,
    setup_run_directory,
    setup_logging,
    log_hyperparameters,
    get_component_indices,
    get_component_degree_map,
    apply_degree_penalty,
    evaluate_deterministic,
    _build_existing_task_component_pairs,
)
from utils.metrics import log_metrics, save_metrics


# ──────────────────────────────────────────────────────────────────────
# Supported model types and their default hyperparameters
# ──────────────────────────────────────────────────────────────────────
MODEL_DEFAULTS = {
    "GCN": {
        "model_type": "GCN",
        "epochs": 300,
        "hidden_channels": 64,
        "out_channels": 64,
        "dropout": 0.2,
        "lr": 0.01,
        "weight_decay": 5e-4,
        "margin": 0.5,
        "num_neg": 1,
        "test_ratio": 0.2,
        "top_k": 5,
        "discovery_strategy": "top_k",
        "relative_threshold_factor": 0.90,
        "max_candidates_per_task": 5,
        "degree_penalty_alpha": 0.80,
        "exclude_existing_pairs": True,
        "temperature_init": 2.0,
    },
    "GAT": {
        "model_type": "GAT",
        "epochs": 300,
        "hidden_channels": 32,
        "out_channels": 64,
        "heads": 4,
        "dropout": 0.2,
        "lr": 0.005,
        "weight_decay": 5e-4,
        "margin": 0.5,
        "num_neg": 1,
        "test_ratio": 0.2,
        "top_k": 5,
        "discovery_strategy": "relative",
        "relative_threshold_factor": 0.90,
        "max_candidates_per_task": 5,
        "degree_penalty_alpha": 2.00,
        "exclude_existing_pairs": True,
        "temperature_init": 2.0,
    },
    "GTN": {
        "model_type": "GTN",
        "epochs": 300,
        "hidden_channels": 16,
        "out_channels": 64,
        "heads": 2,
        "dropout": 0.3,
        "lr": 0.001,
        "weight_decay": 5e-4,
        "margin": 0.5,
        "num_neg": 1,
        "test_ratio": 0.2,
        "top_k": 5,
        "discovery_strategy": "relative",
        "relative_threshold_factor": 0.95,
        "max_candidates_per_task": 3,
        "degree_penalty_alpha": 1.50,
        "exclude_existing_pairs": True,
        "use_layer_norm": True,
        "use_skip_connection": True,
        "beta": True,
        "temperature_init": 2.0,
    },
    "RGCN": {
        "model_type": "RGCN",
        "epochs": 300,
        "hidden_channels": 64,
        "out_channels": 64,
        "num_bases": 4,
        "dropout": 0.2,
        "lr": 0.01,
        "weight_decay": 1e-4,
        "margin": 0.5,
        "num_neg": 1,
        "test_ratio": 0.2,
        "top_k": 5,
        "discovery_strategy": "top_k",
        "relative_threshold_factor": 0.90,
        "max_candidates_per_task": 5,
        "degree_penalty_alpha": 0.00,
        "exclude_existing_pairs": True,
        "temperature_init": 2.0,
    },
    "Node2Vec": {
        "model_type": "Node2Vec",
        "embedding_dim": 64,
        "walk_length": 20,
        "context_size": 10,
        "walks_per_node": 10,
        "p": 1.0,
        "q": 1.0,
        "num_negative_samples": 1,
        "node2vec_epochs": 100,
        "node2vec_lr": 0.01,
        "node2vec_batch_size": 128,
        "decoder_epochs": 300,
        "decoder_lr": 0.01,
        "decoder_weight_decay": 1e-4,
        "decoder_hidden": 64,
        "margin": 0.5,
        "num_neg": 1,
        "test_ratio": 0.2,
        "top_k": 5,
        "discovery_strategy": "top_k",
        "relative_threshold_factor": 0.90,
        "max_candidates_per_task": 5,
        "degree_penalty_alpha": 1.50,
        "exclude_existing_pairs": True,
    },
    "Metapath2Vec": {
        "model_type": "Metapath2Vec",
        "embedding_dim": 64,
        "walk_length": 20,
        "context_size": 10,
        "walks_per_node": 5,
        "num_negative_samples": 5,
        "metapath2vec_epochs": 100,
        "metapath2vec_lr": 0.01,
        "metapath2vec_batch_size": 128,
        "decoder_epochs": 300,
        "decoder_lr": 0.01,
        "decoder_weight_decay": 1e-4,
        "margin": 0.5,
        "num_neg": 1,
        "test_ratio": 0.2,
        "top_k": 5,
        "discovery_strategy": "top_k",
        "relative_threshold_factor": 0.90,
        "max_candidates_per_task": 5,
        "degree_penalty_alpha": 0.05,
        "exclude_existing_pairs": True,
    },
}


# ──────────────────────────────────────────────────────────────────────
# Per-model score matrix extraction
# ──────────────────────────────────────────────────────────────────────
def get_score_matrix(
    model,
    z: torch.Tensor,
    graph_data: dict,
    structure_edge_index: torch.Tensor,
    degree_penalty_alpha: float,
) -> dict:
    """
    Compute degree-penalized sigmoid scores for every (task, component) pair.

    Returns:
        score_matrix: dict mapping (task_idx: int, comp_idx: int) → float score
    """
    task_indices = graph_data["node_type_to_indices"]["Task"]
    all_comp_indices = get_component_indices(graph_data)
    comp_degree_map = get_component_degree_map(graph_data, structure_edge_index)

    score_matrix: dict[tuple[int, int], float] = {}

    model.eval()
    with torch.no_grad():
        for task_idx in task_indices:
            src = torch.full((len(all_comp_indices),), task_idx, dtype=torch.long)
            dst = torch.tensor(all_comp_indices, dtype=torch.long)
            edge_batch = torch.stack([src, dst], dim=0)

            raw_scores = model.decode(z, edge_batch).sigmoid()
            scores = apply_degree_penalty(
                raw_scores, all_comp_indices, comp_degree_map, degree_penalty_alpha
            )
            for comp_idx, score_val in zip(all_comp_indices, scores.tolist()):
                score_matrix[(task_idx, comp_idx)] = score_val

    return score_matrix


# ──────────────────────────────────────────────────────────────────────
# Train + evaluate one seed  (model-type dispatched)
# ──────────────────────────────────────────────────────────────────────
def train_one_seed(
    graph_data: dict,
    hparams: dict,
    model_type: str,
    seed: int,
    seed_idx: int,
    run_dir: str,
) -> tuple[dict, dict]:
    """
    Trains the model for `seed`, evaluates deterministically, and extracts the
    full (task, component) score matrix for ensembling.
    """
    hparams = dict(hparams)
    hparams["seed"] = seed
    set_seed(seed)

    logging.info("=" * 60)
    logging.info(f"  Seed run {seed_idx + 1}  |  seed={seed}")
    logging.info("=" * 60)

    if model_type == "GCN":
        from link_prediction.gcn_link_prediction import train_model
        model, structure_edge_index, train_pos_edge_index, test_pos_edge_index, _ = (
            train_model(graph_data, hparams)
        )
        # Save checkpoint before evaluation
        checkpoint_path = os.path.join(run_dir, f"model_gcn_seed_{seed}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        logging.info(f"Model saved to {checkpoint_path}")

        model.eval()
        with torch.no_grad():
            z = model.encode(graph_data["node_features"], structure_edge_index)
        encode_edge_index = structure_edge_index

    elif model_type == "GAT":
        from link_prediction.gat_link_prediction import train_model
        model, structure_edge_index, train_pos_edge_index, test_pos_edge_index, _ = (
            train_model(graph_data, hparams)
        )
        checkpoint_path = os.path.join(run_dir, f"model_gat_seed_{seed}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        logging.info(f"Model saved to {checkpoint_path}")

        model.eval()
        with torch.no_grad():
            z = model.encode(graph_data["node_features"], structure_edge_index)
        encode_edge_index = structure_edge_index

    elif model_type == "GTN":
        from link_prediction.gtn_link_prediction import train_model
        model, structure_edge_index, train_pos_edge_index, test_pos_edge_index, _ = (
            train_model(graph_data, hparams)
        )
        checkpoint_path = os.path.join(run_dir, f"model_gtn_seed_{seed}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        logging.info(f"Model saved to {checkpoint_path}")

        model.eval()
        with torch.no_grad():
            z = model.encode(graph_data["node_features"], structure_edge_index)
        encode_edge_index = structure_edge_index

    elif model_type == "RGCN":
        from link_prediction.rgcn_link_prediction import train_model, RGCNDiscoveryWrapper
        model, train_edge_index, train_edge_type, train_pos_edge_index, test_pos_edge_index, _ = (
            train_model(graph_data, hparams)
        )
        checkpoint_path = os.path.join(run_dir, f"model_rgcn_seed_{seed}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        logging.info(f"Model saved to {checkpoint_path}")

        # Wrap RGCN so downstream calls use the standard encode(x, edge_index) signature
        model = RGCNDiscoveryWrapper(model, train_edge_index, train_edge_type)
        structure_edge_index = train_edge_index
        model.eval()
        with torch.no_grad():
            z = model.encode(graph_data["node_features"], structure_edge_index)
        encode_edge_index = structure_edge_index

    elif model_type == "Node2Vec":
        from link_prediction.node2vec_link_prediction import train_model
        model, structure_edge_index, train_pos_edge_index, test_pos_edge_index, _ = (
            train_model(graph_data, hparams)
        )
        checkpoint_path = os.path.join(run_dir, f"model_node2vec_seed_{seed}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        logging.info(f"Model saved to {checkpoint_path}")

        model.eval()
        with torch.no_grad():
            z = model.encode(None, None)
        encode_edge_index = structure_edge_index

    elif model_type == "Metapath2Vec":
        from link_prediction.metapath2vec_link_prediction import train_model
        model, structure_edge_index, train_pos_edge_index, test_pos_edge_index, _ = (
            train_model(graph_data, hparams)
        )
        checkpoint_path = os.path.join(run_dir, f"model_metapath2vec_seed_{seed}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        logging.info(f"Model saved to {checkpoint_path}")

        model.eval()
        with torch.no_grad():
            z = model.encode(None, None)
        encode_edge_index = structure_edge_index

    else:
        raise ValueError(f"Unknown model type: {model_type!r}")

    # Deterministic evaluation
    metrics = evaluate_deterministic(
        model, z,
        test_pos_edge_index=test_pos_edge_index,
        train_pos_edge_index=train_pos_edge_index,
        graph_data=graph_data,
        structure_edge_index=encode_edge_index,
        degree_penalty_alpha=hparams["degree_penalty_alpha"],
    )

    log_metrics(metrics, prefix=f"Seed {seed}")

    # Extract full score matrix for ensemble
    score_matrix = get_score_matrix(
        model, z, graph_data, encode_edge_index, hparams["degree_penalty_alpha"]
    )

    return metrics, score_matrix


# ──────────────────────────────────────────────────────────────────────
# Aggregate score matrices (soft vote)
# ──────────────────────────────────────────────────────────────────────
def aggregate_scores(score_matrices: list[dict], k: int = 60) -> dict:
    """
    Perform Reciprocal Rank Fusion (RRF) on the scores across M seed runs.
    For each task, components are ranked by score, and the RRF score is computed.
    The final score is normalized so that the maximum possible score (rank 1 in all seeds) is 1.0.

    Returns:
        ensemble_scores: {(task_idx, comp_idx): normalized_rrf_score}
    """
    all_keys = list(score_matrices[0].keys())
    
    # Group components by task
    from collections import defaultdict
    task_to_comps = defaultdict(list)
    for task_idx, comp_idx in all_keys:
        task_to_comps[task_idx].append(comp_idx)
        
    num_seeds = len(score_matrices)
    ensemble = {}
    
    # Compute RRF for each task
    for task_idx, comp_indices in task_to_comps.items():
        # Initialize RRF sum for all components of this task
        rrf_sums = {comp_idx: 0.0 for comp_idx in comp_indices}
        
        for sm in score_matrices:
            # Get components and scores for this task in this seed
            comp_scores = [(sm.get((task_idx, comp_idx), 0.0), comp_idx) for comp_idx in comp_indices]
            # Sort descending by score to establish ranks
            comp_scores.sort(key=lambda x: x[0], reverse=True)
            
            # Compute RRF contribution: 1 / (k + rank)
            for rank_idx, (_, comp_idx) in enumerate(comp_scores, start=1):
                rrf_sums[comp_idx] += 1.0 / (k + rank_idx)
                
        # Normalize RRF scores so the maximum possible score is 1.0
        max_possible_rrf = num_seeds / (k + 1)
        for comp_idx in comp_indices:
            ensemble[(task_idx, comp_idx)] = float(rrf_sums[comp_idx] / max_possible_rrf)
            
    return ensemble


# ──────────────────────────────────────────────────────────────────────
# Ensemble-based discovery (relative threshold on averaged scores)
# ──────────────────────────────────────────────────────────────────────
def discover_from_ensemble(
    ensemble_scores: dict,
    graph_data: dict,
    run_dir: str,
    model_name: str,
    discovery_strategy: str,
    top_k: int,
    relative_threshold_factor: float,
    max_candidates_per_task: int,
    exclude_existing_pairs: bool,
) -> list:
    """
    Apply either Top-K or Dynamic Relative Thresholding to the ensembled score matrix
    and save novel technique candidates.
    """
    task_indices = graph_data["node_type_to_indices"]["Task"]
    idx_to_node = graph_data["idx_to_node"]
    node_meta = {n["id"]: n for n in graph_data["node_metadata"]}
    observed_pairs = (
        _build_existing_task_component_pairs(graph_data) if exclude_existing_pairs else set()
    )

    candidates = []

    if discovery_strategy == "top_k":
        # Group by component types
        comp_types = {
            "AlgorithmicComponent": graph_data["node_type_to_indices"]["AlgorithmicComponent"],
            "PromptComponent": graph_data["node_type_to_indices"]["PromptComponent"],
            "DataFlow": graph_data["node_type_to_indices"]["DataFlow"],
        }
        
        for task_idx in task_indices:
            task_id = idx_to_node[task_idx]
            task_name = node_meta[task_id]["name"]
            components_found = []

            for type_name, c_indices in comp_types.items():
                if not c_indices:
                    continue

                type_scores = [
                    (ensemble_scores.get((task_idx, comp_idx), 0.0), comp_idx)
                    for comp_idx in c_indices
                ]
                type_scores.sort(key=lambda x: x[0], reverse=True)
                
                k = min(top_k, len(c_indices))
                top_scores = type_scores[:k]

                for score, comp_idx in top_scores:
                    if exclude_existing_pairs and (task_idx, comp_idx) in observed_pairs:
                        continue
                    c_id = idx_to_node[comp_idx]
                    c_info = node_meta[c_id]
                    components_found.append(
                        {
                            "id": c_id,
                            "name": c_info["name"],
                            "type": c_info["type"],
                            "ensemble_score": float(score),
                        }
                    )

            components_found.sort(key=lambda x: x["ensemble_score"], reverse=True)

            candidates.append(
                {
                    "task": task_name,
                    "confidence_avg": (
                        sum(c["ensemble_score"] for c in components_found) / len(components_found)
                        if components_found
                        else 0.0
                    ),
                    "suggested_components": components_found,
                }
            )

    else:  # relative
        all_comp_indices = get_component_indices(graph_data)
        for task_idx in task_indices:
            task_id = idx_to_node[task_idx]

            # Retrieve ensemble scores for this task
            task_scores = [
                (ensemble_scores.get((task_idx, comp_idx), 0.0), comp_idx)
                for comp_idx in all_comp_indices
            ]
            if not task_scores:
                continue

            max_score = max(s for s, _ in task_scores)
            if max_score == 0.0:
                continue
            cutoff = max_score * relative_threshold_factor

            scored_comps = [
                (score, comp_idx)
                for score, comp_idx in task_scores
                if score >= cutoff
                and not (exclude_existing_pairs and (task_idx, comp_idx) in observed_pairs)
            ]
            scored_comps.sort(key=lambda x: x[0], reverse=True)
            scored_comps = scored_comps[:max_candidates_per_task]

            if not scored_comps:
                continue

            components_found = []
            for score, node_idx in scored_comps:
                c_id = idx_to_node[node_idx]
                c_info = node_meta[c_id]
                components_found.append(
                    {
                        "id": c_id,
                        "name": c_info["name"],
                        "type": c_info["type"],
                        "ensemble_score": float(score),
                    }
                )

            candidates.append(
                {
                    "task": node_meta[task_id]["name"],
                    "max_ensemble_score": float(max_score),
                    "confidence_avg": (
                        sum(c["ensemble_score"] for c in components_found)
                        / len(components_found)
                    ),
                    "suggested_components": components_found,
                }
            )

    candidates.sort(key=lambda x: x["confidence_avg"], reverse=True)

    output_path = os.path.join(run_dir, f"novel_techniques_{model_name}_ensembled.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2)

    logging.info(f"Ensemble discovery ({discovery_strategy}): found suggestions for {len(candidates)} tasks.")
    logging.info(f"Saved to {output_path}")
    return candidates


# ──────────────────────────────────────────────────────────────────────
# Metrics aggregation (mean ± std across seeds)
# ──────────────────────────────────────────────────────────────────────
def aggregate_metrics(all_metrics: list[dict]) -> dict:
    """
    Compute mean and std for each metric key across M seed runs.

    Returns:
        {
            "Hits@1": {"mean": ..., "std": ...},
            ...
            "per_seed": [seed_metrics_0, seed_metrics_1, ...]
        }
    """
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


def save_aggregated_metrics(summary: dict, run_dir: str, model_name: str) -> str:
    path = os.path.join(run_dir, f"multi_seed_metrics_{model_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Aggregated metrics saved to {path}")
    return path


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-seed GNN training + soft-voting ensemble discovery"
    )
    parser.add_argument(
        "--model",
        choices=["GCN", "GAT", "GTN", "RGCN", "Node2Vec", "Metapath2Vec"],
        default="GCN",
        help="GNN or baseline model type to ensemble (default: GCN)",
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
        default=None,
        help="Path to a saved graph .pt file. If omitted, shows interactive selector.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_type = args.model
    seeds = args.seeds

    hparams = dict(MODEL_DEFAULTS[model_type])

    # 1. Select graph
    if args.graph_path:
        graph_path = args.graph_path
    else:
        graph_path = select_graph_run()
    hparams["graph_path"] = graph_path

    # 2. Setup run directory
    run_dir, timestamp = setup_run_directory(model_type=f"MultiSeed_{model_type}")
    setup_logging(run_dir)

    logging.info(f"Multi-Seed Discovery Runner — {model_type}")
    logging.info(f"Seeds: {seeds}")
    logging.info(f"Graph: {graph_path}")
    logging.info(f"Run dir: {run_dir}")

    hparams["seeds"] = seeds
    hparams["num_seeds"] = len(seeds)
    log_hyperparameters(run_dir, hparams)

    # 3. Load graph (once)
    logging.info("Loading graph data...")
    graph_data = load_graph_data(graph_path)
    hparams["num_nodes"] = graph_data["num_nodes"]
    hparams["num_edges"] = graph_data["num_edges"]
    hparams["feature_dim"] = int(graph_data["node_features"].shape[1])
    log_hyperparameters(run_dir, hparams)

    # 4. Train M models, collect per-seed metrics and score matrices
    all_metrics: list[dict] = []
    all_score_matrices: list[dict] = []

    for seed_idx, seed in enumerate(seeds):
        metrics, score_matrix = train_one_seed(
            graph_data=graph_data,
            hparams=hparams,
            model_type=model_type,
            seed=seed,
            seed_idx=seed_idx,
            run_dir=run_dir,
        )
        all_metrics.append(metrics)
        all_score_matrices.append(score_matrix)

    # 5. Aggregate metrics (mean ± std)
    summary = aggregate_metrics(all_metrics)
    log_aggregated_metrics(summary, seeds, model_type)
    save_aggregated_metrics(summary, run_dir, model_name=model_type.lower())

    # 6. Rank-Based Fusion (RRF) ensemble: fuse component rankings across seeds
    logging.info("Aggregating score matrices (Rank-Based Fusion - RRF)...")
    ensemble_scores = aggregate_scores(all_score_matrices)

    # 7. Discover from ensembled scores
    discover_from_ensemble(
        ensemble_scores=ensemble_scores,
        graph_data=graph_data,
        run_dir=run_dir,
        model_name=model_type.lower(),
        discovery_strategy=hparams.get("discovery_strategy", "relative"),
        top_k=hparams["top_k"],
        relative_threshold_factor=hparams["relative_threshold_factor"],
        max_candidates_per_task=hparams["max_candidates_per_task"],
        exclude_existing_pairs=hparams["exclude_existing_pairs"],
    )

    logging.info(f"Multi-seed {model_type} run completed. Results in: {run_dir}")


if __name__ == "__main__":
    main()
