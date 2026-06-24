"""
Run Seed-Scoped Discovery from a Saved Hypergraph Checkpoint
============================================================
Replay the discovery pipeline for a single saved hypergraph checkpoint (HGNN or EdgeLogReg)
and persist the result using the standard novel_techniques JSON schema.

Typical usage:
    python link_prediction/run_saved_checkpoint_hyper_discovery.py \
        --checkpoint link_pred_output/MultiSeed_HGNN/run_20260618_092845/model_hgnn_seed_42.pt
"""

import argparse
import datetime
import json
import logging
import os
import re
import pickle
import random
import sys
import shutil
from pathlib import Path

import torch
import numpy as np

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


def infer_model_category(checkpoint_path: Path, source_hparams: dict) -> str:
    """Infer model type (hgnn or edge_logreg) from hyperparameters or checkpoint path."""
    model_type = source_hparams.get("model_type")
    if model_type:
        return str(model_type).lower()

    stem = checkpoint_path.stem.lower()
    if "hgnn" in stem:
        return "hgnn"
    if "edge_logreg" in stem:
        return "edge_logreg"

    parent_name = checkpoint_path.parent.name.lower()
    if "hgnn" in parent_name:
        return "hgnn"
    if "edge_logreg" in parent_name:
        return "edge_logreg"

    raise ValueError(
        f"Could not infer model type from checkpoint '{checkpoint_path.name}'. "
        "Make sure to pass a valid checkpoint from a run containing hyperparameters.json."
    )


def extract_checkpoint_seed(checkpoint_path: Path) -> int | None:
    match = re.search(r"_seed_(\d+)", checkpoint_path.stem)
    return int(match.group(1)) if match else None


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_graph_path(source_hparams: dict) -> str:
    graph_path = source_hparams.get("graph_path")
    if graph_path:
        normalized = str(graph_path).replace("\\", os.sep)
        p = Path(normalized)
        if p.exists():
            return str(p)

        cwd_candidate = Path.cwd() / normalized
        if cwd_candidate.exists():
            return str(cwd_candidate)

        alt_candidate = Path.cwd() / str(graph_path).replace("\\", "/")
        if alt_candidate.exists():
            return str(alt_candidate)

    return "data/hypergraph_output/hypergraph_latest.pkl"


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay discovery from a saved hypergraph checkpoint and write novel_techniques JSON"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Path to a saved model checkpoint such as model_hgnn_seed_42.pt",
    )
    parser.add_argument(
        "--graph-path",
        type=str,
        default=None,
        help="Override the graph path stored in hyperparameters.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to a seed-scoped folder next to the checkpoint.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    source_run_dir = checkpoint_path.parent
    hparams_path = source_run_dir / "hyperparameters.json"
    if not hparams_path.exists():
        raise FileNotFoundError(
            f"No hyperparameters.json found next to checkpoint: {hparams_path}"
        )

    source_hparams_record = load_json(str(hparams_path))
    source_hparams = source_hparams_record.get("hyperparameters", {})

    model_key = infer_model_category(checkpoint_path, source_hparams)
    graph_path = args.graph_path or resolve_graph_path(source_hparams)
    graph_data = load_hypergraph(graph_path)

    seed_from_fname = extract_checkpoint_seed(checkpoint_path)
    if seed_from_fname is not None:
        source_hparams["seed"] = seed_from_fname

    seed = int(source_hparams.get("seed", 42))

    model_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else source_run_dir / f"seed_discovery__{model_key}"
    )
    model_dir.mkdir(parents=True, exist_ok=True)

    per_seed_dir = model_dir / f"{checkpoint_path.stem}"
    per_seed_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(per_seed_dir)

    logging.info("Seed-Scoped Hypergraph Discovery Replay")
    logging.info("Checkpoint: %s", checkpoint_path)
    logging.info("Source run dir: %s", source_run_dir)
    logging.info("Model key: %s", model_key)
    logging.info("Seed: %s", seed)
    logging.info("Graph: %s", graph_path)
    logging.info("Per-seed output dir: %s", per_seed_dir)
    logging.info("Model-level collection dir: %s", model_dir)

    # Re-run strict member-disjoint splits and evaluations
    set_seed(seed)
    
    node_features: torch.Tensor = graph_data["node_features"]
    hyperedge_index: torch.Tensor = graph_data["hyperedge_index"]
    by_type, metadata_by_idx, node_types_by_idx = get_node_index_maps(
        graph_data["node_metadata"], graph_data["node_to_idx"]
    )

    all_memberships = build_membership_map(hyperedge_index.cpu())

    train_edges, test_edges, leakage_report = split_hyperedges_strict_member_disjoint(
        memberships=all_memberships,
        node_types_by_idx=node_types_by_idx,
        test_ratio=source_hparams.get("test_ratio", 0.2),
        seed=seed,
    )
    
    test_memberships = {e: all_memberships[e] for e in test_edges if e in all_memberships}

    if model_key == "hgnn":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = node_features.to(device)
        train_set = set(train_edges)
        train_hyperedge_index = filter_hyperedge_index(hyperedge_index.cpu(), train_set).to(device)

        encoder = make_encoder(
            architecture=source_hparams.get("architecture", "hgnn"),
            in_dim=x.shape[1],
            hidden_dim=int(source_hparams.get("hidden_dim", 256)),
            out_dim=int(source_hparams.get("out_dim", 128)),
            dropout=float(source_hparams.get("dropout", 0.2)),
        ).to(device)

        edge_pooling = source_hparams.get("edge_pooling", "mean_max_std")
        edge_emb_dim = int(source_hparams.get("out_dim", 128)) if edge_pooling == "mean" else int(source_hparams.get("out_dim", 128)) * 3
        scorer = HyperedgeScorer(emb_dim=edge_emb_dim, hidden_dim=int(source_hparams.get("hidden_dim", 256))).to(device)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        scorer.load_state_dict(checkpoint["scorer_state_dict"])

        encoder.eval()
        scorer.eval()

        with torch.no_grad():
            z_all = encoder(x, train_hyperedge_index)

        # Test negatives with seed + 1337
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
                test_pos_vecs = aggregate_hyperedge_embeddings(z_all, test_memberships, test_edge_indices, mode=edge_pooling)
                test_neg_vecs = aggregate_hyperedge_embeddings(z_all, test_neg_memberships, test_neg_indices, mode=edge_pooling)
                test_pos_scores = torch.sigmoid(scorer(test_pos_vecs)).detach().cpu()
                test_neg_scores = torch.sigmoid(scorer(test_neg_vecs)).detach().cpu()
            metrics = evaluate_link_prediction(test_pos_scores, test_neg_scores, k_list=[1, 3, 5, 10])
        else:
            metrics = {k: 0.0 for k in ["Hits@1", "Hits@3", "Hits@5", "Hits@10", "MRR", "AUC", "AP", "PairwiseAcc"]}

        known_signatures = {signature(v) for v in all_memberships.values()}
        candidates, grouped_by_use_case = discover_candidates(
            node_embeddings=z_all,
            scorer=scorer,
            by_type=by_type,
            metadata_by_idx=metadata_by_idx,
            known_signatures=known_signatures,
            top_k=int(source_hparams.get("top_k", 100)),
            max_per_task=int(source_hparams.get("max_per_task", 10)),
            edge_pooling=edge_pooling,
        )
        
        model_str = "HGNN(HypergraphConv)"

    elif model_key == "edge_logreg":
        with open(checkpoint_path, "rb") as f:
            model = pickle.load(f)

        train_memberships = {e: all_memberships[e] for e in train_edges if e in all_memberships}
        train_visible_nodes = sorted({n for m in train_memberships.values() for n in m})

        # Test negatives with seed + 999, and candidate pool is train_visible_nodes
        test_neg_memberships = make_negative_membership_typed(
            memberships=test_memberships,
            edge_indices=list(test_memberships.keys()),
            candidate_nodes=train_visible_nodes,
            node_types_by_idx=node_types_by_idx,
            num_negatives=1,
            seed=seed + 999,
        )

        node_features_np = node_features.detach().cpu().numpy()
        feature_mode = source_hparams.get("feature_mode", "mean_max_std")

        x_test_pos = np.stack([edge_feature(v, node_features_np, mode=feature_mode) for v in test_memberships.values()])
        x_test_neg = np.stack([edge_feature(v, node_features_np, mode=feature_mode) for v in test_neg_memberships.values()])

        pos_scores = torch.tensor(model.predict_proba(x_test_pos)[:, 1], dtype=torch.float32)
        neg_scores = torch.tensor(model.predict_proba(x_test_neg)[:, 1], dtype=torch.float32)

        metrics = evaluate_link_prediction(pos_scores, neg_scores, k_list=[1, 3, 5, 10])

        scorer = SklearnEdgeScorer(model)
        node_embeddings = torch.tensor(node_features_np, dtype=torch.float32)
        known_signatures = {signature(v) for v in all_memberships.values()}

        candidates, grouped_by_use_case = discover_candidates(
            node_embeddings=node_embeddings,
            scorer=scorer,
            by_type=by_type,
            metadata_by_idx=metadata_by_idx,
            known_signatures=known_signatures,
            top_k=int(source_hparams.get("top_k", 100)),
            max_per_task=int(source_hparams.get("max_per_task", 10)),
            edge_pooling=feature_mode,
        )

        model_str = "EdgeLogReg(mean/max/std features)"

    else:
        raise ValueError(f"Unsupported model key: {model_key}")

    log_metrics(metrics, prefix=f"{model_key.upper()} Replay Seed {seed}")

    output = {
        "run_dir": str(model_dir),
        "model": model_str,
        "train_hyperedges": len(train_edges),
        "test_hyperedges": len(test_edges),
        "leakage_guard": leakage_report,
        "evaluation_metrics": metrics,
        "shortlists_by_use_case": grouped_by_use_case,
        "novel_techniques": candidates,
    }

    output_path = model_dir / f"novel_techniques_{model_key}_seed_{seed}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    metadata = {
        "mode": "saved_checkpoint_seed_discovery",
        "checkpoint_path": str(checkpoint_path),
        "source_run_dir": str(source_run_dir),
        "graph_path": graph_path,
        "model_key": model_key,
        "seed": seed,
        "output_json": str(output_path),
        "output_schema": "novel_techniques_standard",
    }
    with open(per_seed_dir / "discovery_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(per_seed_dir / "hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump(source_hparams, f, indent=2)

    logging.info("Discovery replay completed successfully.")
    logging.info("JSON saved to %s", output_path)

    # Clean up the per-seed temporary directory
    for handler in logging.root.handlers[:]:
        handler.close()
        logging.root.removeHandler(handler)

    if per_seed_dir.exists() and per_seed_dir != model_dir:
        shutil.rmtree(per_seed_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}")
        raise
