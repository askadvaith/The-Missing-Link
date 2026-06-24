"""
Run Seed-Scoped Discovery from a Saved Checkpoint
=================================================
Replay the discovery pipeline for a single saved checkpoint and persist the
result using the standard novel_techniques JSON schema.

Typical usage:
    python link_prediction/run_saved_checkpoint_discovery.py \
        --checkpoint link_pred_output/MultiSeed_GAT/RRF_run_20260602_010520/model_gat_seed_42.pt

By default, the script expects the checkpoint to live next to the matching
hyperparameters.json file and writes results into a dedicated subfolder next to
that checkpoint.
"""

import argparse
import json
import logging
import os
import re
import random
import sys
from pathlib import Path

import torch
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from link_prediction.utils import load_graph_data, log_hyperparameters, setup_logging


MODEL_NAME_MAP = {
    "gcn": "GCN",
    "gat": "GAT",
    "gtn": "GTN",
    "rgcn": "RGCN",
    "node2vec": "Node2Vec",
    "metapath2vec": "Metapath2Vec",
}


def infer_model_category(checkpoint_path: Path, source_hparams: dict) -> str:
    """Infer a model category from saved hyperparameters or the checkpoint name."""
    model_type = source_hparams.get("model_type")
    if model_type:
        return str(model_type)

    stem = checkpoint_path.stem.lower()
    match = re.match(r"model_(.+?)(?:_seed_\d+)?$", stem)
    if match:
        candidate = match.group(1)
        if candidate in MODEL_NAME_MAP:
            return MODEL_NAME_MAP[candidate]

    for candidate, display_name in MODEL_NAME_MAP.items():
        if candidate in stem:
            return display_name

    raise ValueError(
        f"Could not infer model type from checkpoint name '{checkpoint_path.name}'. "
        "Pass a checkpoint from a run that contains hyperparameters.json with model_type saved."
    )


def canonical_model_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def extract_checkpoint_seed(checkpoint_path: Path):
    match = re.search(r"_seed_(\d+)", checkpoint_path.stem)
    return int(match.group(1)) if match else None


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_graph_path(source_hparams: dict) -> str:
    """Resolve graph path from saved hyperparameters; fall back to selection prompt."""
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

    from link_prediction.utils import select_graph_run

    return select_graph_run()


def load_model_bundle(model_category, graph_data, source_hparams, checkpoint_path):
    """
    Restore model + structure edges + discovery strategy from a saved run.

    Returns:
        {
          "model": model_object,
          "structure_edge_index": tensor,
          "strategy": "topk" or "relative",
          "defaults": dict,
        }
    """
    # Ensure checkpoint_path is a Path object (caller may pass str)
    checkpoint_path = Path(checkpoint_path)

    key = canonical_model_key(model_category)

    # Prefer seed encoded in the checkpoint filename (model_<type>_seed_<N>.pt).
    # If present, override the saved hyperparameters so all parts of this
    # runner (including load_model_bundle) use the same seed.
    seed_from_fname = extract_checkpoint_seed(checkpoint_path)

    if seed_from_fname is not None:
        seed = seed_from_fname
        source_hparams["seed"] = seed
    else:
        seed = int(source_hparams.get("seed", 42))
    test_ratio = float(source_hparams.get("test_ratio", 0.2))
    in_channels = int(graph_data["node_features"].shape[1])

    if key == "gat":
        from link_prediction.gat_link_prediction import GATLinkPrediction
        from link_prediction.utils import create_technique_wise_split

        model = GATLinkPrediction(
            in_channels=in_channels,
            hidden_channels=int(source_hparams.get("hidden_channels", 32)),
            out_channels=int(source_hparams.get("out_channels", 64)),
            heads=int(source_hparams.get("heads", 4)),
            dropout=float(source_hparams.get("dropout", 0.2)),
            temperature_init=float(source_hparams.get("temperature_init", 2.0)),
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)
        structure_edge_index, _, _ = create_technique_wise_split(
            graph_data, test_ratio=test_ratio, seed=seed
        )
        return {
            "model": model,
            "structure_edge_index": structure_edge_index,
            "strategy": "relative",
            "defaults": {
                "relative_threshold_factor": float(source_hparams.get("relative_threshold_factor", 0.90)),
                "max_candidates_per_task": int(source_hparams.get("max_candidates_per_task", 5)),
                "degree_penalty_alpha": float(source_hparams.get("degree_penalty_alpha", 0.35)),
                "exclude_existing_pairs": bool(source_hparams.get("exclude_existing_pairs", True)),
                "top_k": int(source_hparams.get("top_k", 5)),
            },
        }

    if key == "gcn":
        from link_prediction.gcn_link_prediction import GCNLinkPrediction
        from link_prediction.utils import create_technique_wise_split

        model = GCNLinkPrediction(
            in_channels=in_channels,
            hidden_channels=int(source_hparams.get("hidden_channels", 64)),
            out_channels=int(source_hparams.get("out_channels", 64)),
            dropout=float(source_hparams.get("dropout", 0.2)),
            temperature_init=float(source_hparams.get("temperature_init", 2.0)),
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)
        structure_edge_index, _, _ = create_technique_wise_split(
            graph_data, test_ratio=test_ratio, seed=seed
        )
        return {
            "model": model,
            "structure_edge_index": structure_edge_index,
            "strategy": "topk",
            "defaults": {
                "top_k": int(source_hparams.get("top_k", 5)),
                "degree_penalty_alpha": float(source_hparams.get("degree_penalty_alpha", 0.35)),
                "exclude_existing_pairs": bool(source_hparams.get("exclude_existing_pairs", True)),
                "relative_threshold_factor": float(source_hparams.get("relative_threshold_factor", 0.90)),
                "max_candidates_per_task": int(source_hparams.get("max_candidates_per_task", 5)),
            },
        }

    if key == "gtn":
        from link_prediction.gtn_link_prediction import GTNLinkPrediction
        from link_prediction.utils import create_technique_wise_split

        model = GTNLinkPrediction(
            in_channels=in_channels,
            hidden_channels=int(source_hparams.get("hidden_channels", 16)),
            out_channels=int(source_hparams.get("out_channels", 64)),
            heads=int(source_hparams.get("heads", 2)),
            dropout=float(source_hparams.get("dropout", 0.3)),
            use_layer_norm=bool(source_hparams.get("use_layer_norm", True)),
            use_skip_connection=bool(source_hparams.get("use_skip_connection", True)),
            beta=bool(source_hparams.get("beta", True)),
            temperature_init=float(source_hparams.get("temperature_init", 2.0)),
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)
        structure_edge_index, _, _ = create_technique_wise_split(
            graph_data, test_ratio=test_ratio, seed=seed
        )
        return {
            "model": model,
            "structure_edge_index": structure_edge_index,
            "strategy": "relative",
            "defaults": {
                "relative_threshold_factor": float(source_hparams.get("relative_threshold_factor", 0.95)),
                "max_candidates_per_task": int(source_hparams.get("max_candidates_per_task", 3)),
                "degree_penalty_alpha": float(source_hparams.get("degree_penalty_alpha", 0.35)),
                "exclude_existing_pairs": bool(source_hparams.get("exclude_existing_pairs", True)),
                "top_k": int(source_hparams.get("top_k", 5)),
            },
        }

    if key == "node2vec":
        from link_prediction.node2vec_link_prediction import Node2VecLinkPredictor
        from link_prediction.utils import create_technique_wise_split

        state = torch.load(checkpoint_path, map_location="cpu")
        emb_shape = tuple(state["embeddings"].shape)
        model = Node2VecLinkPredictor(torch.zeros(emb_shape, dtype=torch.float32))
        model.load_state_dict(state)
        structure_edge_index, _, _ = create_technique_wise_split(
            graph_data, test_ratio=test_ratio, seed=seed
        )
        return {
            "model": model,
            "structure_edge_index": structure_edge_index,
            "strategy": "topk",
            "defaults": {
                "top_k": int(source_hparams.get("top_k", 5)),
                "degree_penalty_alpha": float(source_hparams.get("degree_penalty_alpha", 0.35)),
                "exclude_existing_pairs": bool(source_hparams.get("exclude_existing_pairs", True)),
                "relative_threshold_factor": float(source_hparams.get("relative_threshold_factor", 0.90)),
                "max_candidates_per_task": int(source_hparams.get("max_candidates_per_task", 5)),
            },
        }

    if key == "metapath2vec":
        from link_prediction.metapath2vec_link_prediction import Metapath2VecLinkPredictor
        from link_prediction.utils import create_technique_wise_split

        state = torch.load(checkpoint_path, map_location="cpu")
        emb_shape = tuple(state["embeddings"].shape)
        model = Metapath2VecLinkPredictor(
            num_nodes=int(emb_shape[0]),
            embedding_dim=int(emb_shape[1]),
        )
        model.load_state_dict(state)
        structure_edge_index, _, _ = create_technique_wise_split(
            graph_data, test_ratio=test_ratio, seed=seed
        )
        return {
            "model": model,
            "structure_edge_index": structure_edge_index,
            "strategy": "topk",
            "defaults": {
                "top_k": int(source_hparams.get("top_k", 5)),
                "degree_penalty_alpha": float(source_hparams.get("degree_penalty_alpha", 0.35)),
                "exclude_existing_pairs": bool(source_hparams.get("exclude_existing_pairs", True)),
                "relative_threshold_factor": float(source_hparams.get("relative_threshold_factor", 0.90)),
                "max_candidates_per_task": int(source_hparams.get("max_candidates_per_task", 5)),
            },
        }

    if key == "rgcn":
        from link_prediction.rgcn_link_prediction import (
            RGCNDiscoveryWrapper,
            RGCNLinkPrediction,
            build_relational_edge_index,
            filter_relational_edges,
        )

        full_edge_index, full_edge_type, num_relations, _ = build_relational_edge_index(graph_data)
        test_tech_indices = _build_test_technique_set(
            graph_data,
            seed=seed,
            test_ratio=test_ratio,
        )
        train_edge_index, train_edge_type = filter_relational_edges(
            full_edge_index,
            full_edge_type,
            test_tech_indices,
        )

        model = RGCNLinkPrediction(
            in_channels=in_channels,
            hidden_channels=int(source_hparams.get("hidden_channels", 64)),
            out_channels=int(source_hparams.get("out_channels", 64)),
            num_relations=num_relations,
            num_bases=int(source_hparams.get("num_bases", 4)),
            dropout=float(source_hparams.get("dropout", 0.2)),
            temperature_init=float(source_hparams.get("temperature_init", 2.0)),
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)
        wrapper = RGCNDiscoveryWrapper(model, train_edge_index, train_edge_type)

        return {
            "model": wrapper,
            "structure_edge_index": train_edge_index,
            "strategy": "topk",
            "defaults": {
                "top_k": int(source_hparams.get("top_k", 5)),
                "degree_penalty_alpha": float(source_hparams.get("degree_penalty_alpha", 0.35)),
                "exclude_existing_pairs": bool(source_hparams.get("exclude_existing_pairs", True)),
                "relative_threshold_factor": float(source_hparams.get("relative_threshold_factor", 0.90)),
                "max_candidates_per_task": int(source_hparams.get("max_candidates_per_task", 5)),
            },
        }

    raise NotImplementedError(
        f"Selected category '{model_category}' is not yet supported by this runner."
    )


def _build_test_technique_set(graph_data, seed, test_ratio):
    tech_indices = sorted(graph_data["node_type_to_indices"]["Technique"])
    rng = random.Random(seed)
    rng.shuffle(tech_indices)
    split_idx = int(len(tech_indices) * test_ratio)
    return set(tech_indices[:split_idx])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay discovery from a saved checkpoint and write novel_techniques JSON"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Path to a saved model checkpoint such as model_gat_seed_42.pt",
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

    model_category = infer_model_category(checkpoint_path, source_hparams)
    model_key = canonical_model_key(model_category)

    graph_path = args.graph_path or resolve_graph_path(source_hparams)
    graph_data = load_graph_data(graph_path)

    seed_from_fname = extract_checkpoint_seed(checkpoint_path)
    if seed_from_fname is not None:
        source_hparams["seed"] = seed_from_fname

    seed = int(source_hparams.get("seed", 42))
    # Create a model-level directory to collect JSON outputs for all seeds
    model_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else source_run_dir / f"seed_discovery__{model_key}"
    )
    model_dir.mkdir(parents=True, exist_ok=True)

    # Use a per-seed run dir for the discovery functions to write into,
    # then move the produced JSON into the shared model_dir with a seed-specific name.
    per_seed_dir = model_dir / f"{checkpoint_path.stem}"
    per_seed_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(str(per_seed_dir))

    logging.info("Seed-Scoped Discovery Replay")
    logging.info("Checkpoint: %s", checkpoint_path)
    logging.info("Source run dir: %s", source_run_dir)
    logging.info("Model category: %s", model_category)
    logging.info("Seed: %s", seed)
    logging.info("Graph: %s", graph_path)
    logging.info("Per-seed output dir: %s", per_seed_dir)
    logging.info("Model-level collection dir: %s", model_dir)

    bundle = load_model_bundle(
        model_category=model_category,
        graph_data=graph_data,
        source_hparams=source_hparams,
        checkpoint_path=str(checkpoint_path),
    )

    model = bundle["model"]
    structure_edge_index = bundle["structure_edge_index"]

    model.eval()
    with torch.no_grad():
        if model_key in {"node2vec", "metapath2vec"}:
            z = model.encode(None, None)
        else:
            z = model.encode(graph_data["node_features"], structure_edge_index)

    from link_prediction.utils import discover_novel_techniques, discover_novel_techniques_relative

    if bundle["strategy"] == "relative":
        discover_novel_techniques_relative(
            model=model,
            graph_data=graph_data,
            structure_edge_index=structure_edge_index,
            output_dir=str(per_seed_dir),
            model_name=model_key,
            relative_threshold_factor=float(bundle["defaults"]["relative_threshold_factor"]),
            max_candidates_per_task=int(bundle["defaults"]["max_candidates_per_task"]),
            degree_penalty_alpha=float(bundle["defaults"]["degree_penalty_alpha"]),
            exclude_existing_pairs=bool(bundle["defaults"]["exclude_existing_pairs"]),
        )
    else:
        discover_novel_techniques(
            model=model,
            graph_data=graph_data,
            structure_edge_index=structure_edge_index,
            output_dir=str(per_seed_dir),
            model_name=model_key,
            top_k=int(bundle["defaults"]["top_k"]),
            degree_penalty_alpha=float(bundle["defaults"]["degree_penalty_alpha"]),
            exclude_existing_pairs=bool(bundle["defaults"]["exclude_existing_pairs"]),
        )

    # discovery functions write into the per-seed dir; move the produced JSON
    # into the shared model_dir with a filename that includes the seed.
    produced = per_seed_dir / f"novel_techniques_{model_key}.json"
    if not produced.exists():
        # try to find any novel_techniques_*.json produced
        matches = list(per_seed_dir.glob("novel_techniques_*.json"))
        produced = matches[0] if matches else None

    if produced and produced.exists():
        output_path = model_dir / f"novel_techniques_{model_key}_seed_{seed}.json"
        shutil.move(str(produced), str(output_path))
    else:
        output_path = model_dir / f"novel_techniques_{model_key}_seed_{seed}.json"
    metadata = {
        "mode": "saved_checkpoint_seed_discovery",
        "checkpoint_path": str(checkpoint_path),
        "source_run_dir": str(source_run_dir),
        "graph_path": graph_path,
        "model_category": model_category,
        "model_key": model_key,
        "seed": seed,
        "output_json": str(output_path),
        "discovery_strategy": bundle["strategy"],
        "output_schema": "novel_techniques_standard",
    }
    with open(per_seed_dir / "discovery_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    run_hparams = {
        "mode": "saved_checkpoint_seed_discovery",
        "source_checkpoint": str(checkpoint_path),
        "source_run_dir": str(source_run_dir),
        "graph_path": graph_path,
        "model_type": model_category,
        "seed": seed,
        "prediction_strategy": bundle["strategy"],
        "degree_penalty_alpha": float(bundle["defaults"]["degree_penalty_alpha"]),
        "top_k": int(bundle["defaults"]["top_k"]),
        "relative_threshold_factor": float(bundle["defaults"]["relative_threshold_factor"]),
        "max_candidates_per_task": int(bundle["defaults"]["max_candidates_per_task"]),
        "exclude_existing_pairs": bool(bundle["defaults"]["exclude_existing_pairs"]),
    }
    log_hyperparameters(str(per_seed_dir), run_hparams)

    logging.info("Discovery replay completed successfully.")
    logging.info("JSON saved to %s", output_path)

    # Close logging handlers to release file locks on Windows
    for handler in logging.root.handlers[:]:
        handler.close()
        logging.root.removeHandler(handler)

    # Clean up the per-seed temporary directory
    if per_seed_dir.exists() and per_seed_dir != model_dir:
        shutil.rmtree(per_seed_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}")
        raise