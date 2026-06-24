"""
Run Task-Scoped Link Prediction from a Saved Model Run
=======================================================
Interactive inference-only CLI flow:
  1) Select model category under link_pred_output/
  2) Select saved run folder
  3) Select target task

Then load the saved checkpoint, run prediction only for the selected task,
and persist output with the existing novel_techniques JSON schema.
"""

import json
import logging
import os
import random
import re
import sys
from pathlib import Path

# Ensure parent directory is on sys.path so `link_prediction` imports work
# when invoked directly as: python link_prediction/run_saved_model_task_prediction.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from link_prediction.gat_link_prediction import GATLinkPrediction
from link_prediction.gcn_link_prediction import GCNLinkPrediction
from link_prediction.gtn_link_prediction import GTNLinkPrediction
from link_prediction.metapath2vec_link_prediction import Metapath2VecLinkPredictor
from link_prediction.node2vec_link_prediction import Node2VecLinkPredictor
from link_prediction.rgcn_link_prediction import (
    RGCNDiscoveryWrapper,
    RGCNLinkPrediction,
    build_relational_edge_index,
    filter_relational_edges,
)
from link_prediction.utils import (
    create_technique_wise_split,
    discover_single_task_prediction,
    list_model_categories,
    load_graph_data,
    log_hyperparameters,
    select_graph_run,
    select_model_run,
    select_task_from_graph,
    set_seed,
    setup_logging,
    setup_run_directory,
    slugify,
)


def canonical_model_key(name: str):
    """Normalize category names for registry lookup."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def resolve_graph_path(source_hparams):
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

        logging.warning(
            "Saved graph_path was not found: %s. Falling back to interactive graph selection.",
            graph_path,
        )

    return select_graph_run()


def infer_model_name_token(source_run_dir, fallback):
    """Find model token from existing novel_techniques file name or checkpoint name."""
    run_dir = Path(source_run_dir)

    novel_files = sorted(run_dir.glob("novel_techniques_*.json"))
    if novel_files:
        stem = novel_files[0].stem
        token = stem.replace("novel_techniques_", "", 1).strip()
        if token:
            return token

    ckpt_files = sorted(run_dir.glob("model_*.pt"))
    if ckpt_files:
        stem = ckpt_files[0].stem
        token = stem.replace("model_", "", 1).strip()
        if token:
            return token

    return fallback


def _build_test_technique_set(graph_data, seed, test_ratio):
    tech_indices = sorted(graph_data["node_type_to_indices"]["Technique"])
    rng = random.Random(seed)
    rng.shuffle(tech_indices)
    split_idx = int(len(tech_indices) * test_ratio)
    return set(tech_indices[:split_idx])


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
    key = canonical_model_key(model_category)

    seed = int(source_hparams.get("seed", 42))
    test_ratio = float(source_hparams.get("test_ratio", 0.2))
    in_channels = int(graph_data["node_features"].shape[1])

    if key == "gat":
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


def select_supported_model_category(base_dir="link_pred_output"):
    """Allow selection from all category folders, then validate support."""
    categories = list_model_categories(base_dir=base_dir)
    if not categories:
        raise FileNotFoundError(f"No model categories found in {base_dir}/")

    print("\n" + "=" * 60)
    print("Available Model Categories")
    print("=" * 60)
    for idx, cat in enumerate(categories, 1):
        key = canonical_model_key(cat)
        supported = key in {
            "gat",
            "gcn",
            "gtn",
            "node2vec",
            "metapath2vec",
            "rgcn",
        }
        status = "supported" if supported else "not supported by loader yet"
        print(f"  [{idx}] {cat}  [{status}]")
    print("=" * 60)

    while True:
        try:
            choice = input(f"\nSelect model category (1-{len(categories)}): ").strip()
            idx = int(choice) - 1
            if idx < 0 or idx >= len(categories):
                print(f"Please enter a number between 1 and {len(categories)}")
                continue

            selected = categories[idx]
            if canonical_model_key(selected) not in {
                "gat",
                "gcn",
                "gtn",
                "node2vec",
                "metapath2vec",
                "rgcn",
            }:
                print(
                    f"Selected category '{selected}' is not supported yet. "
                    "Choose one of the supported categories listed above."
                )
                continue

            print(f"\n✓ Selected model category: {selected}\n")
            return selected
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\nOperation cancelled.")
            sys.exit(0)


def main():
    model_category = select_supported_model_category(base_dir="link_pred_output")
    source_run = select_model_run(model_category=model_category, base_dir="link_pred_output")

    source_hparams_record = load_json(source_run["hyperparameters_path"])
    source_hparams = source_hparams_record.get("hyperparameters", {})

    graph_path = resolve_graph_path(source_hparams)
    graph_data = load_graph_data(graph_path)

    selected_task_idx, selected_task_name = select_task_from_graph(graph_data)

    task_slug = slugify(selected_task_name)
    run_dir, timestamp = setup_run_directory(
        model_type=model_category,
        base_dir="link_pred_output",
        run_suffix=f"task_{task_slug}",
    )
    setup_logging(run_dir)

    logging.info("Started task-scoped inference run in: %s", run_dir)
    logging.info("Source run: %s", source_run["run_dir"])
    logging.info("Source checkpoint: %s", source_run["checkpoint_path"])
    logging.info("Graph path: %s", graph_path)
    logging.info("Selected task: %s", selected_task_name)

    seed = int(source_hparams.get("seed", 42))
    set_seed(seed)

    bundle = load_model_bundle(
        model_category=model_category,
        graph_data=graph_data,
        source_hparams=source_hparams,
        checkpoint_path=source_run["checkpoint_path"],
    )

    model_name_token = infer_model_name_token(
        source_run_dir=source_run["run_dir"],
        fallback=canonical_model_key(model_category),
    )

    _, output_path = discover_single_task_prediction(
        model=bundle["model"],
        graph_data=graph_data,
        structure_edge_index=bundle["structure_edge_index"],
        output_dir=run_dir,
        model_name=model_name_token,
        task_idx=selected_task_idx,
        strategy=bundle["strategy"],
        top_k=int(bundle["defaults"]["top_k"]),
        relative_threshold_factor=float(bundle["defaults"]["relative_threshold_factor"]),
        max_candidates_per_task=int(bundle["defaults"]["max_candidates_per_task"]),
        degree_penalty_alpha=float(bundle["defaults"]["degree_penalty_alpha"]),
        exclude_existing_pairs=bool(bundle["defaults"]["exclude_existing_pairs"]),
    )

    inference_meta = {
        "timestamp": timestamp,
        "mode": "saved_model_task_inference",
        "selected_model_category": model_category,
        "selected_source_run": source_run["run_dir"],
        "selected_checkpoint": source_run["checkpoint_path"],
        "graph_path": graph_path,
        "selected_task": selected_task_name,
        "selected_task_slug": task_slug,
        "output_json": output_path,
        "prediction_strategy": bundle["strategy"],
    }
    save_json(os.path.join(run_dir, "inference_metadata.json"), inference_meta)

    run_hparams = {
        "mode": "saved_model_task_inference",
        "source_model_category": model_category,
        "source_run": source_run["run_dir"],
        "source_checkpoint": source_run["checkpoint_path"],
        "graph_path": graph_path,
        "task_name": selected_task_name,
        "task_slug": task_slug,
        "prediction_strategy": bundle["strategy"],
        **bundle["defaults"],
    }
    log_hyperparameters(run_dir, run_hparams)

    logging.info("Inference metadata saved to %s", os.path.join(run_dir, "inference_metadata.json"))
    logging.info("Task-scoped prediction completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}")
        raise
