"""
Random Component Link Prediction Baseline
=========================================
Produces exactly 5 random component suggestions per task.
This is a no-learning baseline for prompt synthesis comparison.

Output: link_pred_output/Random/run_<timestamp>/
"""

import json
import logging
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from link_prediction.utils import (
    create_technique_wise_split,
    get_component_degree_map,
    get_component_indices,
    load_graph_data,
    log_hyperparameters,
    select_graph_run,
    set_seed,
    setup_logging,
    setup_run_directory,
)


DEFAULTS = {
    "model_type": "Random",
    "seed": 42,
    "test_ratio": 0.2,
    "num_random_components": 5,
    "sampling_with_replacement": False,
}


def _build_node_metadata(graph_data):
    idx_to_node = graph_data["idx_to_node"]
    node_meta = {n["id"]: n for n in graph_data["node_metadata"]}
    return idx_to_node, node_meta


def _sample_components_for_task(component_indices, k, rng, with_replacement=False):
    if not component_indices:
        return []

    if with_replacement:
        return [rng.choice(component_indices) for _ in range(k)]

    if len(component_indices) >= k:
        return rng.sample(component_indices, k)

    # Fallback to replacement only when KG has fewer than k components.
    return [rng.choice(component_indices) for _ in range(k)]


def build_random_candidates(graph_data, structure_edge_index, hparams):
    task_indices = graph_data["node_type_to_indices"]["Task"]
    component_indices = get_component_indices(graph_data)
    idx_to_node, node_meta = _build_node_metadata(graph_data)
    component_degree = get_component_degree_map(graph_data, structure_edge_index)

    k = int(hparams["num_random_components"])
    rng = random.Random(int(hparams["seed"]))
    with_replacement = bool(hparams["sampling_with_replacement"])

    candidates = []
    for task_idx in task_indices:
        sampled = _sample_components_for_task(
            component_indices,
            k,
            rng,
            with_replacement=with_replacement,
        )

        suggested_components = []
        for comp_idx in sampled:
            comp_id = idx_to_node[comp_idx]
            comp_info = node_meta[comp_id]

            # Random confidence to preserve the existing score field contract.
            score = float(rng.random())
            suggested_components.append(
                {
                    "id": comp_id,
                    "name": comp_info["name"],
                    "type": comp_info["type"],
                    "score": score,
                    "raw_score": score,
                    "component_degree": int(component_degree.get(comp_idx, 0)),
                }
            )

        task_id = idx_to_node[task_idx]
        task_name = node_meta[task_id]["name"]
        confidence_avg = (
            sum(c["score"] for c in suggested_components) / len(suggested_components)
            if suggested_components
            else 0.0
        )

        candidates.append(
            {
                "task": task_name,
                "confidence_avg": confidence_avg,
                "suggested_components": suggested_components,
            }
        )

    candidates.sort(key=lambda x: x["confidence_avg"], reverse=True)
    return candidates


def main():
    hparams = dict(DEFAULTS)

    graph_path = select_graph_run()
    hparams["graph_path"] = graph_path

    run_dir, _ = setup_run_directory(model_type="Random")
    setup_logging(run_dir)

    logging.info("Started Random baseline run in: %s", run_dir)
    logging.info("Using graph: %s", graph_path)

    set_seed(hparams["seed"])
    log_hyperparameters(run_dir, hparams)

    graph_data = load_graph_data(graph_path)
    hparams["num_nodes"] = graph_data["num_nodes"]
    hparams["num_edges"] = graph_data["num_edges"]
    hparams["feature_dim"] = int(graph_data["node_features"].shape[1])
    log_hyperparameters(run_dir, hparams)

    structure_edge_index, _, _ = create_technique_wise_split(
        graph_data,
        test_ratio=hparams["test_ratio"],
        seed=hparams["seed"],
    )

    candidates = build_random_candidates(graph_data, structure_edge_index, hparams)

    output_path = os.path.join(run_dir, "novel_techniques_random.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2)

    # Save a lightweight artifact so the run is compatible with existing run discovery.
    model_path = os.path.join(run_dir, "model_random.pt")
    torch.save(
        {
            "baseline": "random_components",
            "seed": int(hparams["seed"]),
            "num_random_components": int(hparams["num_random_components"]),
            "sampling_with_replacement": bool(hparams["sampling_with_replacement"]),
        },
        model_path,
    )

    logging.info("Generated random suggestions for %d tasks.", len(candidates))
    logging.info("Saved to %s", output_path)
    logging.info("Model artifact saved to %s", model_path)
    logging.info("Random baseline run completed successfully.")


if __name__ == "__main__":
    main()
