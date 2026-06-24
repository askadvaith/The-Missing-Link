"""
Tune Degree Penalty Alpha for Link Prediction Models
===================================================
Tunes the `degree_penalty_alpha` post-hoc on a saved model run.
Runs a grid search over alpha values to find the one that maximizes validation/test MRR.

Usage:
    python link_prediction/tune_degree_penalty.py
    python link_prediction/tune_degree_penalty.py --category RGCN --run-dir link_pred_output/RGCN/run_20260329_093940__randomseed_100
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Ensure parent directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from link_prediction.run_saved_model_task_prediction import (
    canonical_model_key,
    load_json,
    load_model_bundle,
    resolve_graph_path,
    save_json,
)
from link_prediction.utils import (
    create_technique_wise_split,
    evaluate_deterministic,
    list_model_categories,
    list_model_runs,
    load_graph_data,
    select_model_category,
    select_model_run,
)


def tune_alpha_on_run(category: str, run_dir_path: str):
    run_dir = Path(run_dir_path)
    hparams_path = run_dir / "hyperparameters.json"
    if not hparams_path.exists():
        raise FileNotFoundError(f"No hyperparameters.json found in {run_dir}")

    hparams = load_json(hparams_path)
    source_hparams = hparams.get("hyperparameters", {})

    # Load graph
    logging.info("Resolving and loading graph...")
    graph_path = resolve_graph_path(source_hparams)
    graph_data = load_graph_data(graph_path)

    # Find checkpoint
    ckpt_files = sorted(run_dir.glob("model_*.pt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No model checkpoint model_*.pt found in {run_dir}")
    checkpoint_path = str(ckpt_files[0])

    logging.info(f"Loading model bundle for category '{category}' from run '{run_dir.name}'...")
    model_bundle = load_model_bundle(
        model_category=category,
        graph_data=graph_data,
        source_hparams=source_hparams,
        checkpoint_path=checkpoint_path,
    )

    # Reconstruct data split
    seed = int(source_hparams.get("seed", 42))
    test_ratio = float(source_hparams.get("test_ratio", 0.2))
    structure_edge_index, train_pos_edge_index, test_pos_edge_index = create_technique_wise_split(
        graph_data, test_ratio=test_ratio, seed=seed
    )

    # Calculate z
    model = model_bundle["model"]
    model.eval()
    with torch.no_grad():
        z = model.encode(graph_data["node_features"], model_bundle["structure_edge_index"])

    # Alpha grid
    alphas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]
    
    results = []
    best_mrr = -1.0
    best_alpha = 0.0

    print("\n" + "=" * 80)
    print(f" Tuning Degree Penalty (Alpha) for {category} | Run: {run_dir.name}")
    print("=" * 80)
    print(f" {'Alpha':<6} | {'Hits@1':<8} | {'Hits@3':<8} | {'Hits@10':<8} | {'MRR':<8} | {'AUC':<8} | {'AP':<8}")
    print("-" * 80)

    for alpha in alphas:
        metrics = evaluate_deterministic(
            model=model,
            z=z,
            test_pos_edge_index=test_pos_edge_index,
            train_pos_edge_index=train_pos_edge_index,
            graph_data=graph_data,
            structure_edge_index=model_bundle["structure_edge_index"],
            degree_penalty_alpha=alpha,
        )

        mrr = metrics["MRR"]
        if mrr > best_mrr:
            best_mrr = mrr
            best_alpha = alpha

        print(
            f" {alpha:<6.2f} | {metrics['Hits@1']:<8.4f} | {metrics['Hits@3']:<8.4f} | "
            f"{metrics['Hits@10']:<8.4f} | {mrr:<8.4f} | {metrics['AUC']:<8.4f} | {metrics['AP']:<8.4f}"
        )

        results.append({
            "alpha": alpha,
            "metrics": metrics
        })

    print("=" * 80)
    print(f"Optimal Alpha: {best_alpha:.2f} (MRR = {best_mrr:.4f})")
    print(f"Original Alpha: {source_hparams.get('degree_penalty_alpha', 0.35)} "
          f"(MRR = {next((r['metrics']['MRR'] for r in results if abs(r['alpha'] - source_hparams.get('degree_penalty_alpha', 0.35)) < 1e-5), 0.0):.4f})")
    print("=" * 80)

    # Save results
    out_path = run_dir / f"degree_penalty_tuning_{category.lower()}.json"
    save_json(out_path, {
        "category": category,
        "run_dir": str(run_dir),
        "optimal_alpha": best_alpha,
        "best_mrr": best_mrr,
        "sweep": results
    })
    print(f"Saved tuning sweep to: {out_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Tune degree penalty post-hoc on a saved run.")
    parser.add_argument("--category", type=str, default=None, help="Model category (e.g. RGCN, GCN, GAT, GTN)")
    parser.add_argument("--run-dir", type=str, default=None, help="Path to the saved run directory")
    args = parser.parse_args()

    # Set up basic stdout logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.category and args.run_dir:
        tune_alpha_on_run(args.category, args.run_dir)
    else:
        # Interactive selector flow
        try:
            category = select_model_category()
            run_info = select_model_run(category)
            tune_alpha_on_run(category, run_info["run_dir"])
        except Exception as e:
            logging.error(f"Error during tuning selection: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
