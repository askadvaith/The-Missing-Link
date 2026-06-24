"""
Shared Utilities for Link Prediction Scripts
=============================================
Common functions used by GCN, GAT, and GTN link prediction scripts.
Handles graph loading, data splitting, negative sampling, logging setup,
and hyperparameter tracking for reproducibility.
"""

import torch
import numpy as np
import random
import pickle
import json
import os
import sys
import datetime
import logging
import platform
import re
from collections import defaultdict
from pathlib import Path


def set_seed(seed=42):
    """Set all seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_graph_run():
    """
    Interactively select which graph to use from available runs.
    Returns the path to the selected graph_pytorch.pkl file.
    """
    graph_output_dir = Path("graph_output")

    available_graphs = []

    main_graph = graph_output_dir / "graph_pytorch.pkl"
    if main_graph.exists():
        available_graphs.append(("Main (latest)", str(main_graph)))

    for subdir in sorted(graph_output_dir.glob("graph_run_*")):
        if subdir.is_dir():
            graph_file = subdir / "graph_pytorch.pkl"
            if graph_file.exists():
                run_name = subdir.name
                available_graphs.append((run_name, str(graph_file)))

    if not available_graphs:
        raise FileNotFoundError("No graph_pytorch.pkl files found in graph_output/")

    if len(available_graphs) == 1:
        name, path = available_graphs[0]
        print(f"Only one graph found: {name}. Auto-selecting.")
        return path

    print("\n" + "=" * 60)
    print("Available Graph Runs:")
    print("=" * 60)
    for idx, (name, path) in enumerate(available_graphs, 1):
        print(f"  [{idx}] {name}")
        print(f"      Path: {path}")
    print("=" * 60)

    while True:
        try:
            choice = input(f"\nSelect graph run (1-{len(available_graphs)}) [1]: ").strip()
            if not choice:
                choice = "1"
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(available_graphs):
                selected_name, selected_path = available_graphs[choice_idx]
                print(f"\n✓ Selected: {selected_name}")
                print(f"  Loading from: {selected_path}\n")
                return selected_path
            else:
                print(f"Please enter a number between 1 and {len(available_graphs)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\nOperation cancelled.")
            sys.exit(0)


def load_graph_data(filepath: str = "graph_output/graph_pytorch.pkl"):
    """Load pickled graph data from disk."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def setup_run_directory(model_type: str, base_dir: str = "link_pred_output", run_suffix: str = None):
    """
    Create a timestamped run directory under link_pred_output/<model_type>/.
    Returns (run_dir, timestamp).
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    if run_suffix:
        run_name = f"{run_name}__{run_suffix}"
    run_dir = os.path.join(base_dir, model_type, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, timestamp


def slugify(value: str):
    """Convert user-facing labels into filesystem-safe lowercase slugs."""
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "task"


def list_model_categories(base_dir: str = "link_pred_output"):
    """Return all model category folders directly under link_pred_output/."""
    base = Path(base_dir)
    if not base.exists() or not base.is_dir():
        return []
    return sorted([p.name for p in base.iterdir() if p.is_dir()])


def list_model_runs(model_category: str, base_dir: str = "link_pred_output"):
    """
    Return run folders under a model category with basic metadata.

    A valid run folder must contain hyperparameters.json and at least one
    model_*.pt checkpoint file.
    """
    model_dir = Path(base_dir) / model_category
    if not model_dir.exists() or not model_dir.is_dir():
        return []

    runs = []
    for run_dir in sorted(model_dir.glob("run_*"), reverse=True):
        if not run_dir.is_dir():
            continue

        hparams_path = run_dir / "hyperparameters.json"
        ckpt_paths = sorted(run_dir.glob("model_*.pt"))
        if not hparams_path.exists() or not ckpt_paths:
            continue

        runs.append(
            {
                "run_name": run_dir.name,
                "run_dir": str(run_dir),
                "hyperparameters_path": str(hparams_path),
                "checkpoint_path": str(ckpt_paths[0]),
            }
        )

    return runs


def _choose_from_numbered_list(title, options):
    """Interactive numbered selector returning (index, selected_value)."""
    if not options:
        raise ValueError(f"No options available for: {title}")

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    for idx, item in enumerate(options, 1):
        print(f"  [{idx}] {item}")
    print("=" * 60)

    while True:
        try:
            choice = input(f"\nSelect option (1-{len(options)}): ").strip()
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(options):
                return choice_idx, options[choice_idx]
            print(f"Please enter a number between 1 and {len(options)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\nOperation cancelled.")
            sys.exit(0)


def select_model_category(base_dir: str = "link_pred_output"):
    """Interactively select model category from link_pred_output subfolders."""
    categories = list_model_categories(base_dir=base_dir)
    if not categories:
        raise FileNotFoundError(f"No model categories found in {base_dir}/")

    _, selected = _choose_from_numbered_list("Available Model Categories", categories)
    print(f"\n✓ Selected model category: {selected}\n")
    return selected


def select_model_run(model_category: str, base_dir: str = "link_pred_output"):
    """Interactively select a valid saved run for the chosen model category."""
    runs = list_model_runs(model_category=model_category, base_dir=base_dir)
    if not runs:
        raise FileNotFoundError(
            f"No valid runs found under {base_dir}/{model_category}/"
        )

    options = [
        f"{r['run_name']}  (ckpt: {Path(r['checkpoint_path']).name})"
        for r in runs
    ]
    idx, _ = _choose_from_numbered_list(
        f"Available Runs for {model_category}", options
    )
    selected = runs[idx]

    print(f"\n✓ Selected run: {selected['run_name']}")
    print(f"  Run dir: {selected['run_dir']}")
    print(f"  Checkpoint: {selected['checkpoint_path']}\n")
    return selected


def select_task_from_graph(graph_data):
    """
    Select one task from graph data with both typed search and numbered list.

    Returns (task_idx, task_name).
    """
    idx_to_node = graph_data["idx_to_node"]
    node_meta = {n["id"]: n for n in graph_data["node_metadata"]}
    task_indices = graph_data["node_type_to_indices"].get("Task", [])

    if not task_indices:
        raise ValueError("No Task nodes found in graph data.")

    task_entries = []
    for task_idx in task_indices:
        task_id = idx_to_node[task_idx]
        task_name = node_meta[task_id]["name"]
        task_entries.append((task_idx, task_name))

    task_entries = sorted(task_entries, key=lambda x: x[1].lower())

    print("\nTask Selection")
    print("Enter optional text to filter tasks, then choose by number.")
    search = input("Task search text (press Enter to list all): ").strip().lower()

    if search:
        filtered = [t for t in task_entries if search in t[1].lower()]
        if not filtered:
            print("No matches found for that search. Showing all tasks instead.")
            filtered = task_entries
    else:
        filtered = task_entries

    options = [name for _, name in filtered]
    idx, selected_name = _choose_from_numbered_list("Available Tasks", options)
    selected_task_idx = filtered[idx][0]

    print(f"\n✓ Selected task: {selected_name}\n")
    return selected_task_idx, selected_name


def _predict_for_single_task_topk(
    model,
    graph_data,
    z,
    structure_edge_index,
    task_idx,
    top_k,
    degree_penalty_alpha,
    exclude_existing_pairs,
):
    """Compute top-k suggestions for one task in the standard output schema."""
    comp_types = {
        "AlgorithmicComponent": graph_data["node_type_to_indices"]["AlgorithmicComponent"],
        "PromptComponent": graph_data["node_type_to_indices"]["PromptComponent"],
        "DataFlow": graph_data["node_type_to_indices"]["DataFlow"],
    }

    idx_to_node = graph_data["idx_to_node"]
    node_meta = {n["id"]: n for n in graph_data["node_metadata"]}
    observed_pairs = _build_existing_task_component_pairs(graph_data) if exclude_existing_pairs else set()

    components_found = []

    comp_degree_map = get_component_degree_map(graph_data, structure_edge_index)

    for _, c_indices in comp_types.items():
        if not c_indices:
            continue

        src = torch.full((len(c_indices),), task_idx, dtype=torch.long)
        dst = torch.tensor(c_indices, dtype=torch.long)
        edge_batch = torch.stack([src, dst], dim=0)

        raw_scores = model.decode(z, edge_batch).sigmoid()
        scores = apply_degree_penalty(
            raw_scores,
            c_indices,
            comp_degree_map,
            degree_penalty_alpha,
        )

        k = min(top_k, len(c_indices))
        top_vals, top_idxs = torch.topk(scores, k)

        for score, idx_in_batch in zip(top_vals.tolist(), top_idxs.tolist()):
            original_node_idx = c_indices[idx_in_batch]
            if exclude_existing_pairs and (task_idx, original_node_idx) in observed_pairs:
                continue

            c_id = idx_to_node[original_node_idx]
            c_info = node_meta[c_id]

            components_found.append(
                {
                    "id": c_id,
                    "name": c_info["name"],
                    "type": c_info["type"],
                    "score": float(score),
                    "raw_score": float(raw_scores[idx_in_batch]),
                    "component_degree": int(comp_degree_map.get(original_node_idx, 0)),
                }
            )

    components_found.sort(key=lambda x: x["score"], reverse=True)

    task_id = idx_to_node[task_idx]
    task_name = node_meta[task_id]["name"]

    return {
        "task": task_name,
        "confidence_avg": (
            sum(c["score"] for c in components_found) / len(components_found)
            if components_found
            else 0.0
        ),
        "suggested_components": components_found,
    }


def _predict_for_single_task_relative(
    model,
    graph_data,
    z,
    structure_edge_index,
    task_idx,
    relative_threshold_factor,
    max_candidates_per_task,
    degree_penalty_alpha,
    exclude_existing_pairs,
):
    """Compute relative-threshold suggestions for one task in the standard schema."""
    all_comp_indices = (
        graph_data["node_type_to_indices"]["AlgorithmicComponent"]
        + graph_data["node_type_to_indices"]["PromptComponent"]
        + graph_data["node_type_to_indices"]["DataFlow"]
    )

    idx_to_node = graph_data["idx_to_node"]
    node_meta = {n["id"]: n for n in graph_data["node_metadata"]}
    observed_pairs = _build_existing_task_component_pairs(graph_data) if exclude_existing_pairs else set()

    comp_degree_map = get_component_degree_map(graph_data, structure_edge_index)

    src = torch.full((len(all_comp_indices),), task_idx, dtype=torch.long)
    dst = torch.tensor(all_comp_indices, dtype=torch.long)
    edge_batch = torch.stack([src, dst], dim=0)

    raw_scores = model.decode(z, edge_batch).sigmoid()
    scores = apply_degree_penalty(
        raw_scores,
        all_comp_indices,
        comp_degree_map,
        degree_penalty_alpha,
    )

    max_score = scores.max().item()
    cutoff = max_score * relative_threshold_factor

    scored_comps = []
    for i, score_val in enumerate(scores.tolist()):
        if score_val < cutoff:
            continue
        original_idx = all_comp_indices[i]
        if exclude_existing_pairs and (task_idx, original_idx) in observed_pairs:
            continue
        scored_comps.append((score_val, original_idx, float(raw_scores[i])))

    scored_comps.sort(key=lambda x: x[0], reverse=True)
    scored_comps = scored_comps[:max_candidates_per_task]

    components_found = []
    for score, node_idx, raw_score in scored_comps:
        c_id = idx_to_node[node_idx]
        c_info = node_meta[c_id]
        components_found.append(
            {
                "id": c_id,
                "name": c_info["name"],
                "type": c_info["type"],
                "score": score,
                "raw_score": raw_score,
                "component_degree": int(comp_degree_map.get(node_idx, 0)),
            }
        )

    task_id = idx_to_node[task_idx]
    task_name = node_meta[task_id]["name"]

    return {
        "task": task_name,
        "max_score_in_task": max_score,
        "confidence_avg": (
            sum(c["score"] for c in components_found) / len(components_found)
            if components_found
            else 0.0
        ),
        "suggested_components": components_found,
    }


def discover_single_task_prediction(
    model,
    graph_data,
    structure_edge_index,
    output_dir,
    model_name,
    task_idx,
    strategy,
    top_k=5,
    relative_threshold_factor=0.90,
    max_candidates_per_task=5,
    degree_penalty_alpha=0.35,
    exclude_existing_pairs=True,
):
    """
    Run prediction for exactly one task and save output with the existing
    novel_techniques JSON schema as a one-item array.
    """
    model.eval()
    with torch.no_grad():
        z = model.encode(graph_data["node_features"], structure_edge_index)

    if strategy == "relative":
        task_result = _predict_for_single_task_relative(
            model=model,
            graph_data=graph_data,
            z=z,
            structure_edge_index=structure_edge_index,
            task_idx=task_idx,
            relative_threshold_factor=relative_threshold_factor,
            max_candidates_per_task=max_candidates_per_task,
            degree_penalty_alpha=degree_penalty_alpha,
            exclude_existing_pairs=exclude_existing_pairs,
        )
    elif strategy == "topk":
        task_result = _predict_for_single_task_topk(
            model=model,
            graph_data=graph_data,
            z=z,
            structure_edge_index=structure_edge_index,
            task_idx=task_idx,
            top_k=top_k,
            degree_penalty_alpha=degree_penalty_alpha,
            exclude_existing_pairs=exclude_existing_pairs,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    results = [task_result]
    output_path = os.path.join(output_dir, f"novel_techniques_{model_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logging.info("Task-scoped prediction generated for 1 task.")
    logging.info(f"Saved to {output_path}")

    return results, output_path


def setup_logging(run_dir: str):
    """
    Configure logging to file and console for a run.
    Clears any existing handlers to avoid duplicates.
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_path = os.path.join(run_dir, "run.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )


def log_hyperparameters(run_dir: str, hyperparams: dict):
    """
    Save all hyperparameters and environment info to a JSON file for reproducibility.
    """
    env_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
    }

    try:
        import torch_geometric
        env_info["torch_geometric_version"] = torch_geometric.__version__
    except ImportError:
        env_info["torch_geometric_version"] = "unknown"

    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "environment": env_info,
        "hyperparameters": hyperparams,
    }

    path = os.path.join(run_dir, "hyperparameters.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)

    logging.info(f"Hyperparameters saved to {path}")
    return record


def create_technique_wise_split(graph_data, test_ratio=0.2, seed=42):
    """
    Splits the data by TECHNIQUE to prevent leakage.
    Hides 'Test Techniques' from the training graph entirely.
    Returns:
        train_structure_edge_index: The graph structure (technique-component edges)
        train_pos_edge_index: Inferred (Task, Component) edges for training
        test_pos_edge_index: Inferred (Task, Component) edges for testing
    """
    tech_indices = graph_data["node_type_to_indices"]["Technique"]

    rng = random.Random(seed)
    tech_indices = sorted(tech_indices)
    rng.shuffle(tech_indices)

    split_idx = int(len(tech_indices) * test_ratio)
    test_tech_indices = set(tech_indices[:split_idx])
    train_tech_indices = set(tech_indices[split_idx:])

    logging.info("Technique Split:")
    logging.info(f"  Train Techniques: {len(train_tech_indices)}")
    logging.info(f"  Test Techniques:  {len(test_tech_indices)}")

    edge_index_tensors = graph_data["edge_index_tensors"]

    tech_to_tasks = defaultdict(set)
    tech_to_comps = defaultdict(set)

    if "excels_at" in edge_index_tensors:
        edge_index = edge_index_tensors["excels_at"]
        for i in range(edge_index.shape[1]):
            src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
            if src in train_tech_indices or src in test_tech_indices:
                tech_to_tasks[src].add(tgt)

    for et in ["uses_algorithm", "uses_prompt", "uses_data_flow"]:
        if et in edge_index_tensors:
            edge_index = edge_index_tensors[et]
            for i in range(edge_index.shape[1]):
                src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
                if src in train_tech_indices or src in test_tech_indices:
                    tech_to_comps[src].add(tgt)

    def fast_create_edges(tech_set):
        edges = set()
        for tech in tech_set:
            tasks = tech_to_tasks.get(tech, [])
            comps = tech_to_comps.get(tech, [])
            for t in tasks:
                for c in comps:
                    edges.add((t, c))
        return list(edges)

    train_pos_edges = set(fast_create_edges(train_tech_indices))
    test_pos_edges = set(fast_create_edges(test_tech_indices))

    overlap = train_pos_edges & test_pos_edges
    if overlap:
        logging.warning(
            "Detected %d overlapping (Task, Component) pairs between train/test; "
            "removing from test to prevent supervision leakage.",
            len(overlap),
        )
        test_pos_edges -= overlap

    train_pos_edges = sorted(train_pos_edges)
    test_pos_edges = sorted(test_pos_edges)

    train_edge_index = torch.tensor(train_pos_edges, dtype=torch.long).t()
    test_edge_index = torch.tensor(test_pos_edges, dtype=torch.long).t()

    if train_edge_index.numel() == 0:
        logging.warning("No training edges generated! Check graph connectivity.")
        train_edge_index = torch.empty((2, 0), dtype=torch.long)
    if test_edge_index.numel() == 0:
        logging.warning("No testing edges generated!")
        test_edge_index = torch.empty((2, 0), dtype=torch.long)

    logging.info(
        f"  Train (Task-Comp) Edges: {train_edge_index.shape[1] if train_edge_index.numel() > 0 else 0}"
    )
    logging.info(
        f"  Test (Task-Comp) Edges:  {test_edge_index.shape[1] if test_edge_index.numel() > 0 else 0}"
    )

    train_msg_edges_list = []

    def is_safe(src, tgt):
        return (src not in test_tech_indices) and (tgt not in test_tech_indices)

    for et, edge_index in edge_index_tensors.items():
        mask = [
            is_safe(edge_index[0, i].item(), edge_index[1, i].item())
            for i in range(edge_index.shape[1])
        ]
        mask_tensor = torch.tensor(mask, dtype=torch.bool)
        filtered_edges = edge_index[:, mask_tensor]
        train_msg_edges_list.append(filtered_edges)

    train_structure_edge_index = torch.cat(train_msg_edges_list, dim=1)

    return train_structure_edge_index, train_edge_index, test_edge_index


def get_structured_negatives(pos_edge_index, all_comp_indices, num_neg=1):
    """
    For each (Task, Comp) edge, sample 'num_neg' (Task, RandomComp) edges.
    Efficiently constructed by corrupting tails.
    """
    num_edges = pos_edge_index.shape[1]
    src = pos_edge_index[0].repeat_interleave(num_neg)
    dst = torch.tensor(
        random.choices(all_comp_indices, k=num_edges * num_neg), dtype=torch.long
    )
    neg_edge_index = torch.stack([src, dst], dim=0)
    return neg_edge_index


def get_component_indices(graph_data):
    """Return concatenated component indices for negative sampling."""
    return (
        graph_data["node_type_to_indices"]["AlgorithmicComponent"]
        + graph_data["node_type_to_indices"]["PromptComponent"]
        + graph_data["node_type_to_indices"]["DataFlow"]
    )


def _build_existing_task_component_pairs(graph_data):
    """
    Build all already-observed implicit (Task, Component) pairs from technique
    edges in the current graph. Discovery can filter these out to keep only
    novel suggestions.
    """
    edge_index_tensors = graph_data["edge_index_tensors"]

    tech_to_tasks = defaultdict(set)
    tech_to_comps = defaultdict(set)

    if "excels_at" in edge_index_tensors:
        edge_index = edge_index_tensors["excels_at"]
        for i in range(edge_index.shape[1]):
            src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
            tech_to_tasks[src].add(tgt)

    for et in ["uses_algorithm", "uses_prompt", "uses_data_flow"]:
        if et not in edge_index_tensors:
            continue
        edge_index = edge_index_tensors[et]
        for i in range(edge_index.shape[1]):
            src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
            tech_to_comps[src].add(tgt)

    observed_pairs = set()
    for tech_idx, tasks in tech_to_tasks.items():
        comps = tech_to_comps.get(tech_idx, set())
        for task_idx in tasks:
            for comp_idx in comps:
                observed_pairs.add((task_idx, comp_idx))

    return observed_pairs


def get_component_degree_map(graph_data, structure_edge_index):
    """
    Compute structural degree priors for component nodes from the message-passing
    graph. Used to penalize very high-degree components during evaluation 
    and discovery ranking.
    """
    component_indices = set(get_component_indices(graph_data))
    degree = defaultdict(int)

    if structure_edge_index.numel() == 0:
        return degree

    num_edges = structure_edge_index.shape[1]
    for i in range(num_edges):
        src = structure_edge_index[0, i].item()
        dst = structure_edge_index[1, i].item()
        if src in component_indices:
            degree[src] += 1
        if dst in component_indices:
            degree[dst] += 1

    return degree


def apply_degree_penalty(raw_scores, candidate_indices, degree_map, alpha):
    """
    Penalize high-degree candidates by dividing by 1 + alpha * log(1 + degree).
    `candidate_indices` can be a list or a PyTorch tensor corresponding element-wise
    to `raw_scores`.
    """
    if alpha <= 0:
        return raw_scores

    if isinstance(candidate_indices, torch.Tensor):
        candidates_list = candidate_indices.tolist()
    else:
        candidates_list = candidate_indices

    penalties = []
    for idx in candidates_list:
        deg = degree_map.get(idx, 0)
        penalties.append(1.0 + alpha * np.log1p(deg))

    penalty_t = torch.tensor(penalties, dtype=raw_scores.dtype, device=raw_scores.device)
    return raw_scores / penalty_t



def evaluate_deterministic(
    model,
    z,
    test_pos_edge_index: torch.Tensor,
    train_pos_edge_index: torch.Tensor,
    graph_data: dict,
    structure_edge_index: torch.Tensor,
    degree_penalty_alpha: float = 0.35,
) -> dict:
    """
    Deterministic link prediction evaluation with exhaustive negative ranking.

    For every positive test edge (Task_i → Component_j), this function scores
    Component_j against ALL component nodes that are NOT a known positive for
    Task_i (i.e. not in either the training or test positive edge sets).

    This eliminates all random sampling noise from evaluation: the same model
    and split will always produce the same Hits@K, MRR, AUC, and AP.

    Returns:
        dict of metric_name → value (same schema as evaluate_link_prediction).
    """
    from collections import defaultdict
    from utils.metrics import compute_hits_at_k, compute_mrr, compute_auc, compute_ap

    all_comp_indices = get_component_indices(graph_data)
    comp_degree_map = get_component_degree_map(graph_data, structure_edge_index)

    # Build per-task known-positive sets (train + test) to exclude from negatives
    known_pos: dict[int, set] = defaultdict(set)
    for edge_index in [train_pos_edge_index, test_pos_edge_index]:
        if edge_index.numel() == 0:
            continue
        for i in range(edge_index.shape[1]):
            task_i = edge_index[0, i].item()
            comp_i = edge_index[1, i].item()
            known_pos[task_i].add(comp_i)

    # Per-positive-edge: score pos vs all valid negatives
    all_pos_scores_flat: list[float] = []
    all_neg_scores_flat: list[float] = []

    for i in range(test_pos_edge_index.shape[1]):
        task_idx = test_pos_edge_index[0, i].item()
        comp_idx = test_pos_edge_index[1, i].item()

        # Exclude all known positives for this task
        neg_comp_indices = [c for c in all_comp_indices if c not in known_pos[task_idx]]
        if not neg_comp_indices:
            continue

        # Score positive
        pos_edge = torch.tensor([[task_idx], [comp_idx]], dtype=torch.long)
        pos_raw = model.decode(z, pos_edge).sigmoid()
        pos_score = apply_degree_penalty(pos_raw, [comp_idx], comp_degree_map, degree_penalty_alpha)

        # Score all negatives
        src_neg = torch.full((len(neg_comp_indices),), task_idx, dtype=torch.long)
        dst_neg = torch.tensor(neg_comp_indices, dtype=torch.long)
        neg_edge = torch.stack([src_neg, dst_neg], dim=0)
        neg_raw = model.decode(z, neg_edge).sigmoid()
        neg_scores = apply_degree_penalty(neg_raw, neg_comp_indices, comp_degree_map, degree_penalty_alpha)

        all_pos_scores_flat.append(pos_score.item())
        all_neg_scores_flat.extend(neg_scores.tolist())

    if not all_pos_scores_flat:
        return {"Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0, "MRR": 0.0, "AUC": 0.0, "AP": 0.0, "PairwiseAcc": 0.0}

    pos_t = torch.tensor(all_pos_scores_flat)
    neg_t = torch.tensor(all_neg_scores_flat)

    results = {}
    for k in [1, 3, 10]:
        results[f"Hits@{k}"] = compute_hits_at_k(pos_t, neg_t, k)
    results["MRR"] = compute_mrr(pos_t, neg_t)
    results["AUC"] = compute_auc(pos_t, neg_t)
    results["AP"] = compute_ap(pos_t, neg_t)
    results["PairwiseAcc"] = (pos_t.mean() > neg_t.mean()).float().item()

    return results


def discover_novel_techniques(
    model,
    graph_data,
    structure_edge_index,
    output_dir,
    model_name="model",
    top_k=3,
    degree_penalty_alpha=0.35,
    exclude_existing_pairs=True,
):
    """
    Predicts components for Tasks and groups them into 'Novel Technique Candidates'.
    Uses Top-K strategy per component type.
    """
    logging.info(
        f"Running Discovery (Top-{top_k} per component type, degree_penalty_alpha={degree_penalty_alpha})..."
    )

    model.eval()
    with torch.no_grad():
        z = model.encode(graph_data["node_features"], structure_edge_index)

    task_indices = graph_data["node_type_to_indices"]["Task"]

    comp_types = {
        "AlgorithmicComponent": graph_data["node_type_to_indices"]["AlgorithmicComponent"],
        "PromptComponent": graph_data["node_type_to_indices"]["PromptComponent"],
        "DataFlow": graph_data["node_type_to_indices"]["DataFlow"],
    }

    idx_to_node = graph_data["idx_to_node"]
    node_meta = {n["id"]: n for n in graph_data["node_metadata"]}
    observed_pairs = _build_existing_task_component_pairs(graph_data) if exclude_existing_pairs else set()
    comp_degree_map = get_component_degree_map(graph_data, structure_edge_index)

    candidates = []
    all_scores = []

    for task_idx in task_indices:
        task_id = idx_to_node[task_idx]
        task_name = node_meta[task_id]["name"]

        components_found = []

        for type_name, c_indices in comp_types.items():
            if not c_indices:
                continue

            src = torch.full((len(c_indices),), task_idx, dtype=torch.long)
            dst = torch.tensor(c_indices, dtype=torch.long)
            edge_batch = torch.stack([src, dst], dim=0)

            raw_scores = model.decode(z, edge_batch).sigmoid()
            scores = apply_degree_penalty(
                raw_scores,
                c_indices,
                comp_degree_map,
                degree_penalty_alpha,
            )
            all_scores.extend(scores.tolist())

            k = min(top_k, len(c_indices))
            top_vals, top_idxs = torch.topk(scores, k)

            for score, idx_in_batch in zip(top_vals.tolist(), top_idxs.tolist()):
                original_node_idx = c_indices[idx_in_batch]
                if exclude_existing_pairs and (task_idx, original_node_idx) in observed_pairs:
                    continue
                c_id = idx_to_node[original_node_idx]
                c_info = node_meta[c_id]

                components_found.append(
                    {
                        "id": c_id,
                        "name": c_info["name"],
                        "type": c_info["type"],
                        "score": float(score),
                        "raw_score": float(raw_scores[idx_in_batch]),
                        "component_degree": int(comp_degree_map.get(original_node_idx, 0)),
                    }
                )

        components_found.sort(key=lambda x: x["score"], reverse=True)

        candidates.append(
            {
                "task": task_name,
                "confidence_avg": (
                    sum(c["score"] for c in components_found) / len(components_found)
                    if components_found
                    else 0.0
                ),
                "suggested_components": components_found,
            }
        )

    scores_t = torch.tensor(all_scores)
    logging.info(
        f"Score Stats -- Mean: {scores_t.mean():.4f}, Std: {scores_t.std():.4f}, "
        f"Min: {scores_t.min():.4f}, Max: {scores_t.max():.4f}"
    )

    candidates.sort(key=lambda x: x["confidence_avg"], reverse=True)

    output_path = os.path.join(output_dir, f"novel_techniques_{model_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2)

    logging.info(f"Found {len(candidates)} candidates.")
    logging.info(f"Saved to {output_path}")

    return candidates


def discover_novel_techniques_relative(
    model,
    graph_data,
    structure_edge_index,
    output_dir,
    model_name="model",
    relative_threshold_factor=0.90,
    max_candidates_per_task=5,
    degree_penalty_alpha=0.35,
    exclude_existing_pairs=True,
):
    """
    Predicts components for Tasks using Dynamic Relative Thresholding.
    Scores ALL components together for each task and keeps only those
    within a certain fraction of the *best* score for that task,
    with a hard cap on candidates per task.
    """
    logging.info(
        "Running Discovery (Dynamic Relative Thresholding, degree_penalty_alpha=%s)...",
        degree_penalty_alpha,
    )

    model.eval()
    with torch.no_grad():
        z = model.encode(graph_data["node_features"], structure_edge_index)

    task_indices = graph_data["node_type_to_indices"]["Task"]

    all_comp_indices = (
        graph_data["node_type_to_indices"]["AlgorithmicComponent"]
        + graph_data["node_type_to_indices"]["PromptComponent"]
        + graph_data["node_type_to_indices"]["DataFlow"]
    )

    idx_to_node = graph_data["idx_to_node"]
    node_meta = {n["id"]: n for n in graph_data["node_metadata"]}
    observed_pairs = _build_existing_task_component_pairs(graph_data) if exclude_existing_pairs else set()
    comp_degree_map = get_component_degree_map(graph_data, structure_edge_index)

    candidates = []
    all_scores = []

    for task_idx in task_indices:
        task_id = idx_to_node[task_idx]

        # Score ALL components for this task
        src = torch.full((len(all_comp_indices),), task_idx, dtype=torch.long)
        dst = torch.tensor(all_comp_indices, dtype=torch.long)
        edge_batch = torch.stack([src, dst], dim=0)

        raw_scores = model.decode(z, edge_batch).sigmoid()
        scores = apply_degree_penalty(
            raw_scores,
            all_comp_indices,
            comp_degree_map,
            degree_penalty_alpha,
        )
        all_scores.extend(scores.tolist())

        # Dynamic threshold: keep candidates within factor of best
        max_score = scores.max().item()
        cutoff = max_score * relative_threshold_factor

        scored_comps = []
        for i, score_val in enumerate(scores.tolist()):
            if score_val >= cutoff:
                original_idx = all_comp_indices[i]
                if exclude_existing_pairs and (task_idx, original_idx) in observed_pairs:
                    continue
                scored_comps.append((score_val, original_idx, float(raw_scores[i])))

        scored_comps.sort(key=lambda x: x[0], reverse=True)
        scored_comps = scored_comps[:max_candidates_per_task]

        components_found = []
        for score, node_idx, raw_score in scored_comps:
            c_id = idx_to_node[node_idx]
            c_info = node_meta[c_id]
            components_found.append(
                {
                    "id": c_id,
                    "name": c_info["name"],
                    "type": c_info["type"],
                    "score": score,
                    "raw_score": raw_score,
                    "component_degree": int(comp_degree_map.get(node_idx, 0)),
                }
            )

        if components_found:
            candidates.append(
                {
                    "task": node_meta[task_id]["name"],
                    "max_score_in_task": max_score,
                    "confidence_avg": (
                        sum(c["score"] for c in components_found)
                        / len(components_found)
                    ),
                    "suggested_components": components_found,
                }
            )

    scores_t = torch.tensor(all_scores)
    logging.info(
        f"Score Stats -- Mean: {scores_t.mean():.4f}, Std: {scores_t.std():.4f}, "
        f"Min: {scores_t.min():.4f}, Max: {scores_t.max():.4f}"
    )

    candidates.sort(key=lambda x: x["confidence_avg"], reverse=True)

    output_path = os.path.join(output_dir, f"novel_techniques_{model_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2)

    logging.info(f"Found suggestions for {len(candidates)} tasks.")
    logging.info(f"Saved to {output_path}")

    return candidates
