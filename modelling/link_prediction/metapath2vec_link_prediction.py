"""
Metapath2Vec Link Prediction on Prompt Engineering Knowledge Graph
====================================================================
Uses Metapath2Vec random-walk embeddings guided by heterogeneous metapaths
that are aligned with the (Task, Component) link prediction target.

Since the link prediction task predicts (Task → Component) edges — which are
inferred through Techniques — each metapath follows the full bridge path:

  Technique → excels_at → Task → (rev) → Technique → uses_X → Component → (rev) → Technique

One MetaPath2Vec model is trained per component type, and the resulting
embeddings are averaged to produce a single, comprehensive representation
for every node.

Metapaths (target-aligned):
  1. Technique ↔ Task ↔ Technique ↔ AlgorithmicComponent ↔ Technique
  2. Technique ↔ Task ↔ Technique ↔ PromptComponent     ↔ Technique
  3. Technique ↔ Task ↔ Technique ↔ DataFlow             ↔ Technique

Output: link_pred_output/Metapath2Vec/run_<timestamp>/
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
import logging
from collections import defaultdict

from torch_geometric.data import HeteroData
from torch_geometric.nn import MetaPath2Vec as PyGMetaPath2Vec

from link_prediction.utils import (
    set_seed,
    select_graph_run,
    load_graph_data,
    setup_run_directory,
    setup_logging,
    log_hyperparameters,
    create_technique_wise_split,
    get_structured_negatives,
    get_component_indices,
    discover_novel_techniques,
    get_component_degree_map,
    apply_degree_penalty,
    evaluate_deterministic,
)
from utils.metrics import evaluate_link_prediction, log_metrics, save_metrics

# ──────────────────────────────────────────────────────────────────────
# Default hyperparameters
# ──────────────────────────────────────────────────────────────────────
DEFAULTS = {
    "model_type": "Metapath2Vec",
    "seed": 42,
    "embedding_dim": 64,
    "walk_length": 20,
    "context_size": 7,
    "walks_per_node": 5,
    "num_negative_samples": 5,
    "metapath2vec_epochs": 50,
    "metapath2vec_lr": 0.01,
    "metapath2vec_batch_size": 128,
    # Decoder fine-tuning
    "decoder_epochs": 300,
    "decoder_lr": 0.01,
    "decoder_weight_decay": 1e-4,
    "margin": 0.5,
    "num_neg": 1,
    "test_ratio": 0.2,
    "top_k": 5,
    "degree_penalty_alpha": 0.35,
    "exclude_existing_pairs": True,
}


# ──────────────────────────────────────────────────────────────────────
# Build HeteroData from graph_data
# ──────────────────────────────────────────────────────────────────────

# Mapping from edge type string → (src_node_type, edge_type, dst_node_type)
EDGE_TYPE_TRIPLETS = {
    "uses_algorithm": ("Technique", "uses_algorithm", "AlgorithmicComponent"),
    "uses_prompt": ("Technique", "uses_prompt", "PromptComponent"),
    "uses_data_flow": ("Technique", "uses_data_flow", "DataFlow"),
    "excels_at": ("Technique", "excels_at", "Task"),
    "develops": ("Technique", "develops", "CognitiveCapability"),
    "requires": ("Task", "requires", "CognitiveCapability"),
    "belongs_to": ("Technique", "belongs_to", "Category"),
    "defined_in": ("Technique", "defined_in", "Paper"),
    "evaluated_on": ("Technique", "evaluated_on", "Model"),
    "achieved": ("Technique", "achieved", "BenchmarkResult"),
    "extends": ("Technique", "extends", "Technique"),
}

# ──────────────────────────────────────────────────────────────────────
# Target-aligned metapaths
# ──────────────────────────────────────────────────────────────────────
# Each metapath follows: Technique → Task → Technique → Component → Technique
# This captures the full bridge that generates (Task, Component) edges.
TARGET_METAPATHS = {
    "AlgorithmicComponent": [
        ("Technique", "excels_at", "Task"),
        ("Task", "rev_excels_at", "Technique"),
        ("Technique", "uses_algorithm", "AlgorithmicComponent"),
        ("AlgorithmicComponent", "rev_uses_algorithm", "Technique"),
    ],
    "PromptComponent": [
        ("Technique", "excels_at", "Task"),
        ("Task", "rev_excels_at", "Technique"),
        ("Technique", "uses_prompt", "PromptComponent"),
        ("PromptComponent", "rev_uses_prompt", "Technique"),
    ],
    "DataFlow": [
        ("Technique", "excels_at", "Task"),
        ("Task", "rev_excels_at", "Technique"),
        ("Technique", "uses_data_flow", "DataFlow"),
        ("DataFlow", "rev_uses_data_flow", "Technique"),
    ],
}


def build_hetero_data(graph_data):
    """
    Convert our flat graph_data dict into a PyG HeteroData object.
    Creates local (per-type) node indices and remapped edge indices.
    Returns (hetero_data, global_to_local, local_to_global).
    """
    node_type_to_indices = graph_data["node_type_to_indices"]

    # Build global → local index mappings per node type
    global_to_local = {}  # global_idx → (node_type, local_idx)
    local_to_global = defaultdict(dict)  # node_type → {local_idx: global_idx}

    for node_type, global_indices in node_type_to_indices.items():
        for local_idx, global_idx in enumerate(sorted(global_indices)):
            global_to_local[global_idx] = (node_type, local_idx)
            local_to_global[node_type][local_idx] = global_idx

    hetero_data = HeteroData()

    # Set number of nodes per type
    for node_type, global_indices in node_type_to_indices.items():
        hetero_data[node_type].num_nodes = len(global_indices)

    # Set edges with local indices
    edge_index_tensors = graph_data["edge_index_tensors"]

    for edge_type_str, edge_index in edge_index_tensors.items():
        if edge_type_str not in EDGE_TYPE_TRIPLETS:
            logging.warning(f"Unknown edge type '{edge_type_str}', skipping.")
            continue

        src_type, rel, dst_type = EDGE_TYPE_TRIPLETS[edge_type_str]

        # Remap global indices to local
        local_src = []
        local_dst = []
        for i in range(edge_index.shape[1]):
            g_src = edge_index[0, i].item()
            g_dst = edge_index[1, i].item()

            if g_src not in global_to_local or g_dst not in global_to_local:
                continue

            src_nt, src_li = global_to_local[g_src]
            dst_nt, dst_li = global_to_local[g_dst]

            # Verify types match
            if src_nt == src_type and dst_nt == dst_type:
                local_src.append(src_li)
                local_dst.append(dst_li)

        if local_src:
            ei = torch.tensor([local_src, local_dst], dtype=torch.long)
            hetero_data[src_type, rel, dst_type].edge_index = ei

            # Also add the reverse edge (needed for metapath walks)
            rev_ei = torch.stack([ei[1], ei[0]], dim=0)
            rev_rel = f"rev_{rel}"
            hetero_data[dst_type, rev_rel, src_type].edge_index = rev_ei

            logging.info(
                f"  Edge ({src_type}, {rel}, {dst_type}): {ei.shape[1]} edges"
            )

    return hetero_data, global_to_local, local_to_global


def get_valid_metapaths(hetero_data):
    """
    Filter TARGET_METAPATHS to only those whose edges all exist in the data.
    Returns a dict {component_type: metapath} for valid metapaths.
    """
    valid = {}
    for comp_type, mp in TARGET_METAPATHS.items():
        all_present = all(triplet in hetero_data.edge_types for triplet in mp)
        if all_present:
            valid[comp_type] = mp
            path_str = " -> ".join(t[2] for t in mp)
            logging.info(f"  Valid metapath ({comp_type}): Technique -> {path_str}")
        else:
            missing = [t for t in mp if t not in hetero_data.edge_types]
            logging.warning(
                f"  Metapath for {comp_type} skipped — missing edges: {missing}"
            )

    return valid


# ──────────────────────────────────────────────────────────────────────
# Wrapper model
# ──────────────────────────────────────────────────────────────────────
class Metapath2VecLinkPredictor(torch.nn.Module):
    """
    Wraps Metapath2Vec embeddings with a dot-product decoder.
    Uses global indexing for compatibility with the discovery pipeline.
    """

    def __init__(self, num_nodes: int, embedding_dim: int):
        super().__init__()
        self.embeddings = torch.nn.Parameter(
            torch.zeros(num_nodes, embedding_dim), requires_grad=False
        )

    def encode(self, x, edge_index):
        """Return pre-computed embeddings (ignores inputs)."""
        return self.embeddings

    def decode(self, z, edge_index):
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=-1)


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────
def train_single_metapath(hetero_data, metapath, comp_type, hparams):
    """Train one MetaPath2Vec model for a single target-aligned metapath."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PyGMetaPath2Vec(
        edge_index_dict={
            et: hetero_data[et].edge_index for et in hetero_data.edge_types
        },
        embedding_dim=hparams["embedding_dim"],
        metapath=metapath,
        walk_length=hparams["walk_length"],
        context_size=hparams["context_size"],
        walks_per_node=hparams["walks_per_node"],
        num_negative_samples=hparams["num_negative_samples"],
    ).to(device)

    loader = model.loader(
        batch_size=hparams["metapath2vec_batch_size"],
        shuffle=True,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=hparams["metapath2vec_lr"]
    )

    epochs = hparams["metapath2vec_epochs"]
    logging.info(f"  Training Metapath2Vec [{comp_type}] ({epochs} epochs)...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            logging.info(f"    [{comp_type}] Epoch {epoch:3d}: Loss {total_loss:.4f}")

    return model


def collect_embeddings(mp2v_model, graph_data, embedding_dim):
    """Extract per-node-type embeddings and remap to global indices."""
    node_type_to_indices = graph_data["node_type_to_indices"]
    num_nodes = graph_data["num_nodes"]
    embeddings = torch.zeros(num_nodes, embedding_dim)

    for node_type, global_indices in node_type_to_indices.items():
        sorted_globals = sorted(global_indices)
        try:
            type_emb = mp2v_model(node_type).detach().cpu()
            for local_idx, global_idx in enumerate(sorted_globals):
                if local_idx < type_emb.shape[0]:
                    embeddings[global_idx] = type_emb[local_idx]
        except Exception:
            pass  # Node type not in this metapath's walks — will be zero

    return embeddings


def train_metapath2vec_embeddings(hetero_data, valid_metapaths, graph_data, hparams):
    """
    Train one MetaPath2Vec per target-aligned metapath and average
    the resulting embeddings. This ensures every node type (Task,
    AlgorithmicComponent, PromptComponent, DataFlow) gets informative
    embeddings from at least one metapath.
    """
    embedding_dim = hparams["embedding_dim"]
    num_nodes = graph_data["num_nodes"]

    all_embeddings = []

    logging.info(
        f"Training {len(valid_metapaths)} target-aligned Metapath2Vec models..."
    )

    for comp_type, metapath in valid_metapaths.items():
        mp2v = train_single_metapath(hetero_data, metapath, comp_type, hparams)
        emb = collect_embeddings(mp2v, graph_data, embedding_dim)
        all_embeddings.append(emb)
        del mp2v  # Free memory

    # Average embeddings across all metapath runs.
    # For nodes that only appear in some metapaths, averaging with zeros
    # is acceptable since the fine-tuning phase will adjust them.
    averaged = torch.stack(all_embeddings, dim=0).mean(dim=0)
    logging.info(f"Averaged embeddings from {len(all_embeddings)} metapaths: {averaged.shape}")

    return averaged


def train_model(graph_data, hparams: dict):
    """Full pipeline: build hetero data, learn averaged embeddings, fine-tune."""

    structure_edge_index, train_pos_edge_index, test_pos_edge_index = (
        create_technique_wise_split(
            graph_data, test_ratio=hparams["test_ratio"], seed=hparams["seed"]
        )
    )

    comp_indices = get_component_indices(graph_data)
    num_nodes = graph_data["num_nodes"]

    # Phase 1: Build heterogeneous graph
    logging.info("Building HeteroData for Metapath2Vec...")
    hetero_data, global_to_local, local_to_global = build_hetero_data(graph_data)

    # Phase 2: Discover valid target-aligned metapaths
    logging.info("Checking target-aligned metapaths...")
    valid_metapaths = get_valid_metapaths(hetero_data)

    if not valid_metapaths:
        raise ValueError(
            "No valid target-aligned metapaths could be constructed. "
            "Check that excels_at and uses_* edges exist in the graph."
        )

    # Phase 3: Train MetaPath2Vec (one per component type, then average)
    global_embeddings = train_metapath2vec_embeddings(
        hetero_data, valid_metapaths, graph_data, hparams
    )

    # Phase 4: Create link predictor and fine-tune
    embedding_dim = hparams["embedding_dim"]
    model = Metapath2VecLinkPredictor(num_nodes, embedding_dim)
    model.embeddings.data.copy_(global_embeddings)
    model.embeddings.requires_grad_(True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hparams["decoder_lr"],
        weight_decay=hparams["decoder_weight_decay"],
    )
    criterion = torch.nn.MarginRankingLoss(margin=hparams["margin"])

    logging.info(
        f"Fine-tuning embeddings for link prediction ({hparams['decoder_epochs']} epochs)..."
    )

    for epoch in range(hparams["decoder_epochs"]):
        model.train()
        optimizer.zero_grad()

        z = model.encode(None, None)

        pos_scores = model.decode(z, train_pos_edge_index)
        neg_edge_index = get_structured_negatives(
            train_pos_edge_index, comp_indices, num_neg=hparams["num_neg"]
        )
        neg_scores = model.decode(z, neg_edge_index)

        target = torch.ones_like(pos_scores)
        loss = criterion(pos_scores, neg_scores, target)

        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                z = model.encode(None, None)
                if test_pos_edge_index.shape[1] > 0:
                    comp_degree_map = get_component_degree_map(graph_data, structure_edge_index)
                    
                    test_pos_scores_raw = model.decode(z, test_pos_edge_index)
                    test_neg_edge_index = get_structured_negatives(
                        test_pos_edge_index, comp_indices
                    )
                    test_neg_scores_raw = model.decode(z, test_neg_edge_index)
                    
                    test_pos_scores = apply_degree_penalty(
                        test_pos_scores_raw.sigmoid(), test_pos_edge_index[1], comp_degree_map, hparams["degree_penalty_alpha"]
                    )
                    test_neg_scores = apply_degree_penalty(
                        test_neg_scores_raw.sigmoid(), test_neg_edge_index[1], comp_degree_map, hparams["degree_penalty_alpha"]
                    )
                    
                    acc = (test_pos_scores > test_neg_scores).float().mean()
                    logging.info(
                        f"Epoch {epoch:3d}: Loss {loss:.4f} | Test Pairwise Acc: {acc:.4f}"
                    )
                else:
                    logging.info(f"Epoch {epoch:3d}: Loss {loss:.4f}")

    model.embeddings.requires_grad_(False)

    return model, structure_edge_index, train_pos_edge_index, test_pos_edge_index, comp_indices


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    hparams = dict(DEFAULTS)

    # 1. Select graph
    graph_path = select_graph_run()
    hparams["graph_path"] = graph_path

    # 2. Setup run directory
    run_dir, timestamp = setup_run_directory(model_type="Metapath2Vec")
    setup_logging(run_dir)

    logging.info(f"Started Metapath2Vec run in: {run_dir}")
    logging.info(f"Using graph: {graph_path}")

    # 3. Seed
    set_seed(hparams["seed"])

    # 4. Log hyperparameters
    log_hyperparameters(run_dir, hparams)

    # 5. Load & train
    logging.info("Loading graph data...")
    graph_data = load_graph_data(graph_path)

    hparams["num_nodes"] = graph_data["num_nodes"]
    hparams["num_edges"] = graph_data["num_edges"]
    hparams["feature_dim"] = int(graph_data["node_features"].shape[1])
    log_hyperparameters(run_dir, hparams)

    model, structure_edge_index, train_pos_edge_index, test_pos_edge_index, comp_indices = train_model(
        graph_data, hparams
    )

    # 6. Final evaluation with full metrics
    model.eval()
    with torch.no_grad():
        z = model.encode(None, None)
        if test_pos_edge_index.shape[1] > 0:
            metrics = evaluate_deterministic(
                model, z,
                test_pos_edge_index=test_pos_edge_index,
                train_pos_edge_index=train_pos_edge_index,
                graph_data=graph_data,
                structure_edge_index=structure_edge_index,
                degree_penalty_alpha=hparams["degree_penalty_alpha"],
            )
            log_metrics(metrics, prefix="Metapath2Vec Test")
            save_metrics(metrics, run_dir, model_name="metapath2vec")
        else:
            logging.warning("No test edges available for final evaluation.")

    # 7. Discover
    discover_novel_techniques(
        model, graph_data, structure_edge_index, run_dir,
        model_name="metapath2vec",
        top_k=hparams["top_k"],
        degree_penalty_alpha=hparams["degree_penalty_alpha"],
        exclude_existing_pairs=hparams["exclude_existing_pairs"],
    )

    # 8. Save model
    model_path = os.path.join(run_dir, "model_metapath2vec.pt")
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved to {model_path}")
    logging.info("Metapath2Vec run completed successfully.")


if __name__ == "__main__":
    main()
