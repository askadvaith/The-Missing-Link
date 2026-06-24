"""
Node2Vec Link Prediction on Prompt Engineering Knowledge Graph
================================================================
Uses Node2Vec random-walk-based embeddings for link prediction.
Node2Vec learns low-dimensional representations by biased random walks
(parameters p and q control BFS/DFS exploration). A simple MLP decoder
is trained on the learned embeddings for link prediction.

Output: link_pred_output/Node2Vec/run_<timestamp>/
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
import logging

import numpy as np
from torch_geometric.nn import Node2Vec as PyGNode2Vec
from torch_geometric.utils.num_nodes import maybe_num_nodes


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
from utils.metrics import log_metrics, save_metrics

# ──────────────────────────────────────────────────────────────────────
# Default hyperparameters
# ──────────────────────────────────────────────────────────────────────
DEFAULTS = {
    "model_type": "Node2Vec",
    "seed": 42,
    "embedding_dim": 64,
    "walk_length": 20,
    "context_size": 10,
    "walks_per_node": 10,
    "p": 1.0,           # Return parameter (BFS-like when low)
    "q": 1.0,           # In-out parameter (DFS-like when low)
    "num_negative_samples": 1,
    "node2vec_epochs": 100,
    "node2vec_lr": 0.01,
    "node2vec_batch_size": 128,
    # Decoder training
    "decoder_epochs": 300,
    "decoder_lr": 0.01,
    "decoder_weight_decay": 1e-4,
    "decoder_hidden": 64,
    "margin": 0.5,
    "num_neg": 1,
    "test_ratio": 0.2,
    "top_k": 5,
    "degree_penalty_alpha": 0.35,
    "exclude_existing_pairs": True,
}


# ──────────────────────────────────────────────────────────────────────
# Wrapper model so it has the same encode/decode interface
# ──────────────────────────────────────────────────────────────────────
class Node2VecLinkPredictor(torch.nn.Module):
    """
    Wraps Node2Vec embeddings with a simple dot-product decoder
    so the model is compatible with the existing discovery pipeline.
    """

    def __init__(self, embeddings: torch.Tensor):
        super().__init__()
        self.embeddings = torch.nn.Parameter(embeddings, requires_grad=False)

    def encode(self, x, edge_index):
        """Return the pre-computed Node2Vec embeddings (ignores x and edge_index)."""
        return self.embeddings

    def decode(self, z, edge_index):
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=-1)


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────
def train_node2vec_embeddings(structure_edge_index, num_nodes, hparams):
    """Train Node2Vec embeddings using PyG's built-in implementation."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    node2vec = PyGNode2Vec(
        edge_index=structure_edge_index,
        embedding_dim=hparams["embedding_dim"],
        walk_length=hparams["walk_length"],
        context_size=hparams["context_size"],
        walks_per_node=hparams["walks_per_node"],
        p=hparams["p"],
        q=hparams["q"],
        num_negative_samples=hparams["num_negative_samples"],
        num_nodes=num_nodes,
    ).to(device)

    loader = node2vec.loader(
        batch_size=hparams["node2vec_batch_size"],
        shuffle=True,
    )

    optimizer = torch.optim.Adam(
        node2vec.parameters(), lr=hparams["node2vec_lr"]
    )

    logging.info(f"Training Node2Vec ({hparams['node2vec_epochs']} epochs)...")

    for epoch in range(hparams["node2vec_epochs"]):
        node2vec.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = node2vec.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 20 == 0:
            logging.info(f"  Node2Vec Epoch {epoch:3d}: Loss {total_loss:.4f}")

    # Extract learned embeddings
    embeddings = node2vec.embedding.weight.data.cpu()
    logging.info(f"Node2Vec embeddings shape: {embeddings.shape}")

    return embeddings


def train_model(graph_data, hparams: dict):
    """Full pipeline: learn embeddings, then evaluate."""

    structure_edge_index, train_pos_edge_index, test_pos_edge_index = (
        create_technique_wise_split(
            graph_data, test_ratio=hparams["test_ratio"], seed=hparams["seed"]
        )
    )

    comp_indices = get_component_indices(graph_data)
    num_nodes = graph_data["num_nodes"]

    # Phase 1: Train Node2Vec embeddings
    embeddings = train_node2vec_embeddings(
        structure_edge_index, num_nodes, hparams
    )

    # Phase 2: Create the link predictor wrapper
    model = Node2VecLinkPredictor(embeddings)

    # Phase 3: Fine-tune with margin ranking on the link prediction task
    # We make embeddings trainable for fine-tuning
    model.embeddings.requires_grad_(True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hparams["decoder_lr"],
        weight_decay=hparams["decoder_weight_decay"],
    )
    criterion = torch.nn.MarginRankingLoss(margin=hparams["margin"])

    logging.info(f"Fine-tuning embeddings for link prediction ({hparams['decoder_epochs']} epochs)...")

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

    # Freeze embeddings after fine-tuning
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
    run_dir, timestamp = setup_run_directory(model_type="Node2Vec")
    setup_logging(run_dir)

    logging.info(f"Started Node2Vec run in: {run_dir}")
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
            log_metrics(metrics, prefix="Node2Vec Test")
            save_metrics(metrics, run_dir, model_name="node2vec")
        else:
            logging.warning("No test edges available for final evaluation.")

    # 7. Discover
    discover_novel_techniques(
        model, graph_data, structure_edge_index, run_dir,
        model_name="node2vec",
        top_k=hparams["top_k"],
        degree_penalty_alpha=hparams["degree_penalty_alpha"],
        exclude_existing_pairs=hparams["exclude_existing_pairs"],
    )

    # 8. Save model
    model_path = os.path.join(run_dir, "model_node2vec.pt")
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved to {model_path}")
    logging.info("Node2Vec run completed successfully.")


if __name__ == "__main__":
    main()
