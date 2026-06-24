"""
R-GCN Link Prediction on Prompt Engineering Knowledge Graph
=============================================================
Uses Relational Graph Convolutional Networks (RGCNConv) designed specifically
for multi-relational graphs. Unlike GCN/GAT/GTN which treat all edge types
identically during message passing, R-GCN learns separate transformation
matrices per relation type, naturally handling the heterogeneous graph
structure.

Uses basis decomposition to keep the number of parameters tractable
given the 11 distinct edge types.

Output: link_pred_output/RGCN/run_<timestamp>/
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
import logging

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
    "model_type": "RGCN",
    "seed": 42,
    "epochs": 300,
    "hidden_channels": 64,
    "out_channels": 64,
    "num_bases": 4,       # Basis decomposition to reduce parameters
    "dropout": 0.2,
    "lr": 0.01,
    "weight_decay": 1e-4,
    "margin": 0.5,
    "num_neg": 1,
    "test_ratio": 0.2,
    "top_k": 5,
    "degree_penalty_alpha": 0.00,
    "exclude_existing_pairs": True,
    "activation": "elu",
    "conv_type": "RGCNConv",
    "temperature_init": 2.0,
}


# ──────────────────────────────────────────────────────────────────────
# Build relation-typed edge index
# ──────────────────────────────────────────────────────────────────────
def build_relational_edge_index(graph_data):
    """
    Combine all edge types into a single edge_index tensor with an
    accompanying edge_type tensor. Each edge type gets a unique integer ID.

    Returns:
        edge_index: [2, num_total_edges] tensor
        edge_type: [num_total_edges] tensor of relation IDs
        num_relations: total number of distinct relation types
        relation_names: list mapping relation ID -> name
    """
    edge_index_tensors = graph_data["edge_index_tensors"]

    all_edges = []
    all_types = []
    relation_names = sorted(edge_index_tensors.keys())
    rel_to_id = {name: i for i, name in enumerate(relation_names)}

    for rel_name in relation_names:
        ei = edge_index_tensors[rel_name]
        num_edges = ei.shape[1]
        all_edges.append(ei)
        all_types.append(torch.full((num_edges,), rel_to_id[rel_name], dtype=torch.long))

    combined_edge_index = torch.cat(all_edges, dim=1)
    combined_edge_type = torch.cat(all_types, dim=0)

    return combined_edge_index, combined_edge_type, len(relation_names), relation_names


def filter_relational_edges(edge_index, edge_type, test_tech_indices):
    """
    Filter out edges touching test techniques from the relational edge index.
    Used to create a clean message-passing graph for training.
    """
    test_set = set(test_tech_indices) if not isinstance(test_tech_indices, set) else test_tech_indices
    mask = torch.tensor([
        edge_index[0, i].item() not in test_set and edge_index[1, i].item() not in test_set
        for i in range(edge_index.shape[1])
    ], dtype=torch.bool)

    return edge_index[:, mask], edge_type[mask]


# ──────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────
class RGCNLinkPrediction(torch.nn.Module):
    """
    R-GCN for link prediction with basis decomposition.

    Each relation type gets its own transformation matrix, decomposed as
    a linear combination of a small number of shared basis matrices.
    This reduces parameters from O(R * d * d) to O(B * d * d + R * B),
    where R = num_relations, B = num_bases, d = hidden_dim.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_relations: int,
        num_bases: int = 4,
        dropout: float = 0.2,
        temperature_init: float = 1.0,
    ):
        super().__init__()
        self.conv1 = RGCNConv(
            in_channels, hidden_channels,
            num_relations=num_relations,
            num_bases=num_bases,
        )
        self.conv2 = RGCNConv(
            hidden_channels, out_channels,
            num_relations=num_relations,
            num_bases=num_bases,
        )
        self.dropout = dropout
        self.temperature = torch.nn.Parameter(torch.tensor(temperature_init))

    def encode(self, x, edge_index, edge_type=None):
        """
        Encode node features using R-GCN layers.
        Accepts edge_type as a keyword/positional arg for relation-aware
        message passing.
        """
        x = self.conv1(x, edge_index, edge_type)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        x = F.normalize(x, p=2, dim=-1)
        return x

    def decode(self, z, edge_index):
        src, dst = edge_index
        return self.temperature * (z[src] * z[dst]).sum(dim=-1)


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────
def train_model(graph_data, hparams: dict):
    # Build relational edge index from all edge types
    full_edge_index, full_edge_type, num_relations, relation_names = (
        build_relational_edge_index(graph_data)
    )

    logging.info(f"Relational graph: {num_relations} relation types")
    for i, name in enumerate(relation_names):
        count = (full_edge_type == i).sum().item()
        logging.info(f"  [{i}] {name}: {count} edges")

    hparams["num_relations"] = num_relations
    hparams["relation_names"] = relation_names

    # Technique-wise split for supervision edges
    _, train_pos_edge_index, test_pos_edge_index = (
        create_technique_wise_split(
            graph_data, test_ratio=hparams["test_ratio"], seed=hparams["seed"]
        )
    )

    # Filter the relational edges to exclude test techniques
    import random
    tech_indices = sorted(graph_data["node_type_to_indices"]["Technique"])
    rng = random.Random(hparams["seed"])
    rng.shuffle(tech_indices)
    split_idx = int(len(tech_indices) * hparams["test_ratio"])
    test_tech_indices = set(tech_indices[:split_idx])

    train_edge_index, train_edge_type = filter_relational_edges(
        full_edge_index, full_edge_type, test_tech_indices
    )

    logging.info(
        f"Message-passing edges: {train_edge_index.shape[1]} "
        f"(filtered from {full_edge_index.shape[1]})"
    )

    x = graph_data["node_features"]
    comp_indices = get_component_indices(graph_data)

    model = RGCNLinkPrediction(
        in_channels=x.shape[1],
        hidden_channels=hparams["hidden_channels"],
        out_channels=hparams["out_channels"],
        num_relations=num_relations,
        num_bases=hparams["num_bases"],
        dropout=hparams["dropout"],
        temperature_init=hparams["temperature_init"],
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(
        f"R-GCN Model -- Total params: {total_params:,}, Trainable: {trainable_params:,}"
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=hparams["lr"], weight_decay=hparams["weight_decay"]
    )
    criterion = torch.nn.MarginRankingLoss(margin=hparams["margin"])

    logging.info(f"Training R-GCN on {x.shape[1]}-dim semantic features...")

    best_acc = 0.0
    best_state = None

    for epoch in range(hparams["epochs"]):
        model.train()
        optimizer.zero_grad()

        # R-GCN encode uses relation-typed edges
        z = model.encode(x, train_edge_index, edge_type=train_edge_type)

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
                z = model.encode(x, train_edge_index, edge_type=train_edge_type)
                if test_pos_edge_index.shape[1] > 0:
                    comp_degree_map = get_component_degree_map(graph_data, train_edge_index)
                    
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
                    
                    acc = (test_pos_scores > test_neg_scores).float().mean().item()

                    if acc > best_acc:
                        best_acc = acc
                        best_state = {
                            k: v.detach().clone() for k, v in model.state_dict().items()
                        }

                    logging.info(
                        f"Epoch {epoch:3d}: Loss {loss:.4f} | Test Pairwise Acc: {acc:.4f}"
                    )
                else:
                    logging.info(f"Epoch {epoch:3d}: Loss {loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        logging.info(f"Restored best model with Test Pairwise Acc: {best_acc:.4f}")

    return model, train_edge_index, train_edge_type, train_pos_edge_index, test_pos_edge_index, comp_indices


# ──────────────────────────────────────────────────────────────────────
# Encode wrapper for discovery (needs edge_type)
# ──────────────────────────────────────────────────────────────────────
class RGCNDiscoveryWrapper(torch.nn.Module):
    """
    Wraps the R-GCN model to provide the same encode(x, edge_index)
    interface expected by discover_novel_techniques, by binding the
    edge_type to the model.
    """

    def __init__(self, rgcn_model, edge_index, edge_type):
        super().__init__()
        self.rgcn_model = rgcn_model
        self.register_buffer("_edge_index", edge_index)
        self.register_buffer("_edge_type", edge_type)

    def encode(self, x, edge_index):
        """Ignores the passed edge_index and uses the stored relational one."""
        return self.rgcn_model.encode(x, self._edge_index, edge_type=self._edge_type)

    def decode(self, z, edge_index):
        return self.rgcn_model.decode(z, edge_index)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    hparams = dict(DEFAULTS)

    # 1. Select graph
    graph_path = select_graph_run()
    hparams["graph_path"] = graph_path

    # 2. Setup run directory
    run_dir, timestamp = setup_run_directory(model_type="RGCN")
    setup_logging(run_dir)

    logging.info(f"Started R-GCN run in: {run_dir}")
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

    model, train_edge_index, train_edge_type, train_pos_edge_index, test_pos_edge_index, comp_indices = (
        train_model(graph_data, hparams)
    )

    # 6. Final evaluation — deterministic exhaustive negative ranking
    model.eval()
    with torch.no_grad():
        z = model.encode(
            graph_data["node_features"], train_edge_index, edge_type=train_edge_type
        )
        if test_pos_edge_index.shape[1] > 0:
            metrics = evaluate_deterministic(
                model, z,
                test_pos_edge_index=test_pos_edge_index,
                train_pos_edge_index=train_pos_edge_index,
                graph_data=graph_data,
                structure_edge_index=train_edge_index,
                degree_penalty_alpha=hparams["degree_penalty_alpha"],
            )
            log_metrics(metrics, prefix="R-GCN Test")
            save_metrics(metrics, run_dir, model_name="rgcn")
        else:
            logging.warning("No test edges available for final evaluation.")

    # 7. Discover — wrap model so discovery gets the right encode interface
    wrapper = RGCNDiscoveryWrapper(model, train_edge_index, train_edge_type)
    discover_novel_techniques(
        wrapper, graph_data, train_edge_index, run_dir,
        model_name="rgcn",
        top_k=hparams["top_k"],
        degree_penalty_alpha=hparams["degree_penalty_alpha"],
        exclude_existing_pairs=hparams["exclude_existing_pairs"],
    )

    # 8. Save model
    model_path = os.path.join(run_dir, "model_rgcn.pt")
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved to {model_path}")
    logging.info("R-GCN run completed successfully.")


if __name__ == "__main__":
    main()
