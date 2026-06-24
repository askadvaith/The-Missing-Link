"""
GCN Link Prediction on Prompt Engineering Knowledge Graph
==========================================================
Uses SAGEConv (a GCN variant) with MarginRankingLoss and
Structured Negative Sampling. Predicts novel technique compositions
by learning relative rankings.

Output: link_pred_output/GCN/run_<timestamp>/
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
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
    "model_type": "GCN",
    "seed": 42,
    "epochs": 300,
    "hidden_channels": 64,
    "out_channels": 64,
    "dropout": 0.2,
    "lr": 0.01,
    "weight_decay": 1e-4,
    "margin": 0.5,
    "num_neg": 1,
    "test_ratio": 0.2,
    "top_k": 5,
    "degree_penalty_alpha": 0.80,
    "exclude_existing_pairs": True,
    "activation": "elu",
    "conv_type": "SAGEConv",
    "temperature_init": 2.0,
}


# ──────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────
class GCNLinkPrediction(torch.nn.Module):
    """SAGEConv-based GNN for link prediction."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        dropout=0.2,
        temperature_init=1.0,
    ):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout
        self.temperature = torch.nn.Parameter(torch.tensor(temperature_init))

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        return x

    def decode(self, z, edge_index):
        src, dst = edge_index
        return self.temperature * (z[src] * z[dst]).sum(dim=-1)


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────
def train_model(graph_data, hparams: dict):
    structure_edge_index, train_pos_edge_index, test_pos_edge_index = (
        create_technique_wise_split(graph_data, test_ratio=hparams["test_ratio"], seed=hparams["seed"])
    )

    x = graph_data["node_features"]
    comp_indices = get_component_indices(graph_data)

    model = GCNLinkPrediction(
        in_channels=x.shape[1],
        hidden_channels=hparams["hidden_channels"],
        out_channels=hparams["out_channels"],
        dropout=hparams["dropout"],
        temperature_init=hparams["temperature_init"],
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=hparams["lr"], weight_decay=hparams["weight_decay"]
    )
    criterion = torch.nn.MarginRankingLoss(margin=hparams["margin"])

    logging.info(f"Training GCN (SAGEConv) on {x.shape[1]}-dim semantic features...")

    best_acc = 0.0
    best_state = None

    for epoch in range(hparams["epochs"]):
        model.train()
        optimizer.zero_grad()

        z = model.encode(x, structure_edge_index)

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
                z = model.encode(x, structure_edge_index)
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
    run_dir, timestamp = setup_run_directory(model_type="GCN")
    setup_logging(run_dir)

    logging.info(f"Started GCN run in: {run_dir}")
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
    # Re-save with graph stats
    log_hyperparameters(run_dir, hparams)

    model, structure_edge_index, train_pos_edge_index, test_pos_edge_index, comp_indices = train_model(graph_data, hparams)

    # 6. Final evaluation — deterministic exhaustive negative ranking
    model.eval()
    with torch.no_grad():
        z = model.encode(graph_data["node_features"], structure_edge_index)
        if test_pos_edge_index.shape[1] > 0:
            metrics = evaluate_deterministic(
                model, z,
                test_pos_edge_index=test_pos_edge_index,
                train_pos_edge_index=train_pos_edge_index,
                graph_data=graph_data,
                structure_edge_index=structure_edge_index,
                degree_penalty_alpha=hparams["degree_penalty_alpha"],
            )
            log_metrics(metrics, prefix="GCN Test")
            save_metrics(metrics, run_dir, model_name="gcn")
        else:
            logging.warning("No test edges available for final evaluation.")

    # 7. Discover
    discover_novel_techniques(
        model, graph_data, structure_edge_index, run_dir,
        model_name="gcn",
        top_k=hparams["top_k"],
        degree_penalty_alpha=hparams["degree_penalty_alpha"],
        exclude_existing_pairs=hparams["exclude_existing_pairs"],
    )

    # 8. Save model
    model_path = os.path.join(run_dir, "model_gcn.pt")
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved to {model_path}")
    logging.info("GCN run completed successfully.")


if __name__ == "__main__":
    main()
