"""
GTN Link Prediction on Prompt Engineering Knowledge Graph
==========================================================
Uses a Graph Transformer Network built with TransformerConv layers
from PyTorch Geometric. TransformerConv implements a transformer-style
multi-head attention mechanism over graph neighborhoods, enabling the
model to learn weighted aggregations of neighbor features.

Architecture:
  - 2-layer TransformerConv with multi-head attention
  - Residual / skip connections for stability on small graphs
  - Layer normalization after each conv block
  - MarginRankingLoss with structured negative sampling

Output: link_pred_output/GTN/run_<timestamp>/
"""

import sys
import os

# Ensure parent directory is on sys.path so `link_prediction` package is importable
# when the script is invoked directly (e.g. python link_prediction/gtn_link_prediction.py)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch.nn import LayerNorm
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
    discover_novel_techniques_relative,
    get_component_degree_map,
    apply_degree_penalty,
    evaluate_deterministic,
)
from utils.metrics import log_metrics, save_metrics

# ──────────────────────────────────────────────────────────────────────
# Default hyperparameters
# ──────────────────────────────────────────────────────────────────────
DEFAULTS = {
    "model_type": "GTN",
    "seed": 42,
    "epochs": 300,
    "hidden_channels": 16,
    "out_channels": 64,
    "heads": 2,
    "dropout": 0.3,
    "lr": 0.001,
    "weight_decay": 5e-4,
    "margin": 0.5,
    "num_neg": 1,
    "test_ratio": 0.2,
    "top_k": 5,
    "activation": "elu",
    "conv_type": "TransformerConv",
    "use_layer_norm": True,
    "use_skip_connection": True,
    "beta": True,  # learnable skip-connection weighting in TransformerConv
    "temperature_init": 2.0,  # learnable temperature for score scaling
    "relative_threshold_factor": 0.95,
    "max_candidates_per_task": 3,
    "degree_penalty_alpha": 1.50,
    "exclude_existing_pairs": True,
}


# ──────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────
class GTNLinkPrediction(torch.nn.Module):
    """
    Graph Transformer Network for link prediction.

    Uses TransformerConv which applies transformer-style multi-head
    attention over each node's neighborhood. Optional skip connections
    and layer normalization improve training stability on small graphs.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.2,
        use_layer_norm: bool = True,
        use_skip_connection: bool = True,
        beta: bool = True,
        temperature_init: float = 1.0,
    ):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.use_skip_connection = use_skip_connection
        self.dropout = dropout

        # Learnable temperature: scales dot product before sigmoid so the
        # model can learn to spread scores across the full [0, 1] range
        # instead of being squished by L2-normalized cosine similarity.
        self.temperature = torch.nn.Parameter(torch.tensor(temperature_init))

        # Layer 1: in_channels → hidden_channels * heads
        self.conv1 = TransformerConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
            beta=False,  # no skip on first layer (dim mismatch)
        )
        if use_layer_norm:
            self.norm1 = LayerNorm(hidden_channels * heads)

        # Optional linear projection for skip connection when dims differ
        if use_skip_connection:
            self.skip_proj1 = torch.nn.Linear(in_channels, hidden_channels * heads)

        # Layer 2: hidden_channels * heads → out_channels
        self.conv2 = TransformerConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=dropout,
            beta=beta,
        )
        if use_layer_norm:
            self.norm2 = LayerNorm(out_channels)

        # Skip connection for layer 2 (project if dims differ)
        if use_skip_connection:
            self.skip_proj2 = torch.nn.Linear(hidden_channels * heads, out_channels)

    def encode(self, x, edge_index):
        # ── Layer 1 ──
        identity = x
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        if self.use_skip_connection:
            x = x + self.skip_proj1(identity)
        if self.use_layer_norm:
            x = self.norm1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ── Layer 2 ──
        identity = x
        x = self.conv2(x, edge_index)
        if self.use_skip_connection:
            x = x + self.skip_proj2(identity)
        if self.use_layer_norm:
            x = self.norm2(x)

        # L2-normalize so dot-product in decode measures cosine similarity
        # and sigmoid scores stay in a meaningful range instead of saturating.
        x = F.normalize(x, p=2, dim=-1)

        return x

    def decode(self, z, edge_index):
        src, dst = edge_index
        # Scale dot product by learned temperature so sigmoid gives
        # well-separated scores instead of clustering around 0.5.
        return self.temperature * (z[src] * z[dst]).sum(dim=-1)


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────
def train_model(graph_data, hparams: dict):
    structure_edge_index, train_pos_edge_index, test_pos_edge_index = (
        create_technique_wise_split(
            graph_data, test_ratio=hparams["test_ratio"], seed=hparams["seed"]
        )
    )

    x = graph_data["node_features"]
    comp_indices = get_component_indices(graph_data)

    model = GTNLinkPrediction(
        in_channels=x.shape[1],
        hidden_channels=hparams["hidden_channels"],
        out_channels=hparams["out_channels"],
        heads=hparams["heads"],
        dropout=hparams["dropout"],
        use_layer_norm=hparams["use_layer_norm"],
        use_skip_connection=hparams["use_skip_connection"],
        beta=hparams["beta"],
        temperature_init=hparams["temperature_init"],
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=hparams["lr"], weight_decay=hparams["weight_decay"]
    )
    criterion = torch.nn.MarginRankingLoss(margin=hparams["margin"])

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(
        f"GTN Model — Total params: {total_params:,}, Trainable: {trainable_params:,}"
    )
    logging.info(f"Training GTN (TransformerConv) on {x.shape[1]}-dim semantic features...")

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
                        best_state = {k: v.clone() for k, v in model.state_dict().items()}

                    logging.info(
                        f"Epoch {epoch:3d}: Loss {loss:.4f} | Test Pairwise Acc: {acc:.4f}"
                    )
                else:
                    logging.info(f"Epoch {epoch:3d}: Loss {loss:.4f}")

    # Restore best model if we tracked one
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
    run_dir, timestamp = setup_run_directory(model_type="GTN")
    setup_logging(run_dir)

    logging.info(f"Started GTN run in: {run_dir}")
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
            log_metrics(metrics, prefix="GTN Test")
            save_metrics(metrics, run_dir, model_name="gtn")
        else:
            logging.warning("No test edges available for final evaluation.")

    # 7. Discover (Dynamic Relative Thresholding)
    discover_novel_techniques_relative(
        model, graph_data, structure_edge_index, run_dir,
        model_name="gtn",
        relative_threshold_factor=hparams["relative_threshold_factor"],
        max_candidates_per_task=hparams["max_candidates_per_task"],
        degree_penalty_alpha=hparams["degree_penalty_alpha"],
        exclude_existing_pairs=hparams["exclude_existing_pairs"],
    )

    # 8. Save model
    model_path = os.path.join(run_dir, "model_gtn.pt")
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved to {model_path}")
    logging.info("GTN run completed successfully.")


if __name__ == "__main__":
    main()
