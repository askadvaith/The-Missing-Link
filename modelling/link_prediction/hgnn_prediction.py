"""Hypergraph-only technique discovery pipeline.

Trains a lightweight HGNN on hyperedges and ranks candidate novel techniques.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv

# Allow running the script directly via path while importing workspace modules.
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from utils.metrics import evaluate_link_prediction, log_metrics, save_metrics


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_run_dir(base_dir: str = "link_pred_output/HGNN") -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logging(run_dir: Path) -> None:
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(run_dir / "run.log"),
            logging.StreamHandler(),
        ],
    )


def load_hypergraph(path: str = "data/hypergraph_output/hypergraph_latest.pkl") -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def split_hyperedges_strict_member_disjoint(
    memberships: dict[int, list[int]],
    node_types_by_idx: dict[int, str],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], list[int], dict]:
    """Create a leakage-safe split with strict node-type disjointness.

    We avoid any overlap of Task/Component/Capability nodes between train and test.
    This is stricter than ordinary edge disjointness and directly prevents the
    earlier leakage pattern where shared components leak supervision.
    """
    protected_types = {
        "AlgorithmicComponent",
        "PromptComponent",
        "DataFlow",
    }

    def core_members(edge_id: int) -> set[int]:
        return {
            n
            for n in memberships.get(edge_id, [])
            if node_types_by_idx.get(n, "") in protected_types
        }

    edge_ids = sorted(memberships.keys())
    rng = random.Random(seed)
    rng.shuffle(edge_ids)

    target_test = max(1, int(test_ratio * len(edge_ids)))
    test_edges = edge_ids[:target_test]
    test_core_nodes: set[int] = set()
    for e in test_edges:
        test_core_nodes.update(core_members(e))

    train_edges = [
        e for e in edge_ids[target_test:]
        if core_members(e).isdisjoint(test_core_nodes)
    ]

    if not train_edges:
        raise RuntimeError(
            "Strict no-leakage split produced zero train hyperedges. "
            "Reduce test_ratio or increase graph density."
        )

    train_core_nodes: set[int] = set()
    for e in train_edges:
        train_core_nodes.update(core_members(e))

    overlap_nodes = train_core_nodes.intersection(test_core_nodes)
    leakage_report = {
        "protected_node_types": sorted(protected_types),
        "num_train_hyperedges": len(train_edges),
        "num_test_hyperedges": len(test_edges),
        "train_core_nodes": len(train_core_nodes),
        "test_core_nodes": len(test_core_nodes),
        "overlap_core_nodes": len(overlap_nodes),
    }

    return sorted(train_edges), sorted(test_edges), leakage_report


def filter_hyperedge_index(hyperedge_index: torch.Tensor, allowed_edges: set[int]) -> torch.Tensor:
    e_idx = hyperedge_index[1]
    mask = torch.tensor([int(x.item()) in allowed_edges for x in e_idx], dtype=torch.bool)
    out = hyperedge_index[:, mask]
    if out.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return out


def build_membership_map(hyperedge_index: torch.Tensor) -> dict[int, list[int]]:
    mapping: dict[int, list[int]] = {}
    for i in range(hyperedge_index.shape[1]):
        n = int(hyperedge_index[0, i].item())
        e = int(hyperedge_index[1, i].item())
        mapping.setdefault(e, []).append(n)
    for e in mapping:
        mapping[e] = sorted(set(mapping[e]))
    return mapping


def get_node_index_maps(node_metadata: list[dict], node_to_idx: dict[str, int]) -> tuple[dict[str, list[int]], dict[int, dict], dict[int, str]]:
    by_type: dict[str, list[int]] = {}
    metadata_by_idx: dict[int, dict] = {}
    node_types_by_idx: dict[int, str] = {}

    for node in node_metadata:
        idx = node_to_idx[node["id"]]
        node_type = node["type"]
        by_type.setdefault(node_type, []).append(idx)
        metadata_by_idx[idx] = node
        node_types_by_idx[idx] = node_type

    return by_type, metadata_by_idx, node_types_by_idx


class HGNNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = HypergraphConv(in_dim, hidden_dim)
        self.conv2 = HypergraphConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, hyperedge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, hyperedge_index)
        return x


class DeepResidualHGNNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.conv1 = HypergraphConv(hidden_dim, hidden_dim)
        self.conv2 = HypergraphConv(hidden_dim, hidden_dim)
        self.conv3 = HypergraphConv(hidden_dim, out_dim)
        self.residual_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)

        h = self.conv1(x, hyperedge_index)
        h = self.norm1(F.relu(h))
        h = F.dropout(h, p=self.dropout, training=self.training)

        h2 = self.conv2(h, hyperedge_index)
        h = self.norm2(F.relu(h2 + self.residual_proj(h)))
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv3(h, hyperedge_index)
        h = self.norm3(h)
        return h


class HyperedgeScorer(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, edge_embeddings: torch.Tensor) -> torch.Tensor:
        return self.ffn(edge_embeddings).squeeze(-1)


def make_encoder(
    architecture: str,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    dropout: float,
) -> nn.Module:
    arch = architecture.lower()
    if arch == "hgnn":
        return HGNNEncoder(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, dropout=dropout)
    if arch in {"deep_hgnn", "res_hgnn"}:
        return DeepResidualHGNNEncoder(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, dropout=dropout)
    raise ValueError(f"Unsupported architecture: {architecture}")


def aggregate_hyperedge_embeddings(
    node_embeddings: torch.Tensor,
    memberships: dict[int, list[int]],
    edge_indices: list[int],
    mode: str = "mean_max_std",
) -> torch.Tensor:
    vectors = []
    emb_dim = node_embeddings.shape[1]
    for e in edge_indices:
        members = memberships.get(e, [])
        if not members:
            if mode == "mean":
                vectors.append(torch.zeros(emb_dim, device=node_embeddings.device))
            else:
                vectors.append(torch.zeros(emb_dim * 3, device=node_embeddings.device))
            continue
        member_tensor = node_embeddings[torch.tensor(members, dtype=torch.long, device=node_embeddings.device)]
        mean_vec = member_tensor.mean(dim=0)
        if mode == "mean":
            vec = mean_vec
        elif mode == "mean_max_std":
            max_vec = member_tensor.max(dim=0).values
            std_vec = member_tensor.std(dim=0, unbiased=False)
            vec = torch.cat([mean_vec, max_vec, std_vec], dim=0)
        else:
            raise ValueError(f"Unsupported aggregation mode: {mode}")
        vectors.append(vec)
    if not vectors:
        if mode == "mean":
            return torch.empty((0, emb_dim), device=node_embeddings.device)
        return torch.empty((0, emb_dim * 3), device=node_embeddings.device)
    return torch.stack(vectors, dim=0)


def make_negative_membership(
    memberships: dict[int, list[int]],
    edge_indices: list[int],
    candidate_nodes: list[int],
    seed: int = 42,
) -> dict[int, list[int]]:
    rng = random.Random(seed)
    neg = {}
    for e in edge_indices:
        members = list(memberships.get(e, []))
        if not members:
            continue
        pos_set = set(members)
        slot = rng.randrange(len(members))
        replacement = rng.choice(candidate_nodes)
        retries = 0
        while replacement in pos_set and retries < 30:
            replacement = rng.choice(candidate_nodes)
            retries += 1
        members[slot] = replacement
        neg[e] = sorted(set(members))
    return neg


def make_negative_membership_typed(
    memberships: dict[int, list[int]],
    edge_indices: list[int],
    candidate_nodes: list[int],
    node_types_by_idx: dict[int, str],
    num_negatives: int = 1,
    seed: int = 42,
) -> dict[int, list[int]]:
    rng = random.Random(seed)
    by_type: dict[str, list[int]] = {}
    for n in candidate_nodes:
        by_type.setdefault(node_types_by_idx.get(n, ""), []).append(n)

    neg: dict[int, list[int]] = {}
    for e in edge_indices:
        members = list(memberships.get(e, []))
        if not members:
            continue

        for n_idx in range(num_negatives):
            corrupt_members = members.copy()
            slot = rng.randrange(len(corrupt_members))
            original = corrupt_members[slot]
            node_type = node_types_by_idx.get(original, "")

            pool = by_type.get(node_type, candidate_nodes)
            replacement = rng.choice(pool)
            retries = 0
            pos_set = set(corrupt_members)
            while replacement in pos_set and retries < 50:
                replacement = rng.choice(pool)
                retries += 1

            corrupt_members[slot] = replacement
            neg[e * 10_000 + n_idx] = sorted(set(corrupt_members))

    return neg


def signature(members: list[int]) -> tuple[int, ...]:
    return tuple(sorted(set(members)))


def _component_payload(member_indices: list[int], metadata_by_idx: dict[int, dict]) -> tuple[list[dict], list[dict]]:
    components = []
    capabilities = []
    for idx in member_indices:
        meta = metadata_by_idx[idx]
        if meta["type"] in {"AlgorithmicComponent", "PromptComponent", "DataFlow"}:
            components.append({
                "id": meta["id"],
                "name": meta["name"],
                "type": meta["type"],
                "description": meta.get("description", ""),
            })
        if meta["type"] == "CognitiveCapability":
            capabilities.append({
                "id": meta["id"],
                "name": meta["name"],
                "type": meta["type"],
                "description": meta.get("description", ""),
            })
    return components, capabilities


def discover_candidates(
    node_embeddings: torch.Tensor,
    scorer: HyperedgeScorer,
    by_type: dict[str, list[int]],
    metadata_by_idx: dict[int, dict],
    known_signatures: set[tuple[int, ...]],
    top_k: int = 50,
    max_per_task: int = 8,
    edge_pooling: str = "mean_max_std",
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(42)

    task_nodes = by_type.get("Task", [])
    alg_nodes = by_type.get("AlgorithmicComponent", [])
    prompt_nodes = by_type.get("PromptComponent", [])
    flow_nodes = by_type.get("DataFlow", [])
    capability_nodes = by_type.get("CognitiveCapability", [])

    candidates = []
    scorer.eval()
    with torch.no_grad():
        for t in task_nodes:
            generated = 0
            trials = 0
            while generated < max_per_task and trials < max_per_task * 6:
                trials += 1
                members = [t]
                if alg_nodes:
                    members.append(rng.choice(alg_nodes))
                if prompt_nodes:
                    members.append(rng.choice(prompt_nodes))
                if flow_nodes:
                    members.append(rng.choice(flow_nodes))
                if capability_nodes and rng.random() < 0.5:
                    members.append(rng.choice(capability_nodes))

                members = sorted(set(members))
                sig = signature(members)
                if sig in known_signatures:
                    continue

                member_tensor = node_embeddings[torch.tensor(members, dtype=torch.long, device=node_embeddings.device)]
                mean_vec = member_tensor.mean(dim=0)
                if edge_pooling == "mean":
                    vec = mean_vec
                else:
                    max_vec = member_tensor.max(dim=0).values
                    std_vec = member_tensor.std(dim=0, unbiased=False)
                    vec = torch.cat([mean_vec, max_vec, std_vec], dim=0)
                score = torch.sigmoid(scorer(vec.unsqueeze(0))).item()
                task_meta = metadata_by_idx[t]
                components, capabilities = _component_payload(members, metadata_by_idx)

                candidates.append({
                    "task": {
                        "id": task_meta["id"],
                        "name": task_meta["name"],
                    },
                    "score": float(score),
                    "shortlisted_components": components,
                    "supporting_capabilities": capabilities,
                    "member_node_ids": [metadata_by_idx[m]["id"] for m in members],
                })
                generated += 1

    candidates.sort(key=lambda x: x["score"], reverse=True)
    candidates = candidates[:top_k]

    by_use_case = {}
    for c in candidates:
        t_id = c["task"]["id"]
        t_name = c["task"]["name"]
        by_use_case.setdefault(
            t_id,
            {
                "task": {"id": t_id, "name": t_name},
                "shortlisted_technique_blueprints": [],
            },
        )
        by_use_case[t_id]["shortlisted_technique_blueprints"].append(
            {
                "score": c["score"],
                "shortlisted_components": c["shortlisted_components"],
                "supporting_capabilities": c["supporting_capabilities"],
            }
        )

    grouped = sorted(by_use_case.values(), key=lambda x: x["task"]["name"])
    for entry in grouped:
        entry["shortlisted_technique_blueprints"].sort(key=lambda x: x["score"], reverse=True)
        entry["shortlisted_technique_blueprints"] = entry["shortlisted_technique_blueprints"][:5]

    return candidates, grouped


def train_and_discover(
    graph_path: str = "data/hypergraph_output/hypergraph_latest.pkl",
    epochs: int = 200,
    lr: float = 1e-3,
    hidden_dim: int = 256,
    out_dim: int = 128,
    dropout: float = 0.2,
    test_ratio: float = 0.2,
    architecture: str = "hgnn",
    edge_pooling: str = "mean_max_std",
    num_negatives: int = 3,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> dict:
    set_seed(seed)
    run_dir = setup_run_dir()
    setup_logging(run_dir)

    data = load_hypergraph(graph_path)
    node_features: torch.Tensor = data["node_features"]
    hyperedge_index: torch.Tensor = data["hyperedge_index"]

    by_type, metadata_by_idx, node_types_by_idx = get_node_index_maps(
        data["node_metadata"], data["node_to_idx"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = node_features.to(device)
    hyperedge_index = hyperedge_index.to(device)

    all_memberships = build_membership_map(hyperedge_index.cpu())

    train_edges, test_edges, leakage_report = split_hyperedges_strict_member_disjoint(
        memberships=all_memberships,
        node_types_by_idx=node_types_by_idx,
        test_ratio=test_ratio,
        seed=seed,
    )
    if leakage_report["overlap_core_nodes"] != 0:
        raise RuntimeError("Leakage detected in protected node types; aborting training.")

    logging.info(
        "Strict split complete | train=%d test=%d overlap_core_nodes=%d",
        leakage_report["num_train_hyperedges"],
        leakage_report["num_test_hyperedges"],
        leakage_report["overlap_core_nodes"],
    )
    train_set = set(train_edges)

    train_hyperedge_index = filter_hyperedge_index(hyperedge_index.cpu(), train_set).to(device)

    train_memberships = {e: all_memberships[e] for e in train_edges if e in all_memberships}
    train_visible_nodes = sorted({n for members in train_memberships.values() for n in members})
    neg_memberships = make_negative_membership_typed(
        memberships=train_memberships,
        edge_indices=list(train_memberships.keys()),
        candidate_nodes=train_visible_nodes,
        node_types_by_idx=node_types_by_idx,
        num_negatives=num_negatives,
        seed=seed,
    )

    encoder = make_encoder(
        architecture=architecture,
        in_dim=x.shape[1],
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        dropout=dropout,
    ).to(device)

    edge_emb_dim = out_dim if edge_pooling == "mean" else out_dim * 3
    scorer = HyperedgeScorer(emb_dim=edge_emb_dim, hidden_dim=hidden_dim).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(scorer.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    train_edge_indices = sorted(train_memberships.keys())
    neg_edge_indices = sorted(neg_memberships.keys())

    for epoch in range(1, epochs + 1):
        encoder.train()
        scorer.train()
        optimizer.zero_grad()

        z = encoder(x, train_hyperedge_index)

        pos_vecs = aggregate_hyperedge_embeddings(
            z,
            train_memberships,
            train_edge_indices,
            mode=edge_pooling,
        )
        neg_vecs = aggregate_hyperedge_embeddings(
            z,
            neg_memberships,
            neg_edge_indices,
            mode=edge_pooling,
        )

        pos_logits = scorer(pos_vecs)
        neg_logits = scorer(neg_vecs)

        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat([
            torch.ones(pos_logits.shape[0], device=device),
            torch.zeros(neg_logits.shape[0], device=device),
        ])

        bce_loss = F.binary_cross_entropy_with_logits(logits, labels)

        # Pair each positive with one sampled negative for margin-based ranking.
        if pos_logits.shape[0] > 0 and neg_logits.shape[0] > 0:
            sampled_neg = neg_logits[:pos_logits.shape[0]]
            rank_loss = F.relu(1.0 - pos_logits + sampled_neg).mean()
        else:
            rank_loss = torch.tensor(0.0, device=device)

        loss = bce_loss + 0.2 * rank_loss
        loss.backward()
        nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(scorer.parameters()), max_norm=1.0)
        optimizer.step()

        if epoch % 25 == 0 or epoch == 1:
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                pred = (probs >= 0.5).float()
                acc = (pred == labels).float().mean().item()
            logging.info("Epoch %d | loss=%.4f | train_acc=%.4f", epoch, loss.item(), acc)

    encoder.eval()
    scorer.eval()

    with torch.no_grad():
        # Important: keep message passing on TRAIN structure only to avoid test leakage.
        z_all = encoder(x, train_hyperedge_index)

    # Test-set evaluation with the same metrics family used previously.
    test_memberships = {e: all_memberships[e] for e in test_edges if e in all_memberships}
    test_neg_memberships = make_negative_membership_typed(
        memberships=test_memberships,
        edge_indices=list(test_memberships.keys()),
        candidate_nodes=sorted({n for m in test_memberships.values() for n in m}),
        node_types_by_idx=node_types_by_idx,
        num_negatives=1,
        seed=seed + 1337,
    )

    test_edge_indices = sorted(test_memberships.keys())
    test_neg_indices = sorted(test_neg_memberships.keys())

    if test_edge_indices and test_neg_indices:
        with torch.no_grad():
            test_pos_vecs = aggregate_hyperedge_embeddings(z_all, test_memberships, test_edge_indices)
            test_pos_vecs = aggregate_hyperedge_embeddings(
                z_all,
                test_memberships,
                test_edge_indices,
                mode=edge_pooling,
            )
            test_neg_vecs = aggregate_hyperedge_embeddings(
                z_all,
                test_neg_memberships,
                test_neg_indices,
                mode=edge_pooling,
            )
            test_pos_scores = torch.sigmoid(scorer(test_pos_vecs)).detach().cpu()
            test_neg_scores = torch.sigmoid(scorer(test_neg_vecs)).detach().cpu()

        metrics = evaluate_link_prediction(test_pos_scores, test_neg_scores, k_list=[1, 3, 5, 10])
    else:
        metrics = {
            "Hits@1": 0.0,
            "Hits@3": 0.0,
            "Hits@5": 0.0,
            "Hits@10": 0.0,
            "MRR": 0.0,
            "AUC": 0.0,
            "AP": 0.0,
            "PairwiseAcc": 0.0,
        }

    log_metrics(metrics, prefix="HGNN Test")
    save_metrics(metrics, str(run_dir), model_name="hgnn")

    known_signatures = {signature(v) for v in all_memberships.values()}
    candidates, grouped_by_use_case = discover_candidates(
        node_embeddings=z_all,
        scorer=scorer,
        by_type=by_type,
        metadata_by_idx=metadata_by_idx,
        known_signatures=known_signatures,
        top_k=100,
        max_per_task=10,
        edge_pooling=edge_pooling,
    )

    output = {
        "run_dir": str(run_dir),
        "model": "HGNN(HypergraphConv)",
        "train_hyperedges": len(train_edges),
        "test_hyperedges": len(test_edges),
        "leakage_guard": leakage_report,
        "evaluation_metrics": metrics,
        "shortlists_by_use_case": grouped_by_use_case,
        "novel_techniques": candidates,
    }

    with (run_dir / "novel_techniques_hgnn.json").open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    with (run_dir / "hyperparameters.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "epochs": epochs,
                "lr": lr,
                "hidden_dim": hidden_dim,
                "out_dim": out_dim,
                "dropout": dropout,
                "test_ratio": test_ratio,
                "architecture": architecture,
                "edge_pooling": edge_pooling,
                "num_negatives": num_negatives,
                "weight_decay": weight_decay,
                "seed": seed,
            },
            f,
            indent=2,
        )

    logging.info("Saved HGNN output to %s", run_dir)
    return output


if __name__ == "__main__":
    os.makedirs("link_pred_output/HGNN", exist_ok=True)
    train_and_discover()
