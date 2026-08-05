"""GNNExplainer: for a flagged fraud node, extract the exact subgraph and
edges the GraphSAGE GNN's prediction was actually driven by. This is the
operational payoff of using a GNN over XGBoost -- a tabular model can say
"this account is suspicious" but not "here is the transfer ring".
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig

from src.data_pipeline.build_graph import build_full_graph, train_val_test_split
from src.models.gnn import FraudGNN

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
CKPT_DIR = Path(__file__).resolve().parents[2] / "checkpoints"

TOP_K_EDGES = 25


def load_gnn():
    ckpt = torch.load(CKPT_DIR / "gnn.pt", weights_only=False)
    model = FraudGNN(
        in_channels=ckpt["feature_mean"].shape[1],
        hidden_channels=ckpt["hidden_channels"],
        num_layers=ckpt["num_layers"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["feature_mean"], ckpt["feature_std"]


def pick_target_nodes(model, x, edge_index, y, test_mask, n: int = 3):
    """High-confidence true positives -- nodes the model is confident are
    fraud, and actually are. These are the ones worth explaining."""
    with torch.no_grad():
        scores = torch.sigmoid(model(x, edge_index))
    candidates = (test_mask & (y == 1)).nonzero(as_tuple=True)[0]
    candidate_scores = scores[candidates]
    top = candidates[torch.argsort(candidate_scores, descending=True)[:n]]
    return top.tolist(), scores


def explain_node(explainer, x, edge_index, node_idx: int):
    explanation = explainer(x, edge_index, index=node_idx)
    edge_mask = explanation.edge_mask
    top_edge_idx = torch.argsort(edge_mask, descending=True)[:TOP_K_EDGES]
    sub_edges = edge_index[:, top_edge_idx]
    sub_weights = edge_mask[top_edge_idx]
    return sub_edges, sub_weights


def plot_subgraph(node_idx: int, sub_edges, sub_weights, y, out_path: Path):
    G = nx.DiGraph()
    for (s, d), w in zip(sub_edges.t().tolist(), sub_weights.tolist()):
        G.add_edge(s, d, weight=w)

    if node_idx not in G:
        G.add_node(node_idx)

    pos = nx.spring_layout(G, seed=42, k=0.8)
    node_colors = ["#dc2626" if y[n] == 1 else "#94a3b8" for n in G.nodes()]
    node_sizes = [500 if n == node_idx else 200 for n in G.nodes()]
    edge_widths = [1 + 4 * w for w in nx.get_edge_attributes(G, "weight").values()]

    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color="#2563eb", alpha=0.6, arrows=True, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)
    ax.set_title(f"GNNExplainer subgraph for node {node_idx} (target, larger) -- red = fraud-labeled")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(dataset: str = "20K_cycle200"):
    model, mean, std = load_gnn()
    data = build_full_graph(dataset)
    x = (data.x - mean) / std
    _, _, test_mask = train_val_test_split(data.y)

    target_nodes, scores = pick_target_nodes(model, x, data.edge_index, data.y, test_mask)
    print(f"Explaining {len(target_nodes)} high-confidence fraud nodes: {target_nodes}")

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=ModelConfig(mode="binary_classification", task_level="node", return_type="raw"),
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    summary = []
    for node_idx in target_nodes:
        sub_edges, sub_weights = explain_node(explainer, x, data.edge_index, node_idx)
        out_path = RESULTS_DIR / f"explain_node_{node_idx}.png"
        plot_subgraph(node_idx, sub_edges, sub_weights, data.y, out_path)

        involved_nodes = sub_edges.unique().tolist()
        fraud_in_subgraph = int(data.y[involved_nodes].sum())
        summary.append(
            {
                "node_id": node_idx,
                "model_score": float(scores[node_idx]),
                "top_edges": sub_edges.t().tolist(),
                "edge_importance": sub_weights.tolist(),
                "nodes_in_subgraph": len(involved_nodes),
                "fraud_nodes_in_subgraph": fraud_in_subgraph,
            }
        )
        print(
            f"node {node_idx}: score={scores[node_idx]:.3f} | "
            f"{len(involved_nodes)} nodes in explanatory subgraph, {fraud_in_subgraph} fraud-labeled | "
            f"saved {out_path.name}"
        )

    with open(RESULTS_DIR / "explanations.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
