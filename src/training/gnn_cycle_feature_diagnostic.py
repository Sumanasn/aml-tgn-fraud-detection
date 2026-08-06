"""Does the GNN benefit from the same closed-walk features that let plain
XGBoost beat it on time-to-detection? Right now FraudGNN only sees 9 basic
tabular features (degree, volume, balance) -- no explicit cyclic-structure
signal -- and relies entirely on message passing to discover structure. This
tests whether handing it the same explicit signal (on top of what it can
already learn) closes the gap.

Mirrors src/training/cycle_feature_diagnostic.py's methodology exactly, so
results are directly comparable. Writes to separate checkpoint/results
filenames -- doesn't touch the already-reported plain GNN results.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data_pipeline.build_graph import build_cumulative_snapshots, build_full_graph, load_raw, train_val_test_split
from src.models.gnn import FraudGNN
from src.training.metrics import evaluate

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
CKPT_DIR = Path(__file__).resolve().parents[2] / "checkpoints"
TOP_PCT_FLAGGED = 0.02


def train(
    dataset: str = "20K_cycle200",
    hidden_channels: int = 64,
    num_layers: int = 10,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 300,
    patience: int = 30,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = build_full_graph(dataset, include_cycle_features=True)
    train_mask, val_mask, test_mask = train_val_test_split(data.y)
    data = data.to(device)
    train_mask, val_mask, test_mask = train_mask.to(device), val_mask.to(device), test_mask.to(device)

    mean = data.x[train_mask].mean(dim=0, keepdim=True)
    std = data.x[train_mask].std(dim=0, keepdim=True).clamp_min(1e-6)
    data.x = (data.x - mean) / std

    model = FraudGNN(in_channels=data.x.size(1), hidden_channels=hidden_channels, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    y = data.y.float()
    pos_weight = (y[train_mask] == 0).sum() / (y[train_mask] == 1).sum().clamp_min(1)

    best_val_prauc = 0.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.binary_cross_entropy_with_logits(logits[train_mask], y[train_mask], pos_weight=pos_weight)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            val_scores = torch.sigmoid(logits[val_mask]).cpu().numpy()
            val_metrics = evaluate(y[val_mask].cpu().numpy(), val_scores)

        if val_metrics["pr_auc"] > best_val_prauc:
            best_val_prauc = val_metrics["pr_auc"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 20 == 0 or epoch == 1:
            print(f"epoch {epoch:3d} | loss {loss.item():.4f} | val PR-AUC {val_metrics['pr_auc']:.4f}")
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        test_scores = torch.sigmoid(logits[test_mask]).cpu().numpy()
    metrics = evaluate(y[test_mask].cpu().numpy(), test_scores)

    print("\nGraphSAGE GNN + closed-walk features (final accuracy):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "gnn_cycle_features.json", "w") as f:
        json.dump(metrics, f, indent=2)
    torch.save(
        {
            "state_dict": best_state, "hidden_channels": hidden_channels, "num_layers": num_layers,
            "feature_mean": mean.cpu(), "feature_std": std.cpu(),
        },
        CKPT_DIR / "gnn_cycle.pt",
    )
    return model, mean, std, metrics


def top_pct_flagged(scores: np.ndarray, pct: float) -> np.ndarray:
    k = max(1, int(len(scores) * pct))
    threshold = np.partition(scores, -k)[-k]
    return scores >= threshold


def time_to_detection(model, mean, std, dataset: str = "20K_cycle200", step_interval: int = 10):
    device = next(model.parameters()).device
    nodes, _ = load_raw(dataset)
    fraud_nodes = nodes[nodes["isFraud"] == 1].set_index("nodeid")["fraudStep"]

    snapshots = build_cumulative_snapshots(dataset, step_interval=step_interval, include_cycle_features=True)
    first_detected = {}
    model.eval()
    for cutoff, data in snapshots:
        x = ((data.x - mean.cpu()) / std.cpu()).to(device)
        with torch.no_grad():
            scores = torch.sigmoid(model(x, data.edge_index.to(device))).cpu().numpy()
        flagged = top_pct_flagged(scores, TOP_PCT_FLAGGED)
        for node_id in fraud_nodes.index:
            if node_id not in first_detected and flagged[node_id]:
                first_detected[node_id] = cutoff

    rows = []
    for node_id, fraud_step in fraud_nodes.items():
        if fraud_step < 0:
            continue
        step = first_detected.get(node_id)
        rows.append({"detected": step is not None, "lag": (step - fraud_step) if step is not None else None})

    detected_rows = [r for r in rows if r["detected"]]
    summary = {
        "n_fraud_nodes": len(rows),
        "detected_pct": len(detected_rows) / len(rows),
        "median_lag_steps": float(np.median([r["lag"] for r in detected_rows])) if detected_rows else None,
    }
    print("\nGraphSAGE GNN + closed-walk features (time-to-detection):")
    print(json.dumps(summary, indent=2))

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "time_to_detection_gnn_cycle_features.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    model, mean, std, metrics = train()
    time_to_detection(model, mean, std)
