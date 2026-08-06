"""How to actually pick a Tier-1 flagging threshold: sweep it and measure
both sides of the real tradeoff, not just detection speed in isolation.

- Precision@threshold on the held-out test set: of the accounts flagged,
  what fraction are actually fraud? This is the customer-friction cost --
  low precision means holding/step-up-authenticating a lot of legitimate
  transactions to catch each real one.
- detected_pct / median_lag_steps via the same causal snapshot replay used
  throughout this project: the speed/coverage benefit at that threshold.

Answers "what's the best way to solve the problem" concretely: a threshold
choice is a friction-vs-speed tradeoff, and this makes that tradeoff visible
instead of assumed.
"""
import json
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb

from src.data_pipeline.build_graph import build_cumulative_snapshots, build_full_graph, load_raw, train_val_test_split
from src.models.gnn import FraudGNN

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
CKPT_DIR = Path(__file__).resolve().parents[2] / "checkpoints"

THRESHOLDS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]


def top_pct_flagged(scores: np.ndarray, pct: float) -> np.ndarray:
    k = max(1, int(len(scores) * pct))
    threshold = np.partition(scores, -k)[-k]
    return scores >= threshold


def precision_at_thresholds(scores: np.ndarray, y: np.ndarray, thresholds) -> dict:
    out = {}
    for pct in thresholds:
        flagged = top_pct_flagged(scores, pct)
        n_flagged = int(flagged.sum())
        n_fraud_flagged = int((flagged & (y == 1)).sum())
        out[pct] = {
            "n_flagged": n_flagged,
            "precision": n_fraud_flagged / n_flagged if n_flagged else 0.0,
            "recall": n_fraud_flagged / max(int((y == 1).sum()), 1),
        }
    return out


def speed_at_thresholds(snapshot_scores, fraud_nodes, thresholds, max_step, step_interval) -> dict:
    """snapshot_scores: list of (cutoff, scores_array), precomputed once and
    reused across all threshold values (the expensive part -- feature
    computation + model scoring per snapshot -- only happens once)."""
    out = {}
    for pct in thresholds:
        first_detected = {}
        for cutoff, scores in snapshot_scores:
            flagged = top_pct_flagged(scores, pct)
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
        out[pct] = {
            "detected_pct": len(detected_rows) / len(rows),
            "median_lag_steps": float(np.median([r["lag"] for r in detected_rows])) if detected_rows else None,
        }
    return out


def run_xgboost(dataset: str = "20K_cycle200", step_interval: int = 10):
    from src.data_pipeline.cycle_features import closed_walk_features

    model = xgb.XGBClassifier()
    model.load_model(CKPT_DIR / "xgboost_cycle.json")

    data = build_full_graph(dataset, include_cycle_features=True)
    _, _, test_mask = train_val_test_split(data.y)
    full_scores = model.predict_proba(data.x.numpy())[:, 1]
    precision = precision_at_thresholds(full_scores[test_mask.numpy()], data.y.numpy()[test_mask.numpy()], THRESHOLDS)

    nodes, _ = load_raw(dataset)
    fraud_nodes = nodes[nodes["isFraud"] == 1].set_index("nodeid")["fraudStep"]
    snapshots = build_cumulative_snapshots(dataset, step_interval=step_interval, include_cycle_features=True)
    snapshot_scores = [(cutoff, model.predict_proba(d.x.numpy())[:, 1]) for cutoff, d in snapshots]
    speed = speed_at_thresholds(snapshot_scores, fraud_nodes, THRESHOLDS, snapshots[-1][0], step_interval)

    return precision, speed


def run_gnn(dataset: str = "20K_cycle200", step_interval: int = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CKPT_DIR / "gnn_cycle.pt", weights_only=False)
    model = FraudGNN(
        in_channels=ckpt["feature_mean"].shape[1], hidden_channels=ckpt["hidden_channels"], num_layers=ckpt["num_layers"]
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    mean, std = ckpt["feature_mean"], ckpt["feature_std"]

    data = build_full_graph(dataset, include_cycle_features=True)
    _, _, test_mask = train_val_test_split(data.y)
    x = ((data.x - mean) / std).to(device)
    with torch.no_grad():
        full_scores = torch.sigmoid(model(x, data.edge_index.to(device))).cpu().numpy()
    precision = precision_at_thresholds(full_scores[test_mask.numpy()], data.y.numpy()[test_mask.numpy()], THRESHOLDS)

    nodes, _ = load_raw(dataset)
    fraud_nodes = nodes[nodes["isFraud"] == 1].set_index("nodeid")["fraudStep"]
    snapshots = build_cumulative_snapshots(dataset, step_interval=step_interval, include_cycle_features=True)
    snapshot_scores = []
    for cutoff, d in snapshots:
        xs = ((d.x - mean) / std).to(device)
        with torch.no_grad():
            s = torch.sigmoid(model(xs, d.edge_index.to(device))).cpu().numpy()
        snapshot_scores.append((cutoff, s))
    speed = speed_at_thresholds(snapshot_scores, fraud_nodes, THRESHOLDS, snapshots[-1][0], step_interval)

    return precision, speed


def main():
    print("=== XGBoost + closed-walk features ===")
    xgb_precision, xgb_speed = run_xgboost()
    print("\n=== GraphSAGE GNN + closed-walk features ===")
    gnn_precision, gnn_speed = run_gnn()

    results = {"xgboost_cycle": {}, "gnn_cycle": {}}
    print(f"\n{'threshold':>10} | {'model':<12} | {'precision':>9} | {'recall':>7} | {'detected%':>9} | {'median_lag':>10}")
    for pct in THRESHOLDS:
        for name, precision, speed in (("xgboost_cycle", xgb_precision, xgb_speed), ("gnn_cycle", gnn_precision, gnn_speed)):
            p, s = precision[pct], speed[pct]
            print(
                f"{pct:>10.0%} | {name:<12} | {p['precision']:>9.3f} | {p['recall']:>7.3f} | "
                f"{s['detected_pct']:>9.1%} | {str(s['median_lag_steps']):>10}"
            )
            results[name][pct] = {**p, **s}

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "threshold_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR / 'threshold_sweep.json'}")


if __name__ == "__main__":
    main()
