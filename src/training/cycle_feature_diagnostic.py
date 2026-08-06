"""Diagnostic: does explicit structural information (closed-walk counts, not
learned representations) let a tabular model detect laundering rings
earlier than the plain XGBoost baseline?

This isolates the question the GNN/TGN results couldn't answer on their
own: is the missing time-to-detection edge a *data* limitation (the timing
signal genuinely isn't there -- AMLSim's cycles burst concurrently) or a
*model* limitation (the signal exists, but GraphSAGE/TGN weren't extracting
it within the training budget used)? Feeding the structural feature
directly to XGBoost -- no representation learning involved -- tests for the
existence of the signal, independent of any model's ability to learn it.
"""
import json
from pathlib import Path

import numpy as np
import xgboost as xgb

from src.data_pipeline.build_graph import build_cumulative_snapshots, build_full_graph, train_val_test_split
from src.data_pipeline.cycle_features import closed_walk_features
from src.training.metrics import evaluate

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
CKPT_DIR = Path(__file__).resolve().parents[2] / "checkpoints"
TOP_PCT_FLAGGED = 0.02


def augmented_features(data):
    cw = closed_walk_features(data.edge_index, data.num_nodes)
    return np.concatenate([data.x.numpy(), cw], axis=1)


def train(dataset: str = "20K_cycle200"):
    data = build_full_graph(dataset)
    X = augmented_features(data)
    y = data.y.numpy()
    train_mask, val_mask, test_mask = train_val_test_split(data.y)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=(y[train_mask.numpy()] == 0).sum() / max((y[train_mask.numpy()] == 1).sum(), 1),
        eval_metric="aucpr",
        n_jobs=-1,
    )
    model.fit(
        X[train_mask.numpy()], y[train_mask.numpy()],
        eval_set=[(X[val_mask.numpy()], y[val_mask.numpy()])], verbose=False,
    )

    test_scores = model.predict_proba(X[test_mask.numpy()])[:, 1]
    metrics = evaluate(y[test_mask.numpy()], test_scores)
    print("XGBoost + closed-walk features (final accuracy):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "baseline_cycle_features.json", "w") as f:
        json.dump(metrics, f, indent=2)
    model.save_model(CKPT_DIR / "xgboost_cycle.json")
    return model, metrics


def top_pct_flagged(scores: np.ndarray, pct: float) -> np.ndarray:
    k = max(1, int(len(scores) * pct))
    threshold = np.partition(scores, -k)[-k]
    return scores >= threshold


def time_to_detection(model, dataset: str = "20K_cycle200", step_interval: int = 10):
    from src.data_pipeline.build_graph import load_raw

    nodes, _ = load_raw(dataset)
    fraud_nodes = nodes[nodes["isFraud"] == 1].set_index("nodeid")["fraudStep"]

    snapshots = build_cumulative_snapshots(dataset, step_interval=step_interval)
    first_detected = {}
    for cutoff, data in snapshots:
        X = augmented_features(data)
        scores = model.predict_proba(X)[:, 1]
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
    print("\nXGBoost + closed-walk features (time-to-detection):")
    print(json.dumps(summary, indent=2))

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "time_to_detection_cycle_features.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    model, metrics = train()
    time_to_detection(model)
