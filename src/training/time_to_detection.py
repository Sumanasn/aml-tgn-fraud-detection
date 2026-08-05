"""The core portfolio story: replay the transaction stream and ask, for each
fraud node, "at what step does each model's score first cross a suspicion
threshold?" vs. the ground-truth fraudStep (when AMLSim actually wired that
node into a laundering pattern).

Three models, two different replay strategies:
  - XGBoost / GraphSAGE GNN: re-scored from scratch on cumulative snapshots
    (they have no internal state -- this is the only way to ask "what would
    this frozen model have said at step t").
  - TGN: a single causal, chronological replay of the whole stream. Its
    memory *is* an as-of-t representation by construction, so we just
    snapshot its running score buffer at the same step_interval boundaries
    instead of recomputing anything from scratch.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import xgboost as xgb
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn.models.tgn import LastNeighborLoader

from src.data_pipeline.build_graph import build_cumulative_snapshots, load_raw
from src.data_pipeline.temporal_data import build_temporal_data
from src.models.gnn import FraudGNN
from src.models.tgn import TGNFraudDetector

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
CKPT_DIR = Path(__file__).resolve().parents[2] / "checkpoints"

TOP_PCT_FLAGGED = 0.02  # flag the top 2% most suspicious nodes at each snapshot


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


def load_baseline():
    model = xgb.XGBClassifier()
    model.load_model(CKPT_DIR / "xgboost.json")
    return model


def load_tgn(device):
    ckpt = torch.load(CKPT_DIR / "tgn.pt", weights_only=False)
    model = TGNFraudDetector(
        num_nodes=ckpt["num_nodes"],
        raw_msg_dim=ckpt["raw_msg_dim"],
        static_dim=ckpt["static_dim"],
        memory_dim=ckpt["memory_dim"],
        time_dim=ckpt["time_dim"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["neighbor_size"]


def gnn_scores(model, mean, std, data):
    x = (data.x - mean) / std
    with torch.no_grad():
        logits = model(x, data.edge_index)
    return torch.sigmoid(logits).numpy()


def baseline_scores(model, data):
    return model.predict_proba(data.x.numpy())[:, 1]


def top_pct_flagged(scores: np.ndarray, pct: float) -> np.ndarray:
    k = max(1, int(len(scores) * pct))
    threshold = np.partition(scores, -k)[-k]
    return scores >= threshold


def snapshot_first_detected(dataset, step_interval, fraud_ids, gnn_model, mean, std, baseline_model):
    snapshots = build_cumulative_snapshots(dataset, step_interval=step_interval)
    first_detected_gnn, first_detected_baseline = {}, {}

    for cutoff, data in snapshots:
        g_flagged = top_pct_flagged(gnn_scores(gnn_model, mean, std, data), TOP_PCT_FLAGGED)
        b_flagged = top_pct_flagged(baseline_scores(baseline_model, data), TOP_PCT_FLAGGED)
        for node_id in fraud_ids:
            if node_id not in first_detected_gnn and g_flagged[node_id]:
                first_detected_gnn[node_id] = cutoff
            if node_id not in first_detected_baseline and b_flagged[node_id]:
                first_detected_baseline[node_id] = cutoff

    return first_detected_gnn, first_detected_baseline, snapshots[-1][0]


@torch.no_grad()
def tgn_causal_first_detected(dataset, step_interval, fraud_ids, tgn_model, neighbor_size, device, batch_size=200):
    data, static_x, y, fraud_step, num_nodes = build_temporal_data(dataset)
    static_x = static_x.to(device)

    neighbor_loader = LastNeighborLoader(num_nodes, size=neighbor_size, device=device)
    assoc = torch.empty(num_nodes, dtype=torch.long, device=device)
    scores = torch.zeros(num_nodes, device=device)

    tgn_model.eval()
    tgn_model.memory.reset_state()
    neighbor_loader.reset_state()

    loader = TemporalDataLoader(data, batch_size=batch_size)
    first_detected = {}
    max_t_seen = 0

    for batch in loader:
        src, pos_dst, t, msg = (x.to(device) for x in (batch.src, batch.dst, batch.t, batch.msg))

        touched = torch.cat([src, pos_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(touched)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = tgn_model.memory(n_id)
        z = tgn_model.gnn(z, last_update, edge_index, data.t[e_id].to(device), data.msg[e_id].to(device))
        logits = tgn_model.classifier(z[assoc[touched]], static_x[touched])
        scores[touched] = torch.sigmoid(logits)

        tgn_model.memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        batch_max_t = int(t.max())
        # Snapshot the persistent score buffer whenever we cross a step_interval boundary.
        if batch_max_t // step_interval > max_t_seen // step_interval:
            cutoff = (batch_max_t // step_interval) * step_interval
            flagged = top_pct_flagged(scores.cpu().numpy(), TOP_PCT_FLAGGED)
            for node_id in fraud_ids:
                if node_id not in first_detected and flagged[node_id]:
                    first_detected[node_id] = cutoff
        max_t_seen = batch_max_t

    return first_detected, max_t_seen


def main(dataset: str = "20K_cycle200", step_interval: int = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nodes, _ = load_raw(dataset)
    fraud_nodes = nodes[nodes["isFraud"] == 1].set_index("nodeid")["fraudStep"]
    fraud_ids = fraud_nodes.index

    gnn_model, mean, std = load_gnn()
    baseline_model = load_baseline()
    first_detected_gnn, first_detected_baseline, max_step = snapshot_first_detected(
        dataset, step_interval, fraud_ids, gnn_model, mean, std, baseline_model
    )

    first_detected_tgn = {}
    tgn_path = CKPT_DIR / "tgn.pt"
    if tgn_path.exists():
        tgn_model, neighbor_size = load_tgn(device)
        first_detected_tgn, tgn_max_step = tgn_causal_first_detected(
            dataset, step_interval, fraud_ids, tgn_model, neighbor_size, device
        )
        max_step = max(max_step, tgn_max_step)
    else:
        print("No TGN checkpoint found -- skipping TGN in comparison (train it with src/training/train_tgn.py).")

    rows = []
    for node_id, fraud_step in fraud_nodes.items():
        if fraud_step < 0:
            continue
        g_step = first_detected_gnn.get(node_id, max_step + step_interval)
        b_step = first_detected_baseline.get(node_id, max_step + step_interval)
        row = {"node_id": node_id, "fraud_step": fraud_step, "gnn_lag": g_step - fraud_step, "baseline_lag": b_step - fraud_step}
        if first_detected_tgn:
            t_step = first_detected_tgn.get(node_id, max_step + step_interval)
            row["tgn_lag"] = t_step - fraud_step
        rows.append(row)

    rows.sort(key=lambda r: r["fraud_step"])
    fraud_step_arr = np.array([r["fraud_step"] for r in rows])
    gnn_lag = np.array([r["gnn_lag"] for r in rows])
    baseline_lag = np.array([r["baseline_lag"] for r in rows])

    summary = {
        "n_fraud_nodes": len(rows),
        "gnn_median_lag_steps": float(np.median(gnn_lag)),
        "baseline_median_lag_steps": float(np.median(baseline_lag)),
        "gnn_faster_than_baseline_pct": float((gnn_lag < baseline_lag).mean()),
    }
    if first_detected_tgn:
        tgn_lag = np.array([r["tgn_lag"] for r in rows])
        summary["tgn_median_lag_steps"] = float(np.median(tgn_lag))
        summary["tgn_faster_than_baseline_pct"] = float((tgn_lag < baseline_lag).mean())
        summary["tgn_faster_than_gnn_pct"] = float((tgn_lag < gnn_lag).mean())

    print(json.dumps(summary, indent=2))

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "time_to_detection.json", "w") as f:
        json.dump(summary, f, indent=2)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(fraud_step_arr, gnn_lag, s=14, alpha=0.6, label="GraphSAGE GNN (snapshot)", color="#2563eb")
    ax.scatter(fraud_step_arr, baseline_lag, s=14, alpha=0.6, label="XGBoost baseline", color="#dc2626")
    if first_detected_tgn:
        ax.scatter(fraud_step_arr, tgn_lag, s=14, alpha=0.6, label="TGN (causal online)", color="#16a34a")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, label="fraudStep (ground truth)")
    ax.set_xlabel("Ground-truth fraudStep (when node entered the laundering pattern)")
    ax.set_ylabel("Detection lag (steps after fraudStep; negative = detected early)")
    ax.set_title("Time-to-detection: GNN vs. TGN vs. tabular baseline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "time_to_detection.png", dpi=150)
    print(f"\nSaved plot to {RESULTS_DIR / 'time_to_detection.png'}")


if __name__ == "__main__":
    main()
