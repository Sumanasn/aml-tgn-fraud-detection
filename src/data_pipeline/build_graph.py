"""Turn AMLSim nodes.csv/transactions.csv into PyG graphs.

Two products:
  - build_full_graph(): one static Data object (all edges) for training GNN vs
    XGBoost baseline with a standard train/val/test node split.
  - build_cumulative_snapshots(): a sequence of Data objects, each containing
    only edges with time <= step, used for the time-to-detection comparison
    (does the anomaly score cross threshold before fraudStep?).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

DATA_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def load_raw(dataset: str = "20K_cycle200"):
    d = DATA_RAW_DIR / dataset
    nodes = pd.read_csv(d / "nodes.csv")
    tx = pd.read_csv(d / "transactions.csv")
    return nodes, tx


def _node_features(nodes: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    """Structural + volume features engineered from the transaction log.

    No forward-looking info: for a snapshot at step T, only pass tx rows with
    time <= T into this function.
    """
    n = nodes.set_index("nodeid").copy()
    n["out_degree"] = 0
    n["in_degree"] = 0
    n["out_value"] = 0.0
    n["in_value"] = 0.0
    n["tx_count"] = 0
    n["active_steps"] = 0

    if len(tx):
        out_g = tx.groupby("sourceNodeId")
        in_g = tx.groupby("targetNodeId")

        n["out_degree"] = out_g["targetNodeId"].nunique().reindex(n.index, fill_value=0)
        n["in_degree"] = in_g["sourceNodeId"].nunique().reindex(n.index, fill_value=0)
        n["out_value"] = out_g["value"].sum().reindex(n.index, fill_value=0.0)
        n["in_value"] = in_g["value"].sum().reindex(n.index, fill_value=0.0)

        out_cnt = out_g.size().reindex(n.index, fill_value=0)
        in_cnt = in_g.size().reindex(n.index, fill_value=0)
        n["tx_count"] = out_cnt + in_cnt

        out_steps = out_g["time"].nunique().reindex(n.index, fill_value=0)
        in_steps = in_g["time"].nunique().reindex(n.index, fill_value=0)
        n["active_steps"] = np.maximum(out_steps, in_steps)

    n["avg_tx_value"] = (n["out_value"] + n["in_value"]) / n["tx_count"].replace(0, 1)
    n["net_flow"] = n["in_value"] - n["out_value"]

    feature_cols = [
        "init_balance",
        "out_degree",
        "in_degree",
        "out_value",
        "in_value",
        "tx_count",
        "active_steps",
        "avg_tx_value",
        "net_flow",
    ]
    return n[feature_cols]


def _to_data(nodes: pd.DataFrame, tx: pd.DataFrame, include_cycle_features: bool = False) -> Data:
    feats = _node_features(nodes, tx)
    x = torch.tensor(feats.values, dtype=torch.float)
    y = torch.tensor(nodes.set_index("nodeid")["isFraud"].values, dtype=torch.long)

    edge_index = torch.tensor(
        tx[["sourceNodeId", "targetNodeId"]].values.T, dtype=torch.long
    )
    edge_attr = torch.tensor(tx[["value", "time"]].values, dtype=torch.float)

    if include_cycle_features:
        # Computed from this call's edge_index only -- when called per cumulative
        # snapshot (build_cumulative_snapshots), that's already restricted to
        # time <= cutoff, so this stays causal (no future-edge leakage).
        from src.data_pipeline.cycle_features import closed_walk_features

        cw = closed_walk_features(edge_index, num_nodes=len(nodes))
        x = torch.cat([x, torch.tensor(cw, dtype=torch.float)], dim=1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def build_full_graph(dataset: str = "20K_cycle200", include_cycle_features: bool = False) -> Data:
    nodes, tx = load_raw(dataset)
    return _to_data(nodes, tx, include_cycle_features=include_cycle_features)


def build_cumulative_snapshots(
    dataset: str = "20K_cycle200", step_interval: int = 10, include_cycle_features: bool = False
) -> list[tuple[int, Data]]:
    nodes, tx = load_raw(dataset)
    max_step = int(tx["time"].max())
    snapshots = []
    for cutoff in range(step_interval, max_step + step_interval, step_interval):
        tx_upto = tx[tx["time"] <= cutoff]
        snapshots.append((cutoff, _to_data(nodes, tx_upto, include_cycle_features=include_cycle_features)))
    return snapshots


def train_val_test_split(y: torch.Tensor, seed: int = 42):
    """Stratified split preserving the fraud ratio in each split."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    fraud_idx = idx[y.numpy() == 1]
    normal_idx = idx[y.numpy() == 0]
    rng.shuffle(fraud_idx)
    rng.shuffle(normal_idx)

    def split(arr):
        n = len(arr)
        n_train = int(n * 0.6)
        n_val = int(n * 0.2)
        return arr[:n_train], arr[n_train : n_train + n_val], arr[n_train + n_val :]

    f_train, f_val, f_test = split(fraud_idx)
    n_train, n_val, n_test = split(normal_idx)

    train_mask = torch.zeros(len(y), dtype=torch.bool)
    val_mask = torch.zeros(len(y), dtype=torch.bool)
    test_mask = torch.zeros(len(y), dtype=torch.bool)

    train_mask[np.concatenate([f_train, n_train])] = True
    val_mask[np.concatenate([f_val, n_val])] = True
    test_mask[np.concatenate([f_test, n_test])] = True
    return train_mask, val_mask, test_mask


if __name__ == "__main__":
    data = build_full_graph()
    print(data)
    print("fraud nodes:", int(data.y.sum()), "/", data.num_nodes)
    train_mask, val_mask, test_mask = train_val_test_split(data.y)
    print("train/val/test:", train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())
