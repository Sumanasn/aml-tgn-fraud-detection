"""Chronological event-stream view of AMLSim for TGN training.

Unlike build_graph.py (which produces static/cumulative-snapshot PyG Data
objects), this keeps the transaction log as an ordered event stream --
TGNMemory consumes it one chronological batch at a time and updates each
node's memory as events arrive, which is what lets it avoid the
train/inference distribution-shift problem the snapshot GraphSAGE model had.
"""
from pathlib import Path

import torch
from torch_geometric.data import TemporalData

from src.data_pipeline.build_graph import load_raw

DATA_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def build_temporal_data(dataset: str = "20K_cycle200"):
    nodes, tx = load_raw(dataset)
    tx = tx.sort_values("time").reset_index(drop=True)

    src = torch.tensor(tx["sourceNodeId"].values, dtype=torch.long)
    dst = torch.tensor(tx["targetNodeId"].values, dtype=torch.long)
    t = torch.tensor(tx["time"].values, dtype=torch.long)

    value = torch.tensor(tx["value"].values, dtype=torch.float).unsqueeze(-1)
    value = (value - value.mean()) / value.std().clamp_min(1e-6)
    msg = value  # raw_msg_dim = 1; extend here if more edge attrs are added later

    data = TemporalData(src=src, dst=dst, t=t, msg=msg)
    # NOTE: don't set data.num_nodes explicitly -- TemporalData.index_select
    # iterates all attributes expecting per-event tensors, and a plain int
    # attribute breaks that. Confirmed inferred num_nodes (max(src,dst)+1)
    # already matches len(nodes) for this dataset; assert it instead.
    assert data.num_nodes == len(nodes), (
        f"TemporalData inferred num_nodes={data.num_nodes} != {len(nodes)} nodes in nodes.csv "
        "(some node never appears as src/dst) -- negative sampling range would be wrong."
    )

    nodes_sorted = nodes.set_index("nodeid").sort_index()
    static_x = torch.tensor(nodes_sorted[["init_balance"]].values, dtype=torch.float)
    static_x = (static_x - static_x.mean()) / static_x.std().clamp_min(1e-6)
    y = torch.tensor(nodes_sorted["isFraud"].values, dtype=torch.float)
    fraud_step = torch.tensor(nodes_sorted["fraudStep"].values, dtype=torch.long)

    num_nodes = len(nodes_sorted)
    return data, static_x, y, fraud_step, num_nodes


if __name__ == "__main__":
    data, static_x, y, fraud_step, num_nodes = build_temporal_data()
    print(data)
    print("num_nodes:", num_nodes, "| static_x:", static_x.shape, "| fraud:", int(y.sum()))
    print("time range:", int(data.t.min()), "-", int(data.t.max()))
