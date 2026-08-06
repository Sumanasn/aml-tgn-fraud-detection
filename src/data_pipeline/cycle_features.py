"""Explicit structural feature: closed-walk counts via sparse adjacency
matrix powers, trace(A^k)_i = (A^k)_ii = count of length-k walks that leave
node i and return to it. Strongly correlated with "this node sits on a
cycle" without needing exact cycle enumeration (which we already found
explodes combinatorially on this graph, see src/models/gnn.py docstring).

Purpose: this is a diagnostic, not a model. If handing this feature
directly to XGBoost (no learning of *how* to use structure, just raw
availability of the information) doesn't improve time-to-detection over the
plain tabular baseline, that's evidence the timing signal isn't in the data
at all -- not that GraphSAGE/TGN failed to learn it.

Feasibility note: exact A^k blows up in fill-in as k grows (this graph's
~6-hop neighborhoods already cover a large fraction of all 20K nodes, a
small-world effect) -- computing all the way to k=12 densifies into an
effectively dense 20000x20000 matrix, which is infeasible. nnz_cap_ratio
stops the power iteration once a snapshot's graph is dense enough that
further hops stop being informative anyway; remaining columns are left at 0.
Early (sparse) snapshots -- the ones that actually matter for
time-to-detection -- are the least affected by this cap.
"""
import numpy as np
import scipy.sparse as sp
import torch

DEFAULT_MAX_K = 8
DEFAULT_NNZ_CAP_RATIO = 0.05


def closed_walk_features(
    edge_index: torch.Tensor,
    num_nodes: int,
    max_k: int = DEFAULT_MAX_K,
    nnz_cap_ratio: float = DEFAULT_NNZ_CAP_RATIO,
) -> np.ndarray:
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    A = sp.coo_matrix(
        (np.ones(len(src), dtype=np.float32), (src, dst)), shape=(num_nodes, num_nodes)
    ).tocsr()
    nnz_cap = int(num_nodes * num_nodes * nnz_cap_ratio)

    feats = np.zeros((num_nodes, max_k - 1), dtype=np.float32)  # columns for k=2..max_k
    Ak = A
    for i, k in enumerate(range(2, max_k + 1)):
        if Ak.nnz > nnz_cap:
            break  # too dense for further hops to add information cheaply; leave rest at 0
        Ak = Ak @ A
        feats[:, i] = np.asarray(Ak.diagonal()).ravel()

    # Closed-walk counts are heavily right-skewed (a handful of nodes on
    # dense cycles can hit 1000+ while most are 0) -- fine for tree splits
    # (scale-invariant), but z-score normalizing raw counts like this feeds
    # a neural net occasional extreme outlier values that blow up gradients
    # through a deep residual stack. log1p compresses the tail while
    # preserving rank order, independent of any model or result.
    feats = np.log1p(feats)

    return feats


if __name__ == "__main__":
    from src.data_pipeline.build_graph import build_full_graph

    data = build_full_graph()
    feats = closed_walk_features(data.edge_index, data.num_nodes)
    fraud_mask = data.y.numpy() == 1
    print("closed-walk feature shape:", feats.shape)
    print("mean (fraud nodes):  ", feats[fraud_mask].mean(axis=0))
    print("mean (normal nodes): ", feats[~fraud_mask].mean(axis=0))
