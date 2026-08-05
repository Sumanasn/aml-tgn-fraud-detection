"""Temporal Graph Network (Rossi et al. 2020) for continuous-time fraud
detection, following PyTorch Geometric's reference TGN recipe.

Why this over the snapshot GraphSAGE model: that model's BatchNorm running
stats were calibrated on the final, densest graph and then misapplied to
sparse early snapshots during time-to-detection evaluation -- a real
distribution-shift confound. TGNMemory instead updates each node's memory
continuously as individual transactions arrive, so there's no "snapshot" to
mismatch: the memory state at step t IS the causal, as-of-t representation,
by construction.

Two heads share the same memory + embedding:
  - link prediction (self-supervised pretext task -- this is what gives the
    memory's GRU update function a gradient signal at all; TGN memory isn't
    trainable from a node-classification loss alone without it)
  - fraud classification (the actual task we care about, riding on top of
    the same embeddings)
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator, TGNMemory


class GraphAttentionEmbedding(torch.nn.Module):
    """Refines a node's memory using attention over its recent temporal
    neighbors (edges from LastNeighborLoader), time-encoded."""

    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = torch.nn.Linear(in_channels, in_channels)
        self.lin_dst = torch.nn.Linear(in_channels, in_channels)
        self.lin_out = torch.nn.Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = F.relu(h)
        return self.lin_out(h).squeeze(-1)


class FraudClassifierHead(torch.nn.Module):
    def __init__(self, embedding_dim, static_dim, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim + static_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, z, static_x):
        return self.net(torch.cat([z, static_x], dim=-1)).squeeze(-1)


class TGNFraudDetector(torch.nn.Module):
    """Bundles memory + embedding + both heads so checkpointing is one object."""

    def __init__(self, num_nodes: int, raw_msg_dim: int, static_dim: int, memory_dim: int = 100, time_dim: int = 100):
        super().__init__()
        self.memory = TGNMemory(
            num_nodes,
            raw_msg_dim,
            memory_dim,
            time_dim,
            message_module=IdentityMessage(raw_msg_dim, memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        )
        self.gnn = GraphAttentionEmbedding(
            in_channels=memory_dim,
            out_channels=memory_dim,
            msg_dim=raw_msg_dim,
            time_enc=self.memory.time_enc,
        )
        self.link_pred = LinkPredictor(memory_dim)
        self.classifier = FraudClassifierHead(memory_dim, static_dim)
