"""GraphSAGE-based node classifier.

The original spec called for R-GCN, which is built for multiple edge types
(TRANSFERS_TO / SHARES_DIRECTOR / OWNED_BY). The AMLSim data we're training
on Phase 1 is a single-relation transaction graph, so R-GCN would degenerate
to this anyway. Swap SAGEConv for RGCNConv + edge_type once a heterogeneous
graph (Phase 2) is available; the training loop doesn't need to change.

Depth note: cycle enumeration on the fraud-node subgraph showed most embedded
laundering cycles span ~10-12 hops. A plain 3-layer stack can't see past 3
hops, so it can't represent "this account sits on a long cycle" -- it just
re-derives a smoothed version of the same volume/degree signal XGBoost
already has, and loses the time-to-detection advantage. Residual connections
+ JumpingKnowledge let us go to 10-12 layers (receptive field >= cycle
length) without the oversmoothing that plain deep GCN/SAGE stacks suffer
from. The JK-concatenated output keeps both the early (local volume/degree)
and late (long-range cycle) layer representations available to the
classifier head instead of collapsing to only the last layer's output.
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import JumpingKnowledge, SAGEConv


class FraudGNN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 10,
        dropout: float = 0.3,
        jk_mode: str = "cat",
    ):
        super().__init__()
        self.dropout = dropout
        self.input_proj = torch.nn.Linear(in_channels, hidden_channels)
        self.convs = torch.nn.ModuleList(
            [SAGEConv(hidden_channels, hidden_channels) for _ in range(num_layers)]
        )
        self.norms = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)]
        )
        self.jk = JumpingKnowledge(mode=jk_mode, channels=hidden_channels, num_layers=num_layers)
        jk_out_channels = hidden_channels * num_layers if jk_mode == "cat" else hidden_channels
        # MLP head (not a single Linear): the JK output packs together
        # early-layer (local volume/degree) and late-layer (long-range
        # cycle) representations, and a hidden layer lets the classifier
        # actually combine those instead of just linearly re-weighting them.
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(jk_out_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, 1),
        )

    def _layers(self, x, edge_index):
        x = F.relu(self.input_proj(x))
        layer_outputs = []
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h  # residual: keeps gradient/signal flowing at depth
            layer_outputs.append(x)
        return layer_outputs

    def forward(self, x, edge_index):
        layer_outputs = self._layers(x, edge_index)
        out = self.jk(layer_outputs)
        return self.classifier(out).squeeze(-1)

    def embed(self, x, edge_index):
        layer_outputs = self._layers(x, edge_index)
        return self.jk(layer_outputs)
