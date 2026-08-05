"""Train TGN via a joint loss: link prediction (self-supervised pretext task
that actually drives the memory's GRU update function -- classification loss
alone doesn't backprop meaningfully through the memory) + fraud
classification (the task we care about) on the same causal embeddings.

Chronological, non-shuffled batches -- memory state must only ever depend on
past events, since that causal property is the entire point of using TGN
here (see src/models/tgn.py docstring).
"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn.models.tgn import LastNeighborLoader

from src.data_pipeline.build_graph import train_val_test_split
from src.data_pipeline.temporal_data import build_temporal_data
from src.models.tgn import TGNFraudDetector
from src.training.metrics import evaluate

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
CKPT_DIR = Path(__file__).resolve().parents[2] / "checkpoints"


def run_epoch(model, data, static_x, y, assoc, neighbor_loader, loader, device, pos_weight, class_loss_weight, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    model.memory.reset_state()
    neighbor_loader.reset_state()

    total_link_loss = total_class_loss = 0.0
    n_batches = 0

    for batch in loader:
        if train_mode:
            optimizer.zero_grad()

        src, pos_dst, t, msg, neg_dst = batch.src, batch.dst, batch.t, batch.msg, batch.neg_dst
        src, pos_dst, t, msg, neg_dst = (x.to(device) for x in (src, pos_dst, t, msg, neg_dst))

        n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = model.memory(n_id)
        z = model.gnn(z, last_update, edge_index, data.t[e_id].to(device), data.msg[e_id].to(device))

        pos_out = model.link_pred(z[assoc[src]], z[assoc[pos_dst]])
        neg_out = model.link_pred(z[assoc[src]], z[assoc[neg_dst]])
        link_loss = F.binary_cross_entropy_with_logits(
            pos_out, torch.ones_like(pos_out)
        ) + F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))

        touched = torch.cat([src, pos_dst]).unique()
        class_logits = model.classifier(z[assoc[touched]], static_x[touched])
        class_loss = F.binary_cross_entropy_with_logits(
            class_logits, y[touched], pos_weight=pos_weight
        )

        loss = link_loss + class_loss_weight * class_loss

        model.memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        if train_mode:
            loss.backward()
            optimizer.step()
            model.memory.detach()

        total_link_loss += link_loss.detach().item()
        total_class_loss += class_loss.detach().item()
        n_batches += 1

    return total_link_loss / n_batches, total_class_loss / n_batches


@torch.no_grad()
def final_scores(model, data, static_x, assoc, neighbor_loader, loader, device, num_nodes):
    """Replay the full stream once causally and keep the last classification
    score computed for every node -- i.e. "as of the end of the stream"."""
    model.eval()
    model.memory.reset_state()
    neighbor_loader.reset_state()
    scores = torch.zeros(num_nodes, device=device)

    for batch in loader:
        src, pos_dst, t, msg = batch.src.to(device), batch.dst.to(device), batch.t.to(device), batch.msg.to(device)

        touched = torch.cat([src, pos_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(touched)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = model.memory(n_id)
        z = model.gnn(z, last_update, edge_index, data.t[e_id].to(device), data.msg[e_id].to(device))
        logits = model.classifier(z[assoc[touched]], static_x[touched])
        scores[touched] = torch.sigmoid(logits)

        model.memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

    return scores


def main(
    dataset: str = "20K_cycle200",
    memory_dim: int = 100,
    time_dim: int = 100,
    batch_size: int = 200,
    neighbor_size: int = 10,
    lr: float = 1e-4,
    epochs: int = 3,
    class_loss_weight: float = 5.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data, static_x, y, fraud_step, num_nodes = build_temporal_data(dataset)
    data = data.to(device)  # keeps data.t[e_id]/data.msg[e_id] on the same device as e_id (from neighbor_loader)
    static_x, y = static_x.to(device), y.to(device)

    model = TGNFraudDetector(
        num_nodes=num_nodes, raw_msg_dim=data.msg.size(-1), static_dim=static_x.size(-1),
        memory_dim=memory_dim, time_dim=time_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    neighbor_loader = LastNeighborLoader(num_nodes, size=neighbor_size, device=device)
    assoc = torch.empty(num_nodes, dtype=torch.long, device=device)

    train_mask, val_mask, test_mask = train_val_test_split(y.cpu())
    pos_weight = (y[train_mask] == 0).sum() / (y[train_mask] == 1).sum().clamp_min(1)

    loader = TemporalDataLoader(data, batch_size=batch_size, neg_sampling_ratio=1.0)

    for epoch in range(1, epochs + 1):
        link_loss, class_loss = run_epoch(
            model, data, static_x, y, assoc, neighbor_loader, loader, device,
            pos_weight, class_loss_weight, optimizer,
        )
        print(f"epoch {epoch} | link_loss {link_loss:.4f} | class_loss {class_loss:.4f}")

    eval_loader = TemporalDataLoader(data, batch_size=batch_size)  # no neg sampling needed for scoring
    scores = final_scores(model, data, static_x, assoc, neighbor_loader, eval_loader, device, num_nodes)

    test_scores = scores[test_mask].cpu().numpy()
    test_y = y[test_mask].cpu().numpy()
    metrics = evaluate(test_y, test_scores)

    print("\nTGN:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "tgn.json", "w") as f:
        json.dump(metrics, f, indent=2)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "memory_dim": memory_dim,
            "time_dim": time_dim,
            "num_nodes": num_nodes,
            "raw_msg_dim": data.msg.size(-1),
            "static_dim": static_x.size(-1),
            "neighbor_size": neighbor_size,
        },
        CKPT_DIR / "tgn.pt",
    )

    return model, metrics


if __name__ == "__main__":
    main()
