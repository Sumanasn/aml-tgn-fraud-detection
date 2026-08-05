"""Full-batch GNN training. 20K nodes / 118K edges fits in memory even on
CPU; switch to NeighborLoader mini-batching only if you scale the dataset up
(e.g. 20K_fanin200cycle200 combined, or a larger AMLSim run).
"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from src.data_pipeline.build_graph import build_full_graph, train_val_test_split
from src.models.gnn import FraudGNN
from src.training.metrics import evaluate

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
CKPT_DIR = Path(__file__).resolve().parents[2] / "checkpoints"


def main(
    dataset: str = "20K_cycle200",
    hidden_channels: int = 64,
    num_layers: int = 10,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 200,
    patience: int = 20,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = build_full_graph(dataset)
    train_mask, val_mask, test_mask = train_val_test_split(data.y)
    data = data.to(device)
    train_mask, val_mask, test_mask = train_mask.to(device), val_mask.to(device), test_mask.to(device)

    # Normalize features (skip degenerate columns' scale dominating the loss).
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
        loss = F.binary_cross_entropy_with_logits(
            logits[train_mask], y[train_mask], pos_weight=pos_weight
        )
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

        if epoch % 10 == 0 or epoch == 1:
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

    print("\nGraphSAGE GNN:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "gnn.json", "w") as f:
        json.dump(metrics, f, indent=2)
    torch.save(
        {
            "state_dict": best_state,
            "hidden_channels": hidden_channels,
            "num_layers": num_layers,
            "feature_mean": mean.cpu(),
            "feature_std": std.cpu(),
        },
        CKPT_DIR / "gnn.pt",
    )

    return model, metrics


if __name__ == "__main__":
    main()
