"""XGBoost baseline: node-level tabular features + hand-engineered graph
aggregations (degree, volume, tx_count) but no multi-hop message passing.
This is the fair comparison point for the R-GCN in train_gnn.py.
"""
import json
from pathlib import Path

import numpy as np
import xgboost as xgb

from src.data_pipeline.build_graph import build_full_graph, train_val_test_split
from src.training.metrics import evaluate

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
CKPT_DIR = Path(__file__).resolve().parents[2] / "checkpoints"


def main(dataset: str = "20K_cycle200"):
    data = build_full_graph(dataset)
    X = data.x.numpy()
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
        X[train_mask.numpy()],
        y[train_mask.numpy()],
        eval_set=[(X[val_mask.numpy()], y[val_mask.numpy()])],
        verbose=False,
    )

    test_scores = model.predict_proba(X[test_mask.numpy()])[:, 1]
    metrics = evaluate(y[test_mask.numpy()], test_scores)

    print("XGBoost baseline (tabular + hand-engineered graph features):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "baseline_xgboost.json", "w") as f:
        json.dump(metrics, f, indent=2)

    CKPT_DIR.mkdir(exist_ok=True)
    model.save_model(CKPT_DIR / "xgboost.json")

    return model, metrics


if __name__ == "__main__":
    main()
