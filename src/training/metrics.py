import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    order = np.argsort(-scores)
    top_k = order[:k]
    n_positive = y_true.sum()
    if n_positive == 0:
        return 0.0
    return y_true[top_k].sum() / n_positive


def evaluate(y_true: np.ndarray, scores: np.ndarray, k_values=(50, 100, 200)) -> dict:
    out = {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
    }
    for k in k_values:
        out[f"recall@{k}"] = float(recall_at_k(y_true, scores, k))
    return out
