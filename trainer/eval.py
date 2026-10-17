"""Eval metrics: accuracy, per-class PR/F1, confusion matrix."""
from __future__ import annotations

from typing import Any


def evaluate(y_true: list[int], y_pred: list[int],
             classes: list[str]) -> dict[str, Any]:
    n = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    matrix = [[0] * len(classes) for _ in classes]
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
    per_class = []
    for i, name in enumerate(classes):
        tp = matrix[i][i]
        fp = sum(matrix[r][i] for r in range(len(classes))) - tp
        fn = sum(matrix[i]) - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class.append({"class": name, "precision": round(prec, 4),
                          "recall": round(rec, 4), "f1": round(f1, 4),
                          "support": sum(matrix[i])})
    return {"accuracy": round(correct / n, 4) if n else 0.0,
            "n": n, "classes": list(classes),
            "confusion_matrix": matrix, "per_class": per_class}
