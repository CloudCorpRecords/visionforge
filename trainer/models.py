"""Numpy softmax regression: a real trainer with no heavy deps.

Full-batch gradient descent, deterministic given seed. The torch/MPS
trainer plugs in behind the same fit/predict/save/load shape.
"""
from __future__ import annotations
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class SoftmaxClassifier:
    def __init__(self, n_features: int, classes: list[str], seed: int = 42):
        self.classes = list(classes)
        rng = np.random.default_rng(seed)
        self.W = (rng.standard_normal((n_features, len(classes)))
                  .astype(np.float64) * 0.01)
        self.b = np.zeros(len(classes), dtype=np.float64)

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 20,
            lr: float = 1.0, X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None) -> list[dict[str, Any]]:
        X = X.astype(np.float64)
        n = len(X)
        history: list[dict[str, Any]] = []
        for epoch in range(1, epochs + 1):
            probs = self._softmax(X @ self.W + self.b)
            loss = float(-np.log(probs[np.arange(n), y] + 1e-12).mean())
            acc = float((probs.argmax(axis=1) == y).mean())
            row: dict[str, Any] = {"epoch": epoch, "loss": round(loss, 4),
                                   "acc": round(acc, 4)}
            if X_val is not None and y_val is not None and len(X_val):
                Xv = X_val.astype(np.float64)
                pv = self._softmax(Xv @ self.W + self.b)
                nv = len(Xv)
                row["val_loss"] = round(float(
                    -np.log(pv[np.arange(nv), y_val] + 1e-12).mean()), 4)
                row["val_acc"] = round(float(
                    (pv.argmax(axis=1) == y_val).mean()), 4)
            history.append(row)
            grad = probs
            grad[np.arange(n), y] -= 1.0
            grad /= n
            self.W -= lr * (X.T @ grad)
            self.b -= lr * grad.sum(axis=0)
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (X.astype(np.float64) @ self.W + self.b).argmax(axis=1)

    def save(self, path: str | Path) -> None:
        np.savez(path, W=self.W, b=self.b,
                 classes=np.array(self.classes, dtype=str))

    @classmethod
    def load(cls, path: str | Path) -> "SoftmaxClassifier":
        data = np.load(path, allow_pickle=True)
        classes = [str(c) for c in data["classes"].tolist()]
        clf = cls(data["W"].shape[0], classes)
        clf.W = data["W"]
        clf.b = data["b"]
        return clf
