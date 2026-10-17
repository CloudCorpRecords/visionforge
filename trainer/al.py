"""Active learning: rank unlabeled images hardest-first.

Least-confidence sampling against a trained run's weights. Works with
numpy (.npz) and torch (.pt) backends.
"""
from __future__ import annotations
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def rank_unlabeled(dataset_dir: str | Path, items: list[dict],
                   run_dir: str | Path, top_n: int = 10) -> list[dict[str, Any]]:
    from trainer import features as feat_mod

    dataset_dir, run_dir = Path(dataset_dir), Path(run_dir)
    unlabeled = [it for it in items
                 if it.get("label") in (None, "", "unlabeled")
                 and (dataset_dir / it["image"]).exists()]
    if not unlabeled:
        return []
    X = np.stack([feat_mod.featurize(dataset_dir / it["image"])
                  for it in unlabeled])
    pred, conf = _predict_run(run_dir, X)
    ranked = sorted(zip(unlabeled, pred, conf), key=lambda t: t[2])
    return [{"image": it["image"], "predicted": p,
             "confidence": round(float(c), 4)}
            for it, p, c in ranked[:top_n]]


def _predict_run(run_dir: Path, X: np.ndarray) -> tuple[list[str], list[float]]:
    npz = run_dir / "weights.npz"
    pt = run_dir / "model.pt"
    if npz.exists():
        from trainer import models as models_mod

        clf = models_mod.SoftmaxClassifier.load(npz)
        logits = X.astype(np.float64) @ clf.W + clf.b
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        probs = e / e.sum(axis=1, keepdims=True)
        idx = probs.argmax(axis=1)
        return [clf.classes[i] for i in idx], probs.max(axis=1).tolist()
    if pt.exists():
        from trainer import detect as detect_mod
        from trainer import torch_models

        if detect_mod.is_detector_weights(pt):
            model, classes = detect_mod.load_model(pt)
            idx, conf = detect_mod.predict_proba(model, X)
        else:
            model, classes = torch_models.load_model(pt)
            idx, conf = torch_models.predict(model, X)
        return [classes[i] for i in idx], conf
    raise FileNotFoundError("run has no weights (train on labeled images first)")
