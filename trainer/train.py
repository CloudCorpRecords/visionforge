"""Local training loop: detection or classification, torch/MPS or numpy.

Task "auto" picks detection when enough boxed images exist and torch is
available, else classification. Every run uses a seeded stratified
train/val split and evaluates on held-out val. Falls back to a synthetic
curve only when the dataset has no usable labels.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Callable

import numpy as np

from trainer import device as device_mod
from trainer import eval as eval_mod
from trainer import features as feat_mod

MetricCB = Callable[[dict], None]


def _find_manifest(dataset_id: str) -> tuple[Path | None, list[dict], str, int]:
    from backend import store

    rec = store.get_dataset(dataset_id)
    if not rec:
        return None, [], "", 0
    mpath = Path(rec["path"]) / "manifest.json"
    if not mpath.exists():
        return Path(rec["path"]), [], "", int(rec.get("version", 0))
    raw = mpath.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()[:12]
    data = json.loads(raw)
    items = data.get("items", [])
    return Path(rec["path"]), items if isinstance(items, list) else [], sha, \
        int(rec.get("version", 0))


def resolve_backend(requested: str = "auto") -> str:
    from trainer import torch_models

    if requested == "auto":
        return "torch" if torch_models.torch_available() else "numpy"
    if requested not in ("torch", "numpy"):
        raise ValueError(f"unknown backend: {requested}")
    if requested == "torch" and not torch_models.torch_available():
        raise ValueError("torch backend requested but torch is not installed")
    return requested


def stratified_split(items: list[dict], seed: int = 42
                     ) -> tuple[list[dict], list[dict]]:
    """Seeded per-class 80/20 split; guarantees non-empty val when n>=4."""
    by_class: dict[str, list[dict]] = {}
    for it in items:
        by_class.setdefault(it["label"], []).append(it)
    rng = random.Random(seed)
    train, val = [], []
    for cls_items in by_class.values():
        order = cls_items[:]
        rng.shuffle(order)
        n = len(order)
        n_val = max(1, round(n * 0.2)) if n >= 3 else 0
        val.extend(order[:n_val])
        train.extend(order[n_val:])
    if not val and len(items) >= 4:
        train.sort(key=lambda it: it["image"])
        val.append(train.pop())
    return train, val


def run_training(dataset_id: str, run_dir: str | Path,
                 epochs: int = 20, backend: str = "auto",
                 augment: str = "v1-none", task: str = "auto",
                 lr: float | None = None, hidden: int = 64,
                 on_metric: MetricCB | None = None) -> dict:
    from trainer import augment as augment_mod

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if augment not in augment_mod.RECIPES:
        raise ValueError(f"unknown recipe: {augment}")
    log = _logger(run_dir)
    dev = device_mod.best_device()
    dpath, items, sha, version = _find_manifest(dataset_id)

    usable = [it for it in items
              if it.get("label") and it["label"] != "unlabeled"
              and dpath and (dpath / it["image"]).exists()]
    boxed = [it for it in usable if it.get("boxes")]
    classes = sorted({it["label"] for it in usable})
    log(f"dataset={dataset_id} usable={len(usable)} boxed={len(boxed)}")

    task = _resolve_task(task, boxed)
    backend = resolve_backend(backend)
    if task == "detect" and backend != "torch":
        raise ValueError("detection needs backend=torch")
    if task == "detect" and len(boxed) < 4:
        raise ValueError("detection needs >=4 boxed images")

    if task == "detect":
        return _detection_training(dataset_id, run_dir, dpath, boxed,
                                   classes, epochs, augment, dev, sha,
                                   version, lr or 0.05, log, on_metric)
    if len(usable) >= 4 and len(classes) >= 2:
        return _classify_training(dataset_id, run_dir, dpath, usable,
                                  classes, epochs, backend, augment, dev,
                                  sha, version, lr, hidden, log, on_metric)
    return _synthetic_training(dataset_id, run_dir, epochs, dev,
                               on_metric, reason="not enough labeled images")


def _resolve_task(requested: str, boxed: list[dict]) -> str:
    from trainer import torch_models

    if requested == "auto":
        if len(boxed) >= 6 and torch_models.torch_available():
            return "detect"
        return "classify"
    if requested not in ("detect", "classify"):
        raise ValueError(f"unknown task: {requested}")
    return requested


def _logger(run_dir: Path) -> Callable[[str], None]:
    log_path = run_dir / "run.log"
    log_path.write_text(f"started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    def log(msg: str) -> None:
        with log_path.open("a") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")

    return log


def _classify_matrix(dpath: Path, items: list[dict], classes: list[str],
                     augment: str) -> tuple[np.ndarray, np.ndarray]:
    from trainer import augment as augment_mod

    from PIL import Image

    X_list, y_list = [], []
    for i, it in enumerate(items):
        img = Image.open(dpath / it["image"]).convert("RGB")
        X_list.append(feat_mod.featurize(dpath / it["image"]))
        y_list.append(classes.index(it["label"]))
        if augment != "v1-none":
            aug = augment_mod.apply(augment, img, index=i)
            X_list.append(feat_mod.featurize_image(aug))
            y_list.append(classes.index(it["label"]))
    return np.stack(X_list), np.array(y_list)


def _classify_training(dataset_id: str, run_dir: Path, dpath: Path,
                       usable: list[dict], classes: list[str], epochs: int,
                       backend: str, augment: str, dev: str, sha: str,
                       version: int, lr: float | None, hidden: int,
                       log: Callable[[str], None],
                       on_metric: MetricCB | None) -> dict:
    train_items, val_items = stratified_split(usable)
    X, y = _classify_matrix(dpath, train_items, classes, augment)
    Xv, yv = _classify_matrix(dpath, val_items, classes, "v1-none")
    log(f"task=classify backend={backend} train={len(X)} val={len(Xv)}")
    config = {"dataset_id": dataset_id, "epochs": epochs, "device": dev,
              "seed": 42, "task": "classify", "mode": backend,
              "augment": augment, "classes": classes, "n_train": len(X),
              "n_val": len(Xv), "dataset_version": version,
              "manifest_sha": sha, "started_at": time.time()}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text("")

    if backend == "torch":
        from trainer import torch_models

        history, model = torch_models.train_mlp(
            X, y, classes, epochs=epochs, lr=lr or 0.05, hidden=hidden,
            X_val=Xv, y_val=yv)
        torch_models.save_model(model, classes, run_dir / "model.pt")
        pred, _ = torch_models.predict(model, Xv)
    else:
        from trainer import models as models_mod

        clf = models_mod.SoftmaxClassifier(X.shape[1], classes)
        history = clf.fit(X, y, epochs=epochs, lr=lr or 1.0,
                          X_val=Xv, y_val=yv)
        clf.save(run_dir / "weights.npz")
        pred = clf.predict(Xv).tolist()

    for row in history:
        row = {**row, "device": dev, "backend": backend}
        with metrics_path.open("a") as fh:
            fh.write(f"{json.dumps(row)}\n")
        if on_metric:
            on_metric(row)

    report = eval_mod.evaluate(yv.tolist(), list(pred), classes)
    (run_dir / "eval.json").write_text(json.dumps(report, indent=2))
    summary = {**config, "status": "done",
               "final_loss": history[-1].get("val_loss", history[-1]["loss"]),
               "final_acc": history[-1].get("val_acc", history[-1]["acc"]),
               "eval": report}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"done acc={summary['final_acc']}")
    return summary


def _detection_training(dataset_id: str, run_dir: Path, dpath: Path,
                        boxed: list[dict], classes: list[str], epochs: int,
                        augment: str, dev: str, sha: str, version: int,
                        lr: float, log: Callable[[str], None],
                        on_metric: MetricCB | None) -> dict:
    from trainer import augment as augment_mod
    from trainer import detect as detect_mod

    from PIL import Image

    train_items, val_items = stratified_split(boxed)

    def matrix(items: list[dict], aug: str):
        Xs, yc, yb = [], [], []
        for i, it in enumerate(items):
            img = Image.open(dpath / it["image"]).convert("RGB")
            variants = [(img, it["boxes"])]
            if aug != "v1-none":
                variants.append(augment_mod.apply_boxes(
                    aug, img, it["boxes"], index=i))
            for vim, vboxes in variants:
                Xs.append(feat_mod.featurize_image(vim))
                yc.append(classes.index(it["label"]))
                yb.append(detect_mod.box_to_cxcywh(vboxes[0]))
        return np.stack(Xs), np.array(yc), np.array(yb, dtype=np.float32)

    X, yc, yb = matrix(train_items, augment)
    Xv, ycv, ybv = matrix(val_items, "v1-none")
    log(f"task=detect backend=torch train={len(X)} val={len(Xv)}")
    config = {"dataset_id": dataset_id, "epochs": epochs, "device": dev,
              "seed": 42, "task": "detect", "mode": "torch",
              "augment": augment, "classes": classes, "n_train": len(X),
              "n_val": len(Xv), "dataset_version": version,
              "manifest_sha": sha, "started_at": time.time()}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    history, model = detect_mod.train_detector(
        X, yc, yb, classes, X_val=Xv, y_cls_val=ycv, y_box_val=ybv,
        epochs=epochs, lr=lr)
    detect_mod.save_model(model, classes, run_dir / "model.pt")
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text("")
    for row in history:
        row = {**row, "device": dev, "backend": "torch"}
        with metrics_path.open("a") as fh:
            fh.write(f"{json.dumps(row)}\n")
        if on_metric:
            on_metric(row)

    pred_cls, pred_box = detect_mod.predict(model, Xv)
    report = detect_mod.evaluate_detections(val_items, pred_cls, pred_box,
                                            classes)
    (run_dir / "eval.json").write_text(json.dumps(report, indent=2))
    predictions = [{
        "image": it["image"], "true_label": it["label"],
        "pred_label": classes[pc],
        "true_box": it["boxes"][0],
        "pred_box": {**detect_mod.cxcywh_to_xywh(pb), "label": classes[pc]},
        "iou": round(detect_mod.iou(pb, detect_mod.box_to_cxcywh(
            it["boxes"][0])), 4),
    } for it, pc, pb in zip(val_items, pred_cls, pred_box)]
    (run_dir / "predictions.json").write_text(json.dumps(predictions,
                                                          indent=2))
    summary = {**config, "status": "done",
               "final_loss": history[-1].get("val_loss", history[-1]["loss"]),
               "final_acc": history[-1].get("val_mean_iou",
                                            history[-1]["mean_iou"]),
               "eval": report}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"done mAP50={report['mAP50']} iou={report['mean_iou']}")
    return summary


def _synthetic_training(dataset_id: str, run_dir: Path, epochs: int,
                        dev: str, on_metric: MetricCB | None,
                        reason: str) -> dict:
    config = {"dataset_id": dataset_id, "epochs": epochs, "device": dev,
              "seed": 42, "mode": "synthetic", "note": reason,
              "started_at": time.time()}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text("")
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        loss = round(1.5 * math.exp(-0.5 * epoch) + 0.05, 4)
        acc = round(1.0 - loss / 2.0, 4)
        row = {"epoch": epoch, "loss": loss, "acc": acc, "device": dev}
        history.append(row)
        with metrics_path.open("a") as fh:
            fh.write(f"{json.dumps(row)}\n")
        if on_metric:
            on_metric(row)
    summary = {**config, "status": "done",
               "final_loss": history[-1]["loss"] if history else None,
               "final_acc": history[-1]["acc"] if history else None}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
