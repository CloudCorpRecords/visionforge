"""Single-object detection: torch MLP with class + box heads on MPS/CPU.

Scope is honest single-object detection v1: one box per image, class plus
normalized cx,cy,w,h regression. Multi-object YOLO-style heads plug in
behind the same train/predict/evaluate shape.
"""
from __future__ import annotations
from __future__ import annotations

from pathlib import Path
from typing import Any

IMG = 64.0


def iou(a: list[float], b: list[float]) -> float:
    """IoU of two cx,cy,w,h boxes in normalized coords."""
    ax0, ay0 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax1, ay1 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx1, by1 = b[0] + b[2] / 2, b[1] + b[3] / 2
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def box_to_cxcywh(box: dict) -> list[float]:
    x, y, w, h = box["x"] / IMG, box["y"] / IMG, box["w"] / IMG, box["h"] / IMG
    return [x + w / 2, y + h / 2, w, h]


def cxcywh_to_xywh(b: list[float]) -> dict:
    cx, cy, w, h = (max(0.0, min(1.0, v)) for v in b)
    return {"x": round((cx - w / 2) * IMG), "y": round((cy - h / 2) * IMG),
            "w": round(w * IMG), "h": round(h * IMG)}


def build_model(in_dim: int, n_classes: int) -> Any:
    """Detector module. Built inside a function so torch stays lazily imported."""
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self, d: int, c: int):
            super().__init__()
            self.trunk = nn.Sequential(nn.Linear(d, 128), nn.ReLU(),
                                       nn.Linear(128, 64), nn.ReLU())
            self.cls = nn.Linear(64, c)
            # Linear box head: sigmoid outputs saturate and stall L1
            # regression, so we clamp to [0,1] at predict time instead.
            self.box = nn.Linear(64, 4)

        def forward(self, x):  # type: ignore[no-untyped-def]
            f = self.trunk(x)
            return self.cls(f), self.box(f)

    return Net(in_dim, n_classes)


def train_detector(X: Any, y_cls: Any, y_box: Any, classes: list[str],
                   X_val: Any = None, y_cls_val: Any = None,
                   y_box_val: Any = None, epochs: int = 40, lr: float = 0.05,
                   box_weight: float = 1.0, seed: int = 42,
                   device: str | None = None) -> tuple[list[dict], Any]:
    import torch
    import torch.nn as nn

    from trainer import torch_models

    torch.manual_seed(seed)
    dev = device or torch_models.best_torch_device()
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    yt = torch.tensor(y_cls, dtype=torch.long, device=dev)
    bt = torch.tensor(y_box, dtype=torch.float32, device=dev)
    model = build_model(Xt.shape[1], len(classes)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce, l1 = nn.CrossEntropyLoss(), nn.L1Loss()
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits, boxes = model(Xt)
        loss = ce(logits, yt) + box_weight * l1(boxes, bt)
        loss.backward()
        opt.step()
        with torch.no_grad():
            acc = float((logits.argmax(1) == yt).float().mean())
            ious = [iou(b.tolist(), t.tolist())
                     for b, t in zip(boxes.clamp(0, 1), bt)]
            row: dict[str, Any] = {
                "epoch": epoch, "loss": round(float(loss), 4),
                "acc": round(acc, 4),
                "mean_iou": round(sum(ious) / len(ious), 4),
                "device": dev}
            if X_val is not None and len(X_val):
                Xv = torch.tensor(X_val, dtype=torch.float32, device=dev)
                yv = torch.tensor(y_cls_val, dtype=torch.long, device=dev)
                bv = torch.tensor(y_box_val, dtype=torch.float32, device=dev)
                logits_v, boxes_v = model(Xv)
                vloss = ce(logits_v, yv) + box_weight * l1(boxes_v, bv)
                vacc = float((logits_v.argmax(1) == yv).float().mean())
                vious = [iou(b.tolist(), t.tolist())
                         for b, t in zip(boxes_v.clamp(0, 1), bv)]
                row["val_loss"] = round(float(vloss), 4)
                row["val_acc"] = round(vacc, 4)
                row["val_mean_iou"] = round(sum(vious) / len(vious), 4)
            history.append(row)
    model.eval()
    return history, model.to("cpu")


def predict(model: Any, X: Any) -> tuple[list[int], list[list[float]]]:
    import torch

    with torch.no_grad():
        logits, boxes = model(torch.tensor(X, dtype=torch.float32))
    return logits.argmax(1).tolist(), boxes.clamp(0, 1).tolist()


def predict_proba(model: Any, X: Any) -> tuple[list[int], list[float]]:
    import torch

    with torch.no_grad():
        logits, _ = model(torch.tensor(X, dtype=torch.float32))
        probs = torch.softmax(logits, 1)
        conf, pred = probs.max(1)
    return pred.tolist(), conf.tolist()


def load_model(path: str | Path) -> tuple[Any, list[str]]:
    import torch

    data = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(int(data["in_dim"]), len(data["classes"]))
    model.load_state_dict(data["state"])
    model.eval()
    return model, [str(c) for c in data["classes"]]


def is_detector_weights(path: str | Path) -> bool:
    import torch

    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return isinstance(data, dict) and data.get("kind") == "detector"


def save_model(model: Any, classes: list[str], path: str | Path) -> None:
    import torch

    torch.save({"kind": "detector", "state": model.state_dict(),
                "classes": list(classes),
                "in_dim": model.trunk[0].in_features}, path)


def evaluate_detections(items: list[dict], pred_cls: list[int],
                        pred_box: list[list[float]],
                        classes: list[str]) -> dict:
    """mAP@0.5 for single-object images + mean IoU + class accuracy."""
    ious: list[float] = []
    hits: list[int] = []
    per: dict[str, dict[str, int]] = {}
    correct_cls = 0
    for it, pc, pb in zip(items, pred_cls, pred_box):
        true_box = box_to_cxcywh(it["boxes"][0])
        v = iou(pb, true_box)
        ious.append(v)
        tc = classes.index(it["label"])
        if pc == tc:
            correct_cls += 1
        hit = 1 if (pc == tc and v >= 0.5) else 0
        hits.append(hit)
        stat = per.setdefault(it["label"], {"hit": 0, "n": 0})
        stat["hit"] += hit
        stat["n"] += 1
    ap = [s["hit"] / s["n"] for s in per.values()]
    return {"mAP50": round(sum(ap) / len(ap), 4) if ap else 0.0,
            "mean_iou": round(sum(ious) / len(ious), 4) if ious else 0.0,
            "accuracy": round(correct_cls / len(items), 4) if items else 0.0,
            "n": len(items), "classes": list(classes),
            "per_class_ap50": {k: round(v["hit"] / v["n"], 4)
                               for k, v in per.items()}}
