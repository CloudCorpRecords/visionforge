"""Torch MLP trainer (MPS-ready) behind the same fit/predict/save shape.

Imports torch lazily so the module — and the whole app — works without it.
Auto device: MPS on Apple Silicon, else CPU.
"""
from __future__ import annotations
from __future__ import annotations

from pathlib import Path
from typing import Any


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def best_torch_device() -> str:
    import torch

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_mlp(X: Any, y: Any, classes: list[str], epochs: int = 20,
              lr: float = 0.05, hidden: int = 64, seed: int = 42,
              X_val: Any = None, y_val: Any = None,
              device: str | None = None) -> tuple[list[dict], Any]:
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    dev = device or best_torch_device()
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    yt = torch.tensor(y, dtype=torch.long, device=dev)
    model = nn.Sequential(
        nn.Linear(Xt.shape[1], hidden), nn.ReLU(),
        nn.Linear(hidden, len(classes)),
    ).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    has_val = X_val is not None and y_val is not None and len(X_val)
    if has_val:
        Xv = torch.tensor(X_val, dtype=torch.float32, device=dev)
        yv = torch.tensor(y_val, dtype=torch.long, device=dev)
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(Xt)
        loss = loss_fn(out, yt)
        loss.backward()
        opt.step()
        with torch.no_grad():
            acc = float((out.argmax(1) == yt).float().mean())
            row = {"epoch": epoch, "loss": round(float(loss), 4),
                   "acc": round(acc, 4), "device": dev}
            if has_val:
                outv = model(Xv)
                row["val_loss"] = round(float(loss_fn(outv, yv)), 4)
                row["val_acc"] = round(float(
                    (outv.argmax(1) == yv).float().mean()), 4)
        history.append(row)
    model.eval()
    return history, model.to("cpu")


def save_model(model: Any, classes: list[str], path: str | Path) -> None:
    import torch

    torch.save({"state": model.state_dict(),
                "classes": list(classes), "hidden": model[0].out_features,
                "in_dim": model[0].in_features}, path)


def load_model(path: str | Path) -> tuple[Any, list[str]]:
    import torch
    import torch.nn as nn

    data = torch.load(path, map_location="cpu", weights_only=False)
    classes = list(data["classes"])
    hidden = int(data.get("hidden", 64))
    model = nn.Sequential(
        nn.Linear(data["in_dim"], hidden), nn.ReLU(),
        nn.Linear(hidden, len(classes)),
    )
    model.load_state_dict(data["state"])
    model.eval()
    return model, classes


def predict(model: Any, X: Any) -> tuple[list[int], list[float]]:
    import torch

    with torch.no_grad():
        probs = torch.softmax(model(torch.tensor(X, dtype=torch.float32)), 1)
        conf, pred = probs.max(1)
    return pred.tolist(), conf.tolist()
