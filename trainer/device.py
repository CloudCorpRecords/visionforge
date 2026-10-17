"""Device selection for Apple Silicon: MPS > CPU, MLX optional."""
from __future__ import annotations


def best_device() -> str:
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except ImportError:
        return "cpu"


def mlx_available() -> bool:
    try:
        import mlx  # type: ignore[import-not-found]

        return True
    except ImportError:
        return False
