"""Deterministic image features: downsampled pixels + color histogram.

No heavy deps. A torch/MPS featurizer can replace this module without
changing the trainer, eval, or API contracts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 8
HIST_BINS = 8


def load_image(path: str | Path, size: int = 64) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img.resize((size, size))


def featurize(path: str | Path) -> np.ndarray:
    return featurize_image(load_image(path))


def featurize_image(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.resize((SIZE, SIZE)), dtype=np.float32) / 255.0
    pixels = arr.reshape(-1)
    hist_parts = []
    for ch in range(3):
        hist, _ = np.histogram(arr[:, :, ch], bins=HIST_BINS, range=(0.0, 1.0))
        hist_parts.append(hist.astype(np.float32) / (SIZE * SIZE))
    hist = np.concatenate(hist_parts)
    return np.concatenate([pixels, hist]).astype(np.float32)


DIM = SIZE * SIZE * 3 + HIST_BINS * 3
