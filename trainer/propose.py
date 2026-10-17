"""Box proposals: connected-component assist for bootstrapping labels.

Finds saturated blobs on dark backgrounds via flood fill and returns
bounding boxes. An honest bootstrap assist — not a foundation model.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image


def propose_boxes(image_path: str | Path, min_area: int = 60) -> list[dict]:
    img = np.asarray(Image.open(image_path).convert("RGB"),
                     dtype=np.int16)
    mx, mn = img.max(axis=2), img.min(axis=2)
    mask = ((mx - mn) > 40) & (mx > 60)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    boxes = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            q, seen[y, x] = deque([(x, y)]), True
            x0, y0, x1, y1, area = x, y, x, y, 0
            while q:
                cx, cy = q.popleft()
                area += 1
                x0, y0 = min(x0, cx), min(y0, cy)
                x1, y1 = max(x1, cx), max(y1, cy)
                for nx, ny in ((cx + 1, cy), (cx - 1, cy),
                               (cx, cy + 1), (cx, cy - 1)):
                    if (0 <= nx < w and 0 <= ny < h
                            and mask[ny, nx] and not seen[ny, nx]):
                        seen[ny, nx] = True
                        q.append((nx, ny))
            if area >= min_area:
                boxes.append({"x": x0, "y": y0, "w": x1 - x0 + 1,
                              "h": y1 - y0 + 1, "label": "proposal"})
    return sorted(boxes, key=lambda b: b["w"] * b["h"], reverse=True)
