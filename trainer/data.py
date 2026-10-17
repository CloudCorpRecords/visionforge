"""Manifest loader: data/<project>/<dataset>/manifest.json."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_manifest(manifest_path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return []
    data = json.loads(manifest_path.read_text())
    items = data.get("items", [])
    return items if isinstance(items, list) else []


def count_boxes(items: list[dict]) -> int:
    total = 0
    for it in items:
        boxes = it.get("boxes", [])
        if isinstance(boxes, list):
            total += len(boxes)
    return total
