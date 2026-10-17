"""Versioned augmentation recipes (PIL, deterministic per seed+index).

Recipes are versioned so every dataset transform stays reproducible and
comparable across runs.
"""
from __future__ import annotations

import base64
import io
import random
from pathlib import Path

from PIL import Image, ImageEnhance

RECIPES: dict[str, list[str]] = {
    "v1-none": [],
    "v1-flip": ["hflip"],
    "v2-photo": ["hflip", "brightness", "contrast"],
    "v3-rotate": ["rotate", "brightness"],
}


def apply(recipe: str, img: Image.Image, seed: int = 42,
          index: int = 0) -> Image.Image:
    if recipe not in RECIPES:
        raise ValueError(f"unknown recipe: {recipe}")
    rng = random.Random(seed * 100003 + index)
    out = img.copy()
    for op in RECIPES[recipe]:
        if op == "hflip" and rng.random() < 0.5:
            out = out.transpose(Image.FLIP_LEFT_RIGHT)
        elif op == "rotate":
            out = out.rotate(rng.uniform(-15, 15))
        elif op == "brightness":
            out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.8, 1.2))
        elif op == "contrast":
            out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.8, 1.2))
    return out


def apply_boxes(recipe: str, img: Image.Image, boxes: list[dict],
                seed: int = 42, index: int = 0) -> tuple[Image.Image, list[dict]]:
    """Augment image + boxes together. Rotation uses the enclosing box of
    rotated corners (documented approximation for small angles)."""
    if recipe not in RECIPES:
        raise ValueError(f"unknown recipe: {recipe}")
    rng = random.Random(seed * 100003 + index)
    out, w, h = img.copy(), img.width, img.height
    new_boxes = [dict(b) for b in boxes]
    for op in RECIPES[recipe]:
        if op == "hflip" and rng.random() < 0.5:
            out = out.transpose(Image.FLIP_LEFT_RIGHT)
            for b in new_boxes:
                b["x"] = w - b["x"] - b["w"]
        elif op == "rotate":
            angle = rng.uniform(-15, 15)
            out = out.rotate(angle)
            new_boxes = [_rotate_box(b, angle, w, h) for b in new_boxes]
        elif op == "brightness":
            out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.8, 1.2))
        elif op == "contrast":
            out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.8, 1.2))
    return out, new_boxes


def _rotate_box(box: dict, angle: float, w: int, h: int) -> dict:
    import math

    rad = math.radians(angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    cx, cy = w / 2, h / 2
    pts = [(box["x"], box["y"]),
           (box["x"] + box["w"], box["y"]),
           (box["x"], box["y"] + box["h"]),
           (box["x"] + box["w"], box["y"] + box["h"])]
    rot = []
    for px, py in pts:
        dx, dy = px - cx, py - cy
        rot.append((cx + dx * cos_a - dy * sin_a,
                    cy + dx * sin_a + dy * cos_a))
    xs = [p[0] for p in rot]
    ys = [p[1] for p in rot]
    x0, y0 = max(0, min(xs)), max(0, min(ys))
    x1, y1 = min(w, max(xs)), min(h, max(ys))
    return {"x": round(x0), "y": round(y0),
            "w": round(max(1, x1 - x0)), "h": round(max(1, y1 - y0)),
            "label": box.get("label", "")}


def preview(image_paths: list[str | Path], recipe: str = "v2-photo",
            n: int = 4, size: int = 128) -> list[dict]:
    out = []
    for i, p in enumerate(list(image_paths)[:n]):
        img = Image.open(p).convert("RGB")
        aug = apply(recipe, img, index=i).resize((size, size))
        buf = io.BytesIO()
        aug.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        out.append({"image": Path(p).name, "recipe": recipe,
                    "data_url": f"data:image/jpeg;base64,{b64}"})
    return out
