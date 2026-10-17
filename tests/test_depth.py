"""Depth tests: detection, splits, box augment, propose, registry, detail."""
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _seed_boxed(n=12):
    from backend import store

    proj = store.create_project("Depth test", "detect")
    ds = store.create_dataset(proj["id"], {"items": []})
    dpath = Path(ds["path"])
    items = []
    for i in range(n):
        red = i % 2 == 0
        label = "red-circle" if red else "blue-square"
        img = Image.new("RGB", (64, 64), (18, 22, 20))
        draw = ImageDraw.Draw(img)
        box = {"x": 6, "y": 6, "w": 24, "h": 24, "label": label} if red else \
            {"x": 34, "y": 34, "w": 24, "h": 24, "label": label}
        color = (220, 40, 40) if red else (40, 90, 220)
        draw.rectangle([box["x"], box["y"], box["x"] + box["w"],
                        box["y"] + box["h"]], fill=color)
        name = f"d{i}.png"
        img.save(dpath / name)
        items.append({"image": name, "label": label, "boxes": [box]})
    store.save_manifest(ds["id"], items)
    return ds


def test_iou_known_values():
    from trainer import detect

    assert detect.iou([0.5, 0.5, 0.4, 0.4],
                        [0.5, 0.5, 0.4, 0.4]) == pytest.approx(1.0)
    assert detect.iou([0.1, 0.1, 0.1, 0.1],
                      [0.9, 0.9, 0.1, 0.1]) == pytest.approx(0.0)
    half = detect.iou([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.25, 0.25])
    assert half == pytest.approx(0.25, abs=1e-6)


def test_box_roundtrip():
    from trainer import detect

    box = {"x": 8, "y": 12, "w": 32, "h": 24}
    back = detect.cxcywh_to_xywh(detect.box_to_cxcywh(box))
    assert abs(back["x"] - box["x"]) <= 1
    assert abs(back["w"] - box["w"]) <= 1


def test_stratified_split_guarantees():
    from trainer import train as train_mod

    items = [{"image": f"{c}{i}.png", "label": c}
             for c in ("a", "b") for i in range(6)]
    tr, va = train_mod.stratified_split(items)
    assert len(tr) == 10 and len(va) == 2
    assert {it["label"] for it in va} == {"a", "b"}
    tr2, va2 = train_mod.stratified_split(items)
    assert [it["image"] for it in va2] == [it["image"] for it in va]


def test_apply_boxes_deterministic_and_mirrored():
    from trainer import augment

    img = Image.new("RGB", (64, 64), (0, 0, 0))
    boxes = [{"x": 10, "y": 5, "w": 20, "h": 30, "label": "t"}]
    a_img, a_boxes = augment.apply_boxes("v1-flip", img, boxes, seed=3)
    b_img, b_boxes = augment.apply_boxes("v1-flip", img, boxes, seed=3)
    assert list(a_img.getdata()) == list(b_img.getdata())
    assert a_boxes == b_boxes
    n_img, n_boxes = augment.apply_boxes("v1-none", img, boxes, seed=3)
    assert n_boxes == boxes


def test_hflip_mirror_math():
    from trainer import augment

    img = Image.new("RGB", (64, 64), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 5, 30, 35], fill=(255, 0, 0))
    boxes = [{"x": 10, "y": 5, "w": 20, "h": 30, "label": "t"}]
    for seed in range(50):
        out, new = augment.apply_boxes("v1-flip", img, boxes, seed=seed)
        if list(out.getdata()) != list(img.getdata()):
            assert new[0]["x"] == 64 - 10 - 20
            assert new[0]["y"] == 5
            return
    raise AssertionError("no flip in 50 seeds")


def test_detection_trains_and_evaluates(tmp_path):
    pytest.importorskip("torch")
    from backend import store
    from trainer import train as train_mod

    ds = _seed_boxed(12)
    rec = store.create_run(ds["id"], str(tmp_path))
    summary = train_mod.run_training(ds["id"], tmp_path, epochs=60,
                                     backend="torch", task="detect")
    store.mark_run_done(rec["id"])
    assert summary["task"] == "detect"
    assert summary["eval"]["mAP50"] >= 0.9, summary["eval"]
    assert summary["eval"]["mean_iou"] >= 0.7
    assert (tmp_path / "predictions.json").exists()
    assert (tmp_path / "model.pt").exists()
    assert (tmp_path / "run.log").exists()


def test_task_auto_routes_boxed_to_detect(tmp_path):
    pytest.importorskip("torch")
    from backend import store
    from trainer import train as train_mod

    ds = _seed_boxed(8)
    rec = store.create_run(ds["id"], str(tmp_path))
    summary = train_mod.run_training(ds["id"], tmp_path, epochs=2)
    store.mark_run_done(rec["id"])
    assert summary["task"] == "detect"


def test_propose_finds_blob(tmp_path):
    from trainer import propose as propose_mod

    img = Image.new("RGB", (64, 64), (18, 22, 20))
    draw = ImageDraw.Draw(img)
    draw.ellipse([10, 10, 40, 40], fill=(220, 40, 40))
    p = tmp_path / "blob.png"
    img.save(p)
    boxes = propose_mod.propose_boxes(p)
    assert len(boxes) >= 1
    b = boxes[0]
    assert 0 <= b["x"] < 64 and b["w"] > 10 and b["h"] > 10


def test_run_detail_exports_and_download(tmp_path):
    from fastapi.testclient import TestClient

    from backend import store
    from backend.app import app
    from trainer import train as train_mod

    client = TestClient(app)
    proj = store.create_project("Detail test", "classify")
    ds = store.create_dataset(proj["id"], {"items": []})
    dpath = Path(ds["path"])
    items = []
    for i in range(8):
        label = "red-circle" if i % 2 == 0 else "blue-square"
        color = (220, 40, 40) if i % 2 == 0 else (40, 90, 220)
        name = f"w{i}.png"
        Image.new("RGB", (32, 32), color).save(dpath / name)
        items.append({"image": name, "label": label, "boxes": []})
    store.save_manifest(ds["id"], items)
    rec = store.create_run(ds["id"], str(tmp_path))
    train_mod.run_training(ds["id"], tmp_path, epochs=5,
                           backend="numpy", task="classify")
    store.mark_run_done(rec["id"])

    detail = client.get(f"/api/runs/{rec['id']}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["config"]["task"] == "classify"
    assert any(a["name"] == "weights.npz" for a in body["artifacts"])
    assert body["log"]

    pred = client.get(f"/api/runs/{rec['id']}/predictions").json()
    assert pred["predictions"] is None

    exp = client.post(f"/api/runs/{rec['id']}/export",
                      json={"format": "npz"})
    assert exp.status_code == 200
    reg = client.get("/api/exports").json()["exports"]
    mine = next(e for e in reg if e["run_id"] == rec["id"])
    dl = client.get(f"/api/exports/{mine['id']}/download")
    assert dl.status_code == 200
    assert len(dl.content) > 100
