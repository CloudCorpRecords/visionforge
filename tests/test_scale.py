"""Scale tests: torch backend, video, active learning, snapshots, compare."""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _seed_dataset(n_labeled=8, n_unlabeled=4):
    from backend import store

    proj = store.create_project("Scale test", "classify")
    ds = store.create_dataset(proj["id"], {"items": []})
    dpath = Path(ds["path"])
    items = []
    for i in range(n_labeled):
        label = "red-circle" if i % 2 == 0 else "blue-square"
        color = (220, 40, 40) if i % 2 == 0 else (40, 90, 220)
        name = f"s{i}.png"
        Image.new("RGB", (32, 32), color).save(dpath / name)
        items.append({"image": name, "label": label, "boxes": []})
    for i in range(n_unlabeled):
        name = f"u{i}.png"
        Image.new("RGB", (32, 32), (120, 120, 120)).save(dpath / name)
        items.append({"image": name, "label": "unlabeled", "boxes": []})
    store.save_manifest(ds["id"], items)
    return ds


def test_torch_backend_trains_and_saves(tmp_path):
    pytest.importorskip("torch")
    from backend import store
    from trainer import train as train_mod

    ds = _seed_dataset(n_unlabeled=0)
    rec = store.create_run(ds["id"], str(tmp_path))
    summary = train_mod.run_training(ds["id"], tmp_path, epochs=10,
                                     backend="torch")
    store.mark_run_done(rec["id"])
    assert summary["mode"] == "torch"
    assert summary["eval"]["accuracy"] >= 0.9
    assert (tmp_path / "model.pt").exists()


def test_augment_doubles_training_data(tmp_path):
    from backend import store
    from trainer import train as train_mod

    ds = _seed_dataset(n_unlabeled=0)
    rec = store.create_run(ds["id"], str(tmp_path))
    summary = train_mod.run_training(ds["id"], tmp_path, epochs=5,
                                     backend="numpy", augment="v2-photo")
    store.mark_run_done(rec["id"])
    assert summary["n_train"] == 12  # 6 train items x2 with augment
    assert summary["n_val"] == 2
    assert summary["augment"] == "v2-photo"


def test_video_extract_roundtrip(tmp_path):
    imageio = pytest.importorskip("imageio.v2")
    pytest.importorskip("imageio_ffmpeg")
    from trainer import video as video_mod

    assert video_mod.is_available()
    path = tmp_path / "clip.mp4"
    writer = imageio.get_writer(path, fps=4, macro_block_size=None)
    for i in range(8):
        writer.append_data(np.full((32, 32, 3), i * 30, dtype=np.uint8))
    writer.close()
    names = video_mod.extract_frames(path, tmp_path / "frames", fps=2.0)
    assert len(names) == 4
    assert all((tmp_path / "frames" / n).exists() for n in names)


def test_active_learning_queue_hardest_first(tmp_path):
    from fastapi.testclient import TestClient

    from backend import store
    from backend.app import app
    from trainer import train as train_mod

    client = TestClient(app)
    ds = _seed_dataset()
    rec = store.create_run(ds["id"], str(tmp_path))
    train_mod.run_training(ds["id"], tmp_path, epochs=20, backend="numpy")
    store.mark_run_done(rec["id"])
    res = client.get(f"/api/datasets/{ds['id']}/queue?n=4")
    assert res.status_code == 200, res.text
    queue = res.json()["queue"]
    assert len(queue) == 4
    assert all(q["image"].startswith("u") for q in queue)
    confs = [q["confidence"] for q in queue]
    assert confs == sorted(confs)


def test_snapshots_and_compare(tmp_path):
    from fastapi.testclient import TestClient

    from backend import store
    from backend.app import app
    from trainer import train as train_mod

    client = TestClient(app)
    ds = _seed_dataset(n_unlabeled=0)
    v1 = store.get_dataset(ds["id"])["version"]
    dpath = Path(ds["path"])
    import json as _json

    full = _json.loads((dpath / "manifest.json").read_text())["items"]
    store.save_manifest(ds["id"], full[:1])
    v2 = store.get_dataset(ds["id"])["version"]
    assert v2 == v1 + 1
    res = client.get(f"/api/datasets/{ds['id']}/versions")
    assert res.status_code == 200
    assert len(res.json()["versions"]) >= 2
    store.save_manifest(ds["id"], full)

    rec = store.create_run(ds["id"], str(tmp_path))
    train_mod.run_training(ds["id"], tmp_path, epochs=3, backend="numpy")
    store.mark_run_done(rec["id"])
    cmp = client.get("/api/runs/compare").json()
    assert any(r["id"] == rec["id"] for r in cmp["runs"])
    row = next(r for r in cmp["runs"] if r["id"] == rec["id"])
    assert row["backend"] == "numpy"


def test_onnx_export_real_when_torch(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    from backend import store
    from trainer import export as export_mod
    from trainer import train as train_mod

    ds = _seed_dataset(n_unlabeled=0)
    rec = store.create_run(ds["id"], str(tmp_path))
    train_mod.run_training(ds["id"], tmp_path, epochs=5, backend="torch")
    store.mark_run_done(rec["id"])
    out = export_mod.export_model(tmp_path, "onnx")
    assert out["status"] == "ok", out
    assert Path(out["artifact"]).exists()
