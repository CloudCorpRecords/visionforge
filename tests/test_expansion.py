"""Expansion tests: ingest, annotations, real trainer, eval, augment, export."""
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _png_bytes(color=(200, 40, 40)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(buf, format="PNG")
    return buf.getvalue()


def test_augment_deterministic_and_versioned():
    from trainer import augment

    img = Image.new("RGB", (32, 32), (10, 200, 10))
    a = augment.apply("v2-photo", img, seed=7, index=3)
    b = augment.apply("v2-photo", img, seed=7, index=3)
    assert list(a.getdata()) == list(b.getdata())
    assert set(augment.RECIPES) >= {"v1-none", "v2-photo"}
    try:
        augment.apply("nope", img)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown recipe should raise")


def test_softmax_loss_decreases():
    from trainer import models

    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-2, 1, (10, 4)), rng.normal(2, 1, (10, 4))])
    y = np.array([0] * 10 + [1] * 10)
    clf = models.SoftmaxClassifier(4, ["a", "b"])
    history = clf.fit(X, y, epochs=30, lr=1.0)
    assert history[-1]["loss"] < history[0]["loss"]
    assert history[-1]["acc"] == 1.0


def test_eval_metrics_known_case():
    from trainer.eval import evaluate

    report = evaluate([0, 0, 1, 1], [0, 1, 1, 1], ["a", "b"])
    assert report["accuracy"] == 0.75
    assert report["confusion_matrix"] == [[1, 1], [0, 2]]
    by_class = {c["class"]: c for c in report["per_class"]}
    assert by_class["a"]["recall"] == 0.5
    assert by_class["b"]["precision"] == round(2 / 3, 4)


def test_upload_and_annotations_roundtrip():
    from fastapi.testclient import TestClient

    from backend import store
    from backend.app import app

    client = TestClient(app)
    proj = store.create_project("Expansion test", "detect")
    ds = store.create_dataset(proj["id"], {"items": []})
    res = client.post(f"/api/datasets/{ds['id']}/images",
                      files={"file": ("red.png", _png_bytes(), "image/png")})
    assert res.status_code == 200, res.text
    assert res.json()["items"] == 1
    img = client.get(f"/api/datasets/{ds['id']}/images/red.png")
    assert img.status_code == 200
    items = [{"image": "red.png", "label": "red-circle",
              "boxes": [{"x": 1, "y": 2, "w": 10, "h": 10, "label": "red-circle"}]}]
    put = client.put(f"/api/datasets/{ds['id']}/annotations", json={"items": items})
    assert put.status_code == 200
    manifest = client.get(f"/api/datasets/{ds['id']}/manifest").json()
    assert manifest["items"][0]["label"] == "red-circle"
    assert manifest["images"] == ["red.png"]


def test_real_training_eval_and_export(tmp_path):
    from fastapi.testclient import TestClient

    from backend import store
    from backend.app import app
    from trainer import train as train_mod

    client = TestClient(app)
    proj = store.create_project("Train test", "classify")
    ds = store.create_dataset(proj["id"], {"items": []})
    dpath = Path(ds["path"])
    items = []
    for i in range(8):
        label = "red-circle" if i % 2 == 0 else "blue-square"
        color = (220, 40, 40) if i % 2 == 0 else (40, 90, 220)
        name = f"t{i}.png"
        Image.new("RGB", (32, 32), color).save(dpath / name)
        items.append({"image": name, "label": label, "boxes": []})
    store.save_manifest(ds["id"], items)

    rec = store.create_run(ds["id"], str(tmp_path))
    summary = train_mod.run_training(ds["id"], tmp_path, epochs=30,
                                     backend="numpy")
    store.mark_run_done(rec["id"])
    assert summary["mode"] == "numpy"
    assert summary["eval"]["accuracy"] >= 0.9
    assert (tmp_path / "weights.npz").exists()

    res = client.get(f"/api/runs/{rec['id']}/eval")
    assert res.status_code == 200
    assert res.json()["eval"]["accuracy"] >= 0.9

    exp = client.post(f"/api/runs/{rec['id']}/export", json={"format": "npz"})
    assert exp.status_code == 200
    assert exp.json()["status"] == "ok"

    prev = client.post(f"/api/datasets/{ds['id']}/augment-preview",
                       json={"recipe": "v2-photo", "n": 2}).json()
    assert len(prev["previews"]) == 2
    assert prev["previews"][0]["data_url"].startswith("data:image/jpeg")
