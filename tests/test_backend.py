"""Smoke tests for VisionForge backend + trainer."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_device_returns_known_value():
    from trainer import device

    assert device.best_device() in ("mps", "cpu")


def test_manifest_loader_handles_missing():
    from trainer import data

    assert data.load_manifest("/nonexistent/manifest.json") == []
    assert data.count_boxes([{"boxes": [{"x": 1}]}, {"boxes": []}]) == 1


def test_training_loop_writes_summary(tmp_path):
    from trainer import train

    summary = train.run_training("ds_test", tmp_path, epochs=2)
    assert summary["status"] == "done"
    assert (tmp_path / "metrics.jsonl").exists()
    assert (tmp_path / "summary.json").exists()


def test_export_stub_without_weights_or_libs(tmp_path):
    from trainer import export

    out = export.export_model(tmp_path, "coreml")
    assert out["format"] == "coreml"
    assert out["status"] == "stub"
    assert out["artifact"] is None
    assert "reason" in out


def test_api_health():
    from fastapi.testclient import TestClient

    from backend.app import app

    client = TestClient(app)
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["ok"] is True
