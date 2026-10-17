"""Export: real npz weights + real ONNX/CoreML conversion when possible.

Converters run for real when torch plus the converter lib is installed;
otherwise an honest stub manifest explains what is missing.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path


def export_model(run_dir: str | Path, fmt: str = "npz") -> dict:
    run_dir = Path(run_dir)
    fmt = fmt.lower()
    if fmt not in ("npz", "coreml", "onnx"):
        raise ValueError(f"unsupported format: {fmt}")
    if fmt == "npz":
        manifest = _export_npz(run_dir)
    elif fmt == "onnx":
        manifest = _export_onnx(run_dir)
    else:
        manifest = _export_coreml(run_dir)
    (run_dir / f"export-{fmt}.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _export_npz(run_dir: Path) -> dict:
    weights = run_dir / "weights.npz"
    if not weights.exists():
        raise FileNotFoundError("no weights.npz — train with backend=numpy first")
    out = run_dir / "model.npz"
    shutil.copy(weights, out)
    return {"format": "npz", "artifact": str(out),
            "exported_at": time.time(), "status": "ok"}


def _load_torch_artifact(pt: Path):  # type: ignore[no-untyped-def]
    from trainer import detect as detect_mod
    from trainer import torch_models

    if detect_mod.is_detector_weights(pt):
        model, classes = detect_mod.load_model(pt)
        return model, classes, model.trunk[0].in_features, ["logits", "boxes"]
    model, classes = torch_models.load_model(pt)
    return model, classes, model[0].in_features, ["logits"]


def _export_onnx(run_dir: Path) -> dict:
    pt = run_dir / "model.pt"
    try:
        import torch  # noqa: F401
        import onnx  # noqa: F401
    except ImportError as exc:
        return _stub("onnx", f"missing lib: {exc.name}")
    if not pt.exists():
        return _stub("onnx", "no model.pt — train with backend=torch first")
    try:
        import torch

        model, classes, in_dim, outputs = _load_torch_artifact(pt)
        dummy = torch.randn(1, in_dim)
        out = run_dir / "model.onnx"
        torch.onnx.export(model, dummy, out, input_names=["features"],
                          output_names=outputs,
                          dynamic_axes={"features": {0: "batch"}})
        return {"format": "onnx", "artifact": str(out), "classes": classes,
                "exported_at": time.time(), "status": "ok"}
    except Exception as exc:
        return _stub("onnx", f"conversion failed: {exc}")


def _export_coreml(run_dir: Path) -> dict:
    pt = run_dir / "model.pt"
    try:
        import torch  # noqa: F401
        import coremltools  # noqa: F401
    except ImportError as exc:
        return _stub("coreml", f"missing lib: {exc.name}")
    if not pt.exists():
        return _stub("coreml", "no model.pt — train with backend=torch first")
    try:
        import torch
        import coremltools as ct

        model, classes, in_dim, outputs = _load_torch_artifact(pt)
        dummy = torch.randn(1, in_dim)
        traced = torch.jit.trace(model, dummy)
        out = run_dir / "model.mlpackage"
        ct.convert(traced, inputs=[ct.TensorType(shape=dummy.shape)],
                   outputs=[ct.TensorType(name=n) for n in outputs]).save(out)
        return {"format": "coreml", "artifact": str(out), "classes": classes,
                "exported_at": time.time(), "status": "ok"}
    except Exception as exc:
        return _stub("coreml", f"conversion failed: {exc}")


def _stub(fmt: str, reason: str) -> dict:
    return {"format": fmt, "artifact": None, "exported_at": time.time(),
            "status": "stub", "reason": reason}
