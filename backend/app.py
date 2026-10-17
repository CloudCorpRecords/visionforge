"""FastAPI backend serving the bundled UI + REST API."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend import store
from trainer import augment as augment_mod
from trainer import device as device_mod
from trainer import export as export_mod
from trainer import train as train_mod

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "frontend" / "dist"

app = FastAPI(title="VisionForge")


class ProjectIn(BaseModel):
    name: str
    task: str = "detect"


class DatasetIn(BaseModel):
    project_id: str
    manifest: dict[str, Any] = {}


class AnnotationsIn(BaseModel):
    items: list[dict[str, Any]]


class RunIn(BaseModel):
    dataset_id: str
    epochs: int = 20
    backend: str = "auto"
    augment: str = "v1-none"
    task: str = "auto"
    lr: Optional[float] = None
    hidden: int = 64


class AugmentIn(BaseModel):
    recipe: str = "v2-photo"
    n: int = 4


class ExportIn(BaseModel):
    format: str = "npz"


class ProposeIn(BaseModel):
    images: Optional[list[str]] = None
    n: int = 10


def _dataset_dir(dataset_id: str) -> Path:
    rec = store.get_dataset(dataset_id)
    if not rec:
        raise HTTPException(404, f"unknown dataset: {dataset_id}")
    return Path(rec["path"])


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "device": device_mod.best_device()}


@app.get("/api/projects")
def get_projects() -> dict:
    return {"projects": store.list_projects()}


@app.post("/api/projects")
def post_project(body: ProjectIn) -> dict:
    return store.create_project(body.name, body.task)


@app.get("/api/datasets")
def get_datasets() -> dict:
    return {"datasets": store.list_datasets()}


@app.post("/api/datasets")
def post_dataset(body: DatasetIn) -> dict:
    manifest = body.manifest or {"items": []}
    return store.create_dataset(body.project_id, manifest)


@app.get("/api/datasets/{dataset_id}/manifest")
def get_manifest(dataset_id: str) -> dict:
    dpath = _dataset_dir(dataset_id)
    mfile = dpath / "manifest.json"
    items = json.loads(mfile.read_text()).get("items", []) if mfile.exists() else []
    images = sorted(p.name for p in dpath.glob("*")
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    return {"dataset_id": dataset_id, "items": items, "images": images}


@app.post("/api/datasets/{dataset_id}/images")
async def upload_image(dataset_id: str, file: UploadFile = File(...)) -> dict:
    dpath = _dataset_dir(dataset_id)
    name = Path(file.filename or "upload.png").name
    if Path(name).suffix.lower() not in (".png", ".jpg", ".jpeg"):
        raise HTTPException(400, "only .png/.jpg images accepted")
    target = dpath / name
    target.write_bytes(await file.read())
    mfile = dpath / "manifest.json"
    items = json.loads(mfile.read_text()).get("items", []) if mfile.exists() else []
    if not any(it.get("image") == name for it in items):
        items.append({"image": name, "label": "unlabeled", "boxes": []})
    store.save_manifest(dataset_id, items)
    return {"image": name, "items": len(items)}


@app.get("/api/datasets/{dataset_id}/images/{name}")
def get_image(dataset_id: str, name: str):
    dpath = _dataset_dir(dataset_id)
    if "/" in name or name.startswith("."):
        raise HTTPException(400, "bad image name")
    target = dpath / name
    if not target.exists():
        raise HTTPException(404, "image not found")
    return FileResponse(target)


@app.put("/api/datasets/{dataset_id}/annotations")
def put_annotations(dataset_id: str, body: AnnotationsIn) -> dict:
    for it in body.items:
        if "image" not in it:
            raise HTTPException(400, "every item needs an image")
    manifest = store.save_manifest(dataset_id, body.items)
    return {"dataset_id": dataset_id, "items": len(manifest["items"])}


@app.post("/api/datasets/{dataset_id}/augment-preview")
def augment_preview(dataset_id: str, body: AugmentIn) -> dict:
    if body.recipe not in augment_mod.RECIPES:
        raise HTTPException(400, f"unknown recipe: {body.recipe}")
    dpath = _dataset_dir(dataset_id)
    images = sorted(p for p in dpath.glob("*")
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not images:
        raise HTTPException(400, "dataset has no images")
    previews = augment_mod.preview(images, body.recipe, n=min(body.n, 8))
    return {"recipe": body.recipe, "recipes": sorted(augment_mod.RECIPES),
            "previews": previews}


@app.get("/api/datasets/{dataset_id}/versions")
def get_versions(dataset_id: str) -> dict:
    try:
        return {"dataset_id": dataset_id,
                "versions": store.list_versions(dataset_id)}
    except KeyError:
        raise HTTPException(404, f"unknown dataset: {dataset_id}")


@app.get("/api/datasets/{dataset_id}/queue")
def get_queue(dataset_id: str, run_id: str = "", n: int = 10) -> dict:
    from trainer import al as al_mod

    dpath = _dataset_dir(dataset_id)
    mfile = dpath / "manifest.json"
    items = json.loads(mfile.read_text()).get("items", []) if mfile.exists() else []
    run = store.get_run(run_id) if run_id else None
    if run is None:
        runs = [r for r in store.list_runs() if r["dataset_id"] == dataset_id]
        run = runs[0] if runs else None
    if run is None:
        raise HTTPException(400, "no runs for this dataset — train first")
    try:
        queue = al_mod.rank_unlabeled(dpath, items, run["path"], top_n=n)
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc))
    return {"dataset_id": dataset_id, "run_id": run["id"], "queue": queue}


@app.get("/api/runs/compare")
def compare_runs(ids: str = "") -> dict:
    wanted = {x for x in ids.split(",") if x} if ids else set()
    rows = []
    for r in store.list_runs():
        if wanted and r["id"] not in wanted:
            continue
        cfg_file = Path(r["path"]) / "config.json"
        eval_file = Path(r["path"]) / "eval.json"
        cfg = json.loads(cfg_file.read_text()) if cfg_file.exists() else {}
        ev = json.loads(eval_file.read_text()) if eval_file.exists() else {}
        rows.append({"id": r["id"], "dataset_id": r["dataset_id"],
                     "status": r["status"], "backend": cfg.get("mode"),
                     "augment": cfg.get("augment", "v1-none"),
                     "epochs": cfg.get("epochs"),
                     "final_loss": (json.loads((Path(r["path"]) / "summary.json").read_text()).get("final_loss")
                                    if (Path(r["path"]) / "summary.json").exists() else None),
                     "accuracy": ev.get("accuracy")})
    return {"runs": rows}


@app.get("/api/runs")
def get_runs() -> dict:
    runs = store.list_runs()
    for r in runs:
        mfile = Path(r["path"]) / "metrics.jsonl"
        if mfile.exists():
            lines = mfile.read_text().strip().splitlines()
            r["metrics"] = [json.loads(x) for x in lines[-50:]]
    return {"runs": runs}


@app.post("/api/runs")
def post_run(body: RunIn) -> dict:
    if not store.get_dataset(body.dataset_id):
        raise HTTPException(404, f"unknown dataset: {body.dataset_id}")
    run_dir = ROOT / "data" / "runs" / body.dataset_id
    run_dir.mkdir(parents=True, exist_ok=True)
    rec = store.create_run(body.dataset_id, str(run_dir))

    def _job() -> None:
        try:
            train_mod.run_training(
                dataset_id=body.dataset_id,
                run_dir=run_dir,
                epochs=body.epochs,
                backend=body.backend,
                augment=body.augment,
                task=body.task,
                lr=body.lr,
                hidden=body.hidden,
            )
        except Exception as exc:
            store.mark_run_failed(rec["id"], str(exc))
            return
        store.mark_run_done(rec["id"])

    threading.Thread(target=_job, daemon=True).start()
    return rec


@app.get("/api/runs/{run_id}/eval")
def get_eval(run_id: str) -> dict:
    rec = store.get_run(run_id)
    if not rec:
        raise HTTPException(404, f"unknown run: {run_id}")
    efile = Path(rec["path"]) / "eval.json"
    if not efile.exists():
        return {"run_id": run_id, "eval": None,
                "note": "no eval yet — run may still be training or used synthetic mode"}
    return {"run_id": run_id, "eval": json.loads(efile.read_text())}


@app.post("/api/runs/{run_id}/export")
def post_export(run_id: str, body: ExportIn) -> dict:
    rec = store.get_run(run_id)
    if not rec:
        raise HTTPException(404, f"unknown run: {run_id}")
    try:
        out = export_mod.export_model(rec["path"], body.format)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc))
    store.record_export(run_id, body.format, out.get("artifact"),
                        out.get("status", "ok"))
    return out


@app.get("/api/runs/{run_id}")
def get_run_detail(run_id: str) -> dict:
    rec = store.get_run(run_id)
    if not rec:
        raise HTTPException(404, f"unknown run: {run_id}")
    rpath = Path(rec["path"])

    def _json(name: str):  # type: ignore[no-untyped-def]
        f = rpath / name
        return json.loads(f.read_text()) if f.exists() else None

    metrics = None
    mfile = rpath / "metrics.jsonl"
    if mfile.exists():
        metrics = [json.loads(x) for x in mfile.read_text().strip().splitlines()]
    log_file = rpath / "run.log"
    artifacts = [{"name": p.name, "bytes": p.stat().st_size}
                 for p in sorted(rpath.glob("*")) if p.is_file()]
    return {"run": rec, "config": _json("config.json"),
            "summary": _json("summary.json"), "eval": _json("eval.json"),
            "metrics": (metrics or [])[-100:],
            "log": log_file.read_text().splitlines()[-30:] if log_file.exists() else [],
            "artifacts": artifacts,
            "exports": store.list_exports(run_id)}


@app.get("/api/runs/{run_id}/predictions")
def get_predictions(run_id: str) -> dict:
    rec = store.get_run(run_id)
    if not rec:
        raise HTTPException(404, f"unknown run: {run_id}")
    pfile = Path(rec["path"]) / "predictions.json"
    if not pfile.exists():
        return {"run_id": run_id, "predictions": None,
                "note": "no predictions — detection runs write predictions.json"}
    return {"run_id": run_id, "predictions": json.loads(pfile.read_text())}


@app.post("/api/datasets/{dataset_id}/propose")
def post_propose(dataset_id: str, body: ProposeIn) -> dict:
    from trainer import propose as propose_mod

    dpath = _dataset_dir(dataset_id)
    if body.images:
        names = [n for n in body.images if (dpath / Path(n).name).exists()]
    else:
        mfile = dpath / "manifest.json"
        items = json.loads(mfile.read_text()).get("items", []) if mfile.exists() else []
        names = [it["image"] for it in items
                 if not it.get("boxes") and (dpath / it["image"]).exists()]
    out = {}
    for n in names[: body.n]:
        out[n] = propose_mod.propose_boxes(dpath / Path(n).name)
    return {"dataset_id": dataset_id, "proposals": out}


@app.get("/api/exports")
def get_exports(run_id: str = "") -> dict:
    return {"exports": store.list_exports(run_id or None)}


@app.get("/api/exports/{export_id}/download")
def download_export(export_id: str):
    rec = store.get_export(export_id)
    if not rec or not rec["artifact"]:
        raise HTTPException(404, "export not found")
    target = Path(rec["artifact"])
    if not target.exists():
        raise HTTPException(404, "artifact file missing")
    if target.is_dir():
        raise HTTPException(400, "mlpackage is a directory — copy from disk")
    return FileResponse(target, filename=target.name)


if STATIC.exists():
    app.mount("/assets", StaticFiles(directory=STATIC / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):  # type: ignore[no-untyped-def]
        target = STATIC / full_path
        if full_path and target.exists() and target.is_file():
            return FileResponse(target)
        return FileResponse(STATIC / "index.html")
else:

    @app.get("/")
    def index() -> dict:
        return {
            "app": "VisionForge",
            "ui": "frontend not built yet — run: cd frontend && npm install && npm run build",
            "health": "/api/health",
        }
