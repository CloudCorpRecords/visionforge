"""SQLite + filesystem store. SQLite is the index, files are the truth."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DB = DATA / "visionforge.db"


def _connect() -> sqlite3.Connection:
    DATA.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    con.execute(
        """CREATE TABLE IF NOT EXISTS projects
           (id TEXT PRIMARY KEY, name TEXT, task TEXT, created_at REAL)"""
    )
    con.execute(
        """CREATE TABLE IF NOT EXISTS datasets
           (id TEXT PRIMARY KEY, project_id TEXT, version INTEGER,
            path TEXT, created_at REAL)"""
    )
    con.execute(
        """CREATE TABLE IF NOT EXISTS runs
           (id TEXT PRIMARY KEY, dataset_id TEXT, status TEXT,
            path TEXT, created_at REAL)"""
    )
    con.execute(
        """CREATE TABLE IF NOT EXISTS exports
           (id TEXT PRIMARY KEY, run_id TEXT, format TEXT,
            artifact TEXT, status TEXT, created_at REAL)"""
    )
    return con


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def list_projects() -> list[dict]:
    con = _connect()
    rows = con.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()
    con.close()
    return [dict(r) for r in rows]


def create_project(name: str, task: str = "detect") -> dict:
    pid = _new_id("proj")
    now = time.time()
    con = _connect()
    con.execute(
        "INSERT INTO projects VALUES (?,?,?,?)", (pid, name, task, now)
    )
    con.commit()
    con.close()
    (DATA / pid).mkdir(parents=True, exist_ok=True)
    return {"id": pid, "name": name, "task": task, "created_at": now}


def list_datasets() -> list[dict]:
    con = _connect()
    rows = con.execute("SELECT * FROM datasets ORDER BY created_at DESC").fetchall()
    con.close()
    return [dict(r) for r in rows]


def create_dataset(project_id: str, manifest: dict) -> dict:
    did = _new_id("ds")
    now = time.time()
    dpath = DATA / project_id / did
    dpath.mkdir(parents=True, exist_ok=True)
    (dpath / "manifest.json").write_text(json.dumps(manifest, indent=2))
    con = _connect()
    con.execute(
        "INSERT INTO datasets VALUES (?,?,?,?,?)",
        (did, project_id, 1, str(dpath), now),
    )
    con.commit()
    con.close()
    return {"id": did, "project_id": project_id, "version": 1,
            "path": str(dpath), "created_at": now}


def list_runs() -> list[dict]:
    con = _connect()
    rows = con.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
    con.close()
    return [dict(r) for r in rows]


def create_run(dataset_id: str, run_path: str) -> dict:
    rid = _new_id("run")
    now = time.time()
    con = _connect()
    con.execute(
        "INSERT INTO runs VALUES (?,?,?,?,?)",
        (rid, dataset_id, "running", run_path, now),
    )
    con.commit()
    con.close()
    return {"id": rid, "dataset_id": dataset_id, "status": "running",
            "path": run_path, "created_at": now}


def get_dataset(dataset_id: str) -> dict | None:
    con = _connect()
    row = con.execute("SELECT * FROM datasets WHERE id=?",
                      (dataset_id,)).fetchone()
    con.close()
    return dict(row) if row else None


def save_manifest(dataset_id: str, items: list[dict]) -> dict:
    rec = get_dataset(dataset_id)
    if not rec:
        raise KeyError(f"unknown dataset: {dataset_id}")
    dpath = Path(rec["path"])
    manifest = {"items": items}
    (dpath / "manifest.json").write_text(json.dumps(manifest, indent=2))
    version = int(rec.get("version", 1)) + 1
    con = _connect()
    con.execute("UPDATE datasets SET version=? WHERE id=?",
                (version, dataset_id))
    con.commit()
    con.close()
    snaps = dpath / "snapshots"
    snaps.mkdir(exist_ok=True)
    (snaps / f"manifest_v{version}.json").write_text(
        json.dumps(manifest, indent=2))
    manifest["version"] = version
    return manifest


def list_versions(dataset_id: str) -> list[dict]:
    rec = get_dataset(dataset_id)
    if not rec:
        raise KeyError(f"unknown dataset: {dataset_id}")
    snaps = Path(rec["path"]) / "snapshots"
    if not snaps.exists():
        return [{"version": int(rec.get("version", 1)), "snapshot": None}]
    out = [{"version": int(rec.get("version", 1)), "snapshot": None}]
    for p in sorted(snaps.glob("manifest_v*.json")):
        try:
            v = int(p.stem.split("_v")[1])
        except (IndexError, ValueError):
            continue
        out.append({"version": v, "snapshot": p.name})
    return sorted(out, key=lambda r: r["version"])


def get_run(run_id: str) -> dict | None:
    con = _connect()
    row = con.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    con.close()
    return dict(row) if row else None


def mark_run_done(run_id: str) -> None:
    con = _connect()
    con.execute("UPDATE runs SET status='done' WHERE id=?", (run_id,))
    con.commit()
    con.close()


def mark_run_failed(run_id: str, error: str) -> None:
    con = _connect()
    con.execute("UPDATE runs SET status='failed' WHERE id=?", (run_id,))
    con.commit()
    con.close()
    rec = get_run(run_id)
    if rec:
        with open(Path(rec["path"]) / "run.log", "a") as fh:
            fh.write(f"FAILED {error}\n")


def record_export(run_id: str, fmt: str, artifact: str | None,
                  status: str) -> dict:
    eid = _new_id("exp")
    now = time.time()
    con = _connect()
    con.execute("INSERT INTO exports VALUES (?,?,?,?,?,?)",
                (eid, run_id, fmt, artifact, status, now))
    con.commit()
    con.close()
    return {"id": eid, "run_id": run_id, "format": fmt,
            "artifact": artifact, "status": status, "created_at": now}


def list_exports(run_id: str | None = None) -> list[dict]:
    con = _connect()
    if run_id:
        rows = con.execute("SELECT * FROM exports WHERE run_id=? "
                           "ORDER BY created_at DESC", (run_id,)).fetchall()
    else:
        rows = con.execute("SELECT * FROM exports ORDER BY created_at DESC"
                           ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def get_export(export_id: str) -> dict | None:
    con = _connect()
    row = con.execute("SELECT * FROM exports WHERE id=?",
                      (export_id,)).fetchone()
    con.close()
    return dict(row) if row else None
