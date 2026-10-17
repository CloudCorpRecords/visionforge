"""VisionForge CLI: init / import-folder / train / export / serve."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

CLASSES = ("red-circle", "blue-square")


def _demo_image(path: Path, label: str, seed: int) -> dict:
    rng = random.Random(seed)
    img = Image.new("RGB", (64, 64), (18, 22, 20))
    draw = ImageDraw.Draw(img)
    x0, y0 = rng.randint(4, 20), rng.randint(4, 20)
    x1, y1 = x0 + rng.randint(20, 32), y0 + rng.randint(20, 32)
    if label == "red-circle":
        draw.ellipse([x0, y0, x1, y1], fill=(220, 40, 40))
    else:
        draw.rectangle([x0, y0, x1, y1], fill=(40, 90, 220))
    img.save(path)
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0, "label": label}


def cmd_init(args: argparse.Namespace) -> int:
    from backend import store

    proj = store.create_project("Demo project", "detect")
    ds = store.create_dataset(proj["id"], {"items": []})
    dpath = Path(ds["path"])
    items = []
    for i in range(12):
        label = CLASSES[i % 2]
        name = f"demo_{i:02d}.png"
        box = _demo_image(dpath / name, label, seed=1000 + i)
        items.append({"image": name, "label": label, "boxes": [box]})
    store.save_manifest(ds["id"], items)
    ds = store.get_dataset(ds["id"]) or ds
    print(json.dumps({"project": proj, "dataset": ds,
                      "images": len(items)}, indent=2))
    print("Demo seeded with real images. Next: python visionforge.py serve")
    return 0


def cmd_import_folder(args: argparse.Namespace) -> int:
    import shutil

    from backend import store

    src = Path(args.dir)
    if not src.is_dir():
        print(f"not a directory: {src}")
        return 1
    images = sorted(p for p in src.iterdir()
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not images:
        print("no images found in folder")
        return 1
    projects = store.list_projects()
    proj = projects[0] if projects else store.create_project("Imported", "detect")
    ds = store.create_dataset(proj["id"], {"items": []})
    dpath = Path(ds["path"])
    items = []
    for p in images:
        shutil.copy(p, dpath / p.name)
        items.append({"image": p.name, "label": "unlabeled", "boxes": []})
    store.save_manifest(ds["id"], items)
    print(json.dumps({"project": proj["id"], "dataset": ds["id"],
                      "images": len(items)}, indent=2))
    return 0


def cmd_import_video(args: argparse.Namespace) -> int:
    from backend import store
    from trainer import video as video_mod

    if not video_mod.is_available():
        print("video support needs imageio-ffmpeg: pip install imageio imageio-ffmpeg")
        return 1
    src = Path(args.file)
    if not src.exists():
        print(f"file not found: {src}")
        return 1
    projects = store.list_projects()
    proj = projects[0] if projects else store.create_project("Video", "detect")
    ds = store.create_dataset(proj["id"], {"items": []})
    dpath = Path(ds["path"])
    names = video_mod.extract_frames(src, dpath, fps=args.fps)
    items = [{"image": n, "label": "unlabeled", "boxes": []} for n in names]
    store.save_manifest(ds["id"], items)
    print(json.dumps({"project": proj["id"], "dataset": ds["id"],
                      "frames": len(names), "fps": args.fps}, indent=2))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from backend import store
    from trainer import train as train_mod

    datasets = store.list_datasets()
    if not datasets:
        print("No datasets. Run: python visionforge.py init --demo")
        return 1
    did = args.dataset or datasets[0]["id"]
    run_dir = ROOT / "data" / "runs" / did
    rec = store.create_run(did, str(run_dir))
    print(f"Run {rec['id']} on {did} ...")
    summary = train_mod.run_training(did, run_dir, epochs=args.epochs,
                                     backend=args.backend,
                                     augment=args.augment,
                                     task=args.task, lr=args.lr,
                                     hidden=args.hidden,
                                     on_metric=lambda r: print(r, flush=True))
    store.mark_run_done(rec["id"])
    print(json.dumps({k: v for k, v in summary.items() if k != "eval"},
                     indent=2))
    if summary.get("eval"):
        print("eval:", json.dumps(summary["eval"], indent=2))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from backend import store
    from trainer import export as export_mod

    if args.run:
        rec = store.get_run(args.run)
        target = Path(rec["path"]) if rec else Path(args.run)
    else:
        runs = store.list_runs()
        target = Path(runs[0]["path"]) if runs else None
    if not target or not target.exists():
        print("No runs. Run: python visionforge.py train --epochs 20")
        return 1
    try:
        out = export_mod.export_model(target, args.format)
        print(json.dumps(out, indent=2))
    except (ValueError, FileNotFoundError) as exc:
        print(f"export failed: {exc}")
        return 1
    runs = store.list_runs()
    rid = next((r["id"] for r in runs if r["path"] == str(target)), None)
    if rid:
        store.record_export(rid, args.format, out.get("artifact"),
                            out.get("status", "ok"))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("backend.app:app", host=args.host, port=args.port,
                reload=False)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="visionforge")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="seed demo project")
    p_init.add_argument("--demo", action="store_true")
    p_init.set_defaults(func=cmd_init)

    p_imp = sub.add_parser("import-folder", help="import image folder")
    p_imp.add_argument("--dir", required=True)
    p_imp.set_defaults(func=cmd_import_folder)

    p_vid = sub.add_parser("import-video", help="import video frames")
    p_vid.add_argument("--file", required=True)
    p_vid.add_argument("--fps", type=float, default=1.0)
    p_vid.set_defaults(func=cmd_import_video)

    p_train = sub.add_parser("train", help="run local training")
    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--dataset", default=None)
    p_train.add_argument("--backend", default="auto")
    p_train.add_argument("--augment", default="v1-none")
    p_train.add_argument("--task", default="auto")
    p_train.add_argument("--lr", type=float, default=None)
    p_train.add_argument("--hidden", type=int, default=64)
    p_train.set_defaults(func=cmd_train)

    p_exp = sub.add_parser("export", help="export model")
    p_exp.add_argument("--format", default="npz")
    p_exp.add_argument("--run", default=None)
    p_exp.set_defaults(func=cmd_export)

    p_serve = sub.add_parser("serve", help="start API + UI")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
