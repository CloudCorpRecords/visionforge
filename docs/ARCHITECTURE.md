# VisionForge architecture

```
visionforge/
  backend/app.py      FastAPI: health, projects, datasets, images,
                      annotations, augment-preview, runs, eval, export
  backend/store.py    SQLite + data/ filesystem store
  trainer/device.py   MPS / CPU detect, MLX optional
  trainer/data.py     Manifest loader (images + labels)
  trainer/features.py Deterministic image features (pixels + hist)
  trainer/models.py   Numpy softmax classifier (zero-dep fallback)
  trainer/torch_models.py Torch MLP on MPS/CPU (.pt save/load/predict)
  trainer/al.py       Active-learning hardest-first ranking
  trainer/video.py    Video frame extraction (imageio-ffmpeg)
  trainer/train.py    Detection/classify routing, stratified val split
  trainer/detect.py   Single-object detector (class+box heads, mAP@0.5)
  trainer/propose.py  Connected-component box proposals
  trainer/eval.py     Accuracy, per-class PR/F1, confusion matrix
  trainer/augment.py  Versioned augmentation recipes + previews
  trainer/export.py   Real npz weights + coreml/onnx converters
  visionforge.py      CLI: init / import-folder / train / export / serve
  frontend/           Vite + React local UI (Projects, Datasets, Label, Train, Models)
  data/               Projects, datasets, runs (gitignored contents)
  models/             Exported artifacts (gitignored)
```

## Data model

- Project: id, name, task (detect/segment/classify), created_at.
- Dataset: id, project_id, version, manifest.json (items: image, boxes).
- Run: id, dataset_id, config.json, metrics.jsonl, status.

## API

- `GET /api/health` → `{ok, device}`
- `GET/POST /api/projects`
- `GET/POST /api/datasets`
- `GET/POST /api/runs` (POST starts background loop, streams via metrics file)

## Trainer

`train.py` runs epochs over the manifest, computes a dummy loss curve,
writes `metrics.jsonl` + `summary.json`. Swap the loop for a real PyTorch
MPS trainer without changing the API or UI contract.

## Local-first rules

- No network calls at train time.
- Every run directory is self-contained and reproducible.
- SQLite is the index; files are the truth.
