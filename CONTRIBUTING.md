# Contributing to VisionForge

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python visionforge.py init --demo
python visionforge.py serve
```

Frontend:

```bash
cd frontend && npm install && npm run build
```

## Tests

```bash
.venv/bin/python -m pytest -q
```

Add or update the focused test in `tests/` for every behavior change.
Torch/video/export tests skip honestly when optional deps are missing.

## Conventions

- Python 3.9 compatible: no `X | Y` unions in evaluated annotations
  (pydantic models), lazy imports for torch/onnx/coremltools/imageio.
- Local-first: no network calls at train time; SQLite is the index,
  files are the truth; every run directory stays self-contained.
- Small, boring diffs. Match the module's existing shape before inventing.
