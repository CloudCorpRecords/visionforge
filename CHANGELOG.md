# Changelog

All notable changes to VisionForge. Format follows Keep a Changelog.

## [0.1.0] - 2026-10-17

First public release: a complete local-first vision training vertical slice.

### Added
- Projects, datasets, and runs with a SQLite + filesystem store.
- Image upload, folder import, and video frame extraction (bundled ffmpeg).
- Box labeling canvas: drag-to-draw, keyboard-first (n/p/s/d), box proposals.
- Classification trainer (numpy fallback, torch MLP on MPS) with real metrics.
- Single-object detection trainer (class + box heads) with mAP@0.5, mean IoU,
  and per-image val predictions.
- Seeded stratified train/val splits on every run; versioned augmentation
  recipes, including training-time augmentation and box-aware transforms.
- Eval: accuracy, per-class precision/recall/F1, confusion matrix.
- Active-learning hardest-first labeling queue.
- Dataset versioning with manifest snapshots; run comparison table.
- Run detail cards: config, artifacts, logs, predictions.
- Export registry with downloads: real `.npz`, `.onnx`, and `.mlpackage`.
- Local web UI (React + Vite) served by the same backend; full CLI.
- 25-test pytest suite covering trainers, metrics, API, and export.

### Engineering notes
- Zero-dep core: the app runs on numpy/PIL; torch, onnx, coremltools, and
  imageio-ffmpeg unlock the accelerated paths with honest fallbacks.
- Every run directory is self-contained: config, metrics, eval, weights, log.
