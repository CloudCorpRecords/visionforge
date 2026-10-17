"""Video frame extraction via imageio-ffmpeg (bundled ffmpeg binary)."""
from __future__ import annotations

from pathlib import Path


def is_available() -> bool:
    try:
        import imageio_ffmpeg  # noqa: F401
        import imageio.v2 as imageio  # noqa: F401

        imageio_ffmpeg.get_ffmpeg_exe()
        return True
    except Exception:
        return False


def extract_frames(video_path: str | Path, out_dir: str | Path,
                   fps: float = 1.0) -> list[str]:
    import imageio.v2 as imageio

    video_path, out_dir = Path(video_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reader = imageio.get_reader(video_path, "ffmpeg")
    meta = reader.get_meta_data()
    src_fps = float(meta.get("fps", 24)) or 24.0
    every = max(1, round(src_fps / fps))
    names: list[str] = []
    for i, frame in enumerate(reader):
        if i % every != 0:
            continue
        name = f"{video_path.stem}_f{i:04d}.png"
        imageio.imwrite(out_dir / name, frame)
        names.append(name)
    reader.close()
    return names
