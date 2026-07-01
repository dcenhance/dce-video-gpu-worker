#!/usr/bin/env python3
"""Prefetch DCE video GPU worker models into the Docker image."""
from __future__ import annotations

import os
from pathlib import Path

import requests

MODEL_URL = "https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx"
MODEL_DIR = Path(os.getenv("DCE_MODEL_DIR", "/models"))
INSIGHTFACE_ROOT = Path(os.getenv("DCE_INSIGHTFACE_ROOT", str(MODEL_DIR / "insightface")))
INSWAPPER_PATH = MODEL_DIR / "inswapper_128.onnx"


def download(url: str, dest: Path, *, min_bytes: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size >= min_bytes:
        print(f"model exists: {dest} ({dest.stat().st_size} bytes)")
        return
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    print(f"downloading {url} -> {dest}", flush=True)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    tmp.replace(dest)
    size = dest.stat().st_size
    if size < min_bytes:
        raise RuntimeError(f"downloaded model too small: {dest} {size} bytes")
    print(f"downloaded {dest} ({size} bytes)")


def prefetch_insightface() -> None:
    import insightface

    # FaceAnalysis downloads buffalo_l into root/models/buffalo_l on first prepare().
    app = insightface.app.FaceAnalysis(
        name="buffalo_l",
        root=str(INSIGHTFACE_ROOT),
        providers=["CPUExecutionProvider"],
        allowed_modules=["detection", "recognition"],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))
    model_dir = INSIGHTFACE_ROOT / "models" / "buffalo_l"
    files = sorted(str(p.relative_to(INSIGHTFACE_ROOT)) for p in model_dir.glob("*.onnx"))
    if not files:
        raise RuntimeError(f"buffalo_l was not prefetched into {model_dir}")
    print("prefetched insightface files:", files)


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    download(MODEL_URL, INSWAPPER_PATH, min_bytes=100 * 1024 * 1024)
    prefetch_insightface()
    print("prefetch complete")


if __name__ == "__main__":
    main()
