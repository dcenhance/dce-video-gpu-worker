#!/usr/bin/env python3
"""RunPod Serverless handler for DCE GPU face-swap segments.

Input example:
{
  "input": {
    "source_url": "https://dcenhancements.com/.../source.jpg",
    "target_url": "https://dcenhancements.com/.../segment_001.mp4",
    "segment_index": 1,
    "mode": "source",
    "source_face": "largest",
    "target_face": "center",
    "fps": 30,
    "audio_offset_seconds": 0,
    "return_base64": true
  }
}
"""
from __future__ import annotations

import base64
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import runpod

from dce_gpu_faceswap import render_segment

WORK_DIR = Path(os.getenv("DCE_WORK_DIR", "/tmp/dce-video-gpu"))
DEFAULT_MAX_INLINE_MB = float(os.getenv("DCE_MAX_INLINE_MB", "75"))


def _safe_suffix(url_or_name: str, fallback: str) -> str:
    path = urlparse(url_or_name).path if "://" in url_or_name else url_or_name
    suffix = Path(path).suffix.lower()
    return suffix if suffix else fallback


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    tmp.replace(dest)
    return dest


def _write_b64(data_b64: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(base64.b64decode(data_b64))
    return dest


def _input_file(inp: dict[str, Any], *, url_key: str, b64_key: str, dest: Path) -> Path:
    if inp.get(url_key):
        return _download(str(inp[url_key]), dest.with_suffix(_safe_suffix(str(inp[url_key]), dest.suffix)))
    if inp.get(b64_key):
        return _write_b64(str(inp[b64_key]), dest)
    raise ValueError(f"Missing {url_key} or {b64_key}")


def _upload_put(path: Path, url: str, content_type: str = "video/mp4") -> dict[str, Any]:
    with path.open("rb") as fh:
        r = requests.put(url, data=fh, headers={"Content-Type": content_type}, timeout=300)
    return {"status_code": r.status_code, "ok": 200 <= r.status_code < 300, "text": r.text[:500]}


def handler(job: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    inp = job.get("input") or {}
    work = WORK_DIR / str(job.get("id") or f"job_{int(started)}")
    work.mkdir(parents=True, exist_ok=True)
    try:
        source = _input_file(inp, url_key="source_url", b64_key="source_b64", dest=work / "source.jpg")
        target = _input_file(inp, url_key="target_url", b64_key="target_b64", dest=work / "target.mp4")
        out = work / f"segment_{int(inp.get('segment_index') or 1):04d}.mp4"
        metrics = render_segment(
            source=source,
            video=target,
            output=out,
            mode_name=str(inp.get("mode") or "source"),
            source_face_policy=str(inp.get("source_face") or "largest"),
            target_face_policy=str(inp.get("target_face") or "center"),
            fps_out=float(inp.get("fps") or 30.0),
            crf_override=(int(inp["crf"]) if inp.get("crf") is not None else None),
            audio_offset_seconds=float(inp.get("audio_offset_seconds") or 0.0),
        )
        result: dict[str, Any] = {
            "ok": True,
            "segment_index": inp.get("segment_index"),
            "seconds_total": time.time() - started,
            "metrics": metrics,
        }

        if inp.get("output_put_url"):
            result["upload"] = _upload_put(out, str(inp["output_put_url"]))
            result["output_uploaded"] = bool(result["upload"].get("ok"))
            return result

        return_base64 = bool(inp.get("return_base64", True))
        max_inline_mb = float(inp.get("max_inline_mb") or DEFAULT_MAX_INLINE_MB)
        size_mb = out.stat().st_size / (1024 * 1024)
        if return_base64 and size_mb <= max_inline_mb:
            result["output_base64"] = base64.b64encode(out.read_bytes()).decode("ascii")
            result["output_filename"] = out.name
            result["output_size_mb"] = size_mb
        else:
            result["output_base64"] = None
            result["output_size_mb"] = size_mb
            result["warning"] = f"Output is {size_mb:.1f} MB; provide output_put_url or raise max_inline_mb to return inline."
        return result
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc()[-4000:],
            "seconds_total": time.time() - started,
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
