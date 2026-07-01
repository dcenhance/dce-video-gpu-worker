#!/usr/bin/env python3
"""DCE GPU face-swap segment renderer for RunPod.

Single-segment GPU renderer used by handler.py. It intentionally avoids the CPU
ProcessPool path: one GPU worker processes one segment/job at a time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import requests

MODEL_URL = "https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx"
MODEL_DIR = Path(os.getenv("DCE_MODEL_DIR", "/models"))
INSWAPPER_PATH = MODEL_DIR / "inswapper_128.onnx"
DEFAULT_OUTPUT_FPS = 30.0


@dataclass(frozen=True)
class ModeSpec:
    key: str
    label: str
    width: int
    height: int
    crf: int
    tier: str


MODES: dict[str, ModeSpec] = {
    "source": ModeSpec("source", "Source Preserve / Original Quality", 0, 0, 18, "GOOD"),
    "preview540": ModeSpec("preview540", "Preview 540p / Balanced", 960, 540, 20, "GOOD"),
    "preview480": ModeSpec("preview480", "Preview 480p / Fast", 854, 480, 20, "GOOD_MIN"),
    "preview360": ModeSpec("preview360", "Preview 360p / Turbo", 640, 360, 23, "PREVIEW"),
}
ALIASES = {
    "quality": "source",
    "original": "source",
    "source720": "source",
    "540p": "preview540",
    "480p": "preview480",
    "fast480": "preview480",
    "360p": "preview360",
    "turbo360": "preview360",
}


def sh(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(map(str, cmd)), flush=True)
    return subprocess.run(
        cmd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def resolve_mode(name: str) -> ModeSpec:
    key = ALIASES.get(name, name)
    if key not in MODES:
        raise SystemExit(f"Unknown mode {name!r}. Valid: {', '.join(MODES)}")
    return MODES[key]


def download_url(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.replace(dest)
    return dest


def ensure_models() -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return download_url(MODEL_URL, INSWAPPER_PATH)


def providers() -> list[str]:
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    selected: list[str] = []
    if "CUDAExecutionProvider" in available:
        selected.append("CUDAExecutionProvider")
    selected.append("CPUExecutionProvider")
    print(f"ORT_PROVIDERS available={sorted(available)} selected={selected}", flush=True)
    return selected


def bbox_list(face: Any) -> list[float]:
    return [float(x) for x in face.bbox]


def face_area(face: Any) -> float:
    x1, y1, x2, y2 = bbox_list(face)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def select_face(faces: list[Any], policy: str, frame_shape: tuple[int, ...] | None = None) -> Any:
    if not faces:
        raise ValueError("No faces to select")
    if policy == "leftmost":
        return min(faces, key=lambda f: f.bbox[0])
    if policy == "rightmost":
        return max(faces, key=lambda f: f.bbox[2])
    if policy == "largest":
        return max(faces, key=face_area)
    if policy == "center":
        if frame_shape is None:
            return max(faces, key=face_area)
        h, w = frame_shape[:2]
        cx, cy = w / 2.0, h / 2.0
        return min(faces, key=lambda f: (((f.bbox[0] + f.bbox[2]) / 2.0 - cx) ** 2 + ((f.bbox[1] + f.bbox[3]) / 2.0 - cy) ** 2))
    raise ValueError(f"Unknown face policy {policy!r}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ffprobe_json(video: Path) -> dict[str, Any]:
    cp = sh([
        "ffprobe", "-v", "error",
        "-show_entries", "format=format_name,duration,size,bit_rate",
        "-show_entries", "stream=index,codec_type,codec_name,width,height,avg_frame_rate,duration,nb_frames",
        "-of", "json", str(video),
    ], capture=True)
    return json.loads(cp.stdout or "{}")


def load_models(source: Path, source_face_policy: str):
    import insightface

    prov = providers()
    app = insightface.app.FaceAnalysis(
        name="buffalo_l",
        providers=prov,
        allowed_modules=["detection", "recognition"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    src = cv2.imread(str(source))
    if src is None:
        raise RuntimeError(f"Could not read source image: {source}")
    source_faces = app.get(src) or []
    if not source_faces:
        raise RuntimeError("No face detected in source image")
    source_face = select_face(source_faces, source_face_policy, src.shape)

    ensure_models()
    swapper = insightface.model_zoo.get_model(str(INSWAPPER_PATH), providers=prov)
    return app, swapper, source_face


def probe_video(video: Path) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or DEFAULT_OUTPUT_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if frames <= 0:
        raise RuntimeError("Could not determine input frame count")
    return fps, width, height, frames


def render_segment(
    *,
    source: Path,
    video: Path,
    output: Path,
    mode_name: str = "source",
    source_face_policy: str = "largest",
    target_face_policy: str = "center",
    fps_out: float = DEFAULT_OUTPUT_FPS,
    crf_override: int | None = None,
    audio_offset_seconds: float = 0.0,
) -> dict[str, Any]:
    t_start = time.time()
    mode = resolve_mode(mode_name)
    input_fps, in_w, in_h, total_frames = probe_video(video)
    out_w = in_w if mode.width == 0 else mode.width
    out_h = in_h if mode.height == 0 else mode.height
    crf = crf_override if crf_override is not None else mode.crf
    output.parent.mkdir(parents=True, exist_ok=True)
    silent = output.with_name(output.stem + "_silent.mp4")

    analyzer, swapper, source_face = load_models(source, source_face_policy)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open target video: {video}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(silent), fourcc, fps_out, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video writer: {silent}")

    processed = 0
    swapped = 0
    first_face_frame = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] != out_w or frame.shape[0] != out_h:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            faces = analyzer.get(frame) or []
            if faces:
                target_face = select_face(faces, target_face_policy, frame.shape)
                frame = swapper.get(frame, target_face, source_face, paste_back=True)
                swapped += 1
                if first_face_frame is None:
                    first_face_frame = processed
            writer.write(frame)
            processed += 1
    finally:
        cap.release()
        writer.release()

    if processed == 0:
        raise RuntimeError("No frames processed")
    if swapped == 0:
        raise RuntimeError("No target faces detected/swapped")

    duration = processed / max(fps_out, 1e-6)
    audio_input = ["-i", str(video)]
    if audio_offset_seconds > 0.001:
        audio_input = ["-ss", f"{audio_offset_seconds:.6f}", "-i", str(video)]
    sh([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(silent),
        *audio_input,
        "-map", "0:v:0", "-map", "1:a:0?",
        "-r", f"{fps_out:g}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-t", f"{duration:.6f}",
        "-shortest", str(output),
    ])

    seconds = time.time() - t_start
    metrics = {
        "mode": mode.key,
        "mode_label": mode.label,
        "mode_tier": mode.tier,
        "width": out_w,
        "height": out_h,
        "input_fps": input_fps,
        "output_fps": fps_out,
        "input_frames": total_frames,
        "processed": processed,
        "swapped": swapped,
        "first_face_frame": first_face_frame,
        "seconds": seconds,
        "effective_fps": processed / max(seconds, 1e-6),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "ffprobe": ffprobe_json(output),
    }
    output.with_suffix(".json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="DCE GPU face-swap segment renderer")
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--mode", default="source")
    ap.add_argument("--source-face", choices=["leftmost", "rightmost", "largest", "center"], default="largest")
    ap.add_argument("--target-face", choices=["leftmost", "rightmost", "largest", "center"], default="center")
    ap.add_argument("--fps", type=float, default=DEFAULT_OUTPUT_FPS)
    ap.add_argument("--crf", type=int)
    ap.add_argument("--audio-offset-seconds", type=float, default=0.0)
    args = ap.parse_args()
    metrics = render_segment(
        source=args.source,
        video=args.video,
        output=args.output,
        mode_name=args.mode,
        source_face_policy=args.source_face,
        target_face_policy=args.target_face,
        fps_out=args.fps,
        crf_override=args.crf,
        audio_offset_seconds=args.audio_offset_seconds,
    )
    print("METRICS=" + json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
