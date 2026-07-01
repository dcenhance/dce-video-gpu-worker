#!/usr/bin/env python3
"""Prepare public 30s input segments for a DCE RunPod GPU face-swap job."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from html import escape

PUBLIC_ROOT = Path("/srv/apps/website/public/archive/outputs/video-gpu")
PUBLIC_URL = "https://dcenhancements.com/archive/outputs/video-gpu"


def sh(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE if capture else None, stderr=subprocess.STDOUT if capture else None)


def slugify(text: str) -> str:
    keep = [c.lower() if c.isalnum() else "_" for c in text]
    out = "".join(keep).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "job"


def ffprobe_duration(path: Path) -> float:
    cp = sh(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)], capture=True)
    try:
        return float((cp.stdout or "0").strip())
    except ValueError:
        return 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare public source + 30s target segments for RunPod")
    ap.add_argument("source", type=Path)
    ap.add_argument("video", type=Path)
    ap.add_argument("--name")
    ap.add_argument("--segment-seconds", type=float, default=30.0)
    ap.add_argument("--reencode", action="store_true", help="Re-encode segments to exact-ish 30fps CRF18 instead of stream-copy splitting")
    args = ap.parse_args()

    if not args.source.exists():
        raise SystemExit(f"Missing source: {args.source}")
    if not args.video.exists():
        raise SystemExit(f"Missing video: {args.video}")

    base = slugify(args.name or f"runpod_{args.video.stem}_{int(time.time())}")
    outdir = PUBLIC_ROOT / "runpod_jobs" / base
    outdir.mkdir(parents=True, exist_ok=True)
    source_out = outdir / f"source{args.source.suffix.lower() or '.jpg'}"
    shutil.copy2(args.source, source_out)
    source_out.chmod(0o644)

    pattern = outdir / "input_seg_%03d.mp4"
    if args.reencode:
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(args.video),
            "-map", "0",
            "-r", "30",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-f", "segment", "-segment_time", f"{args.segment_seconds:g}", "-reset_timestamps", "1",
            str(pattern),
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(args.video),
            "-map", "0",
            "-c", "copy",
            "-f", "segment", "-segment_time", f"{args.segment_seconds:g}", "-reset_timestamps", "1",
            str(pattern),
        ]
    sh(cmd)

    segs = sorted(outdir.glob("input_seg_*.mp4"))
    if not segs:
        raise SystemExit("No segments created")
    records = []
    offset = 0.0
    for idx, seg in enumerate(segs, start=1):
        seg.chmod(0o644)
        dur = ffprobe_duration(seg)
        records.append({
            "index": idx,
            "input_path": str(seg),
            "input_url": f"{PUBLIC_URL}/runpod_jobs/{base}/{seg.name}",
            "duration": dur,
            "start_offset_estimate": offset,
        })
        offset += dur

    manifest = {
        "name": base,
        "source_path": str(source_out),
        "source_url": f"{PUBLIC_URL}/runpod_jobs/{base}/{source_out.name}",
        "video_path": str(args.video),
        "segment_seconds": args.segment_seconds,
        "reencoded": args.reencode,
        "segments": records,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path = outdir / "runpod_inputs.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest_path.chmod(0o644)

    rows = "".join(
        f"<tr><td>{r['index']}</td><td>{r['duration']:.2f}s</td><td><a href='{escape(r['input_url'])}'>input segment</a></td></tr>"
        for r in records
    )
    html = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>{escape(base)} RunPod Inputs</title><style>body{{background:#0b0d10;color:#e8edf2;font:14px system-ui,sans-serif}}main{{max-width:960px;margin:auto;padding:24px}}a{{color:#7cc7ff}}td,th{{padding:8px;border-bottom:1px solid #242a33}}</style></head>
<body><main><h1>{escape(base)} RunPod Inputs</h1><p>Source: <a href='{escape(manifest['source_url'])}'>source</a> · JSON: <a href='runpod_inputs.json'>runpod_inputs.json</a></p><table><tr><th>#</th><th>Duration</th><th>Input</th></tr>{rows}</table></main></body></html>"""
    html_path = outdir / "index.html"
    html_path.write_text(html, encoding="utf-8")
    html_path.chmod(0o644)

    print(json.dumps({
        "ok": True,
        "segments": len(records),
        "manifest_url": f"{PUBLIC_URL}/runpod_jobs/{base}/runpod_inputs.json",
        "index_url": f"{PUBLIC_URL}/runpod_jobs/{base}/index.html",
        "source_url": manifest["source_url"],
    }, indent=2))


if __name__ == "__main__":
    main()
