#!/usr/bin/env python3
"""DCE RunPod orchestration for a prepared long-video job.

Consumes prepare_runpod_inputs.py manifest, submits each input segment to RunPod,
stores finished GPU outputs publicly, refreshes a partial MP4 after every segment,
and concatenates the final output when done.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import time
from pathlib import Path
from html import escape
from typing import Any
from urllib.parse import urlparse

from runpod_segment_client import submit_job, extract_output

PUBLIC_ROOT = Path("/srv/apps/website/public/archive/outputs/video-gpu")
PUBLIC_URL = "https://dcenhancements.com/archive/outputs/video-gpu"


def sh(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE if capture else None, stderr=subprocess.STDOUT if capture else None)


def public_url(path: Path) -> str:
    rel = path.relative_to(PUBLIC_ROOT)
    return f"{PUBLIC_URL}/{str(rel).replace(os.sep, '/')}"


def concat_mp4s(inputs: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if len(inputs) == 1:
        output.write_bytes(inputs[0].read_bytes())
        output.chmod(0o644)
        return
    concat = output.with_suffix(".concat.txt")
    concat.write_text("".join(f"file '{p}'\n" for p in inputs), encoding="utf-8")
    sh(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", str(output)])
    output.chmod(0o644)


def write_status(job_dir: Path, name: str, records: list[dict[str, Any]], total: int, partial: Path | None = None, final: Path | None = None) -> dict[str, str]:
    status = {
        "name": name,
        "completed_segments": len(records),
        "total_segments": total,
        "complete": len(records) >= total,
        "partial_url": public_url(partial) if partial and partial.exists() else None,
        "final_url": public_url(final) if final and final.exists() else None,
        "segments": records,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    js = job_dir / "runpod_outputs.json"
    js.write_text(json.dumps(status, indent=2), encoding="utf-8")
    js.chmod(0o644)
    rows = []
    for r in records:
        rows.append(
            f"<tr><td>{r['index']}</td><td><a href='{escape(r['output_url'])}'>output</a></td>"
            f"<td>{float(r.get('seconds_total') or 0):.1f}s</td><td>{r.get('swapped')}/{r.get('processed')}</td></tr>"
        )
    partial_link = f" · <a href='{escape(status['partial_url'])}'>partial</a>" if status.get("partial_url") else ""
    final_link = f" · <a href='{escape(status['final_url'])}'>final</a>" if status.get("final_url") else ""
    html = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>{escape(name)} RunPod Outputs</title><style>body{{background:#0b0d10;color:#e8edf2;font:14px system-ui,sans-serif}}main{{max-width:960px;margin:auto;padding:24px}}a{{color:#7cc7ff}}td,th{{padding:8px;border-bottom:1px solid #242a33}}</style></head>
<body><main><h1>{escape(name)} RunPod Outputs</h1><p>{len(records)}/{total} fertig{partial_link}{final_link} · <a href='runpod_outputs.json'>JSON</a></p><table><tr><th>#</th><th>Output</th><th>GPU seconds</th><th>Swapped</th></tr>{''.join(rows)}</table></main></body></html>"""
    hp = job_dir / "runpod_outputs.html"
    hp.write_text(html, encoding="utf-8")
    hp.chmod(0o644)
    return {"status_url": public_url(hp), "status_json_url": public_url(js)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run prepared DCE segments on RunPod and publish outputs")
    ap.add_argument("manifest", type=Path, help="runpod_inputs.json from prepare_runpod_inputs.py")
    ap.add_argument("--endpoint-id", default=os.getenv("RUNPOD_ENDPOINT_ID"))
    ap.add_argument("--api-key", default=os.getenv("RUNPOD_API_KEY"))
    ap.add_argument("--mode", default="source")
    ap.add_argument("--source-face", default="largest")
    ap.add_argument("--target-face", default="center")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--max-inline-mb", type=float, default=75.0)
    ap.add_argument("--limit", type=int, help="Submit only first N segments for smoke testing")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    name = data["name"]
    job_dir = args.manifest.parent
    segments = data["segments"][: args.limit or None]
    total = len(segments)
    print(json.dumps({"name": name, "segments_to_submit": total, "source_url": data["source_url"]}, indent=2))

    if args.dry_run:
        example = {
            "source_url": data["source_url"],
            "target_url": segments[0]["input_url"] if segments else None,
            "mode": args.mode,
            "source_face": args.source_face,
            "target_face": args.target_face,
            "fps": args.fps,
        }
        print("DRY_RUN_PAYLOAD=" + json.dumps(example, indent=2))
        return
    if not args.endpoint_id:
        raise SystemExit("Missing --endpoint-id or RUNPOD_ENDPOINT_ID")
    if not args.api_key:
        raise SystemExit("Missing --api-key or RUNPOD_API_KEY")

    out_paths: list[Path] = []
    records: list[dict[str, Any]] = []
    for seg in segments:
        idx = int(seg["index"])
        out = job_dir / f"gpu_out_seg_{idx:03d}.mp4"
        metrics_path = job_dir / f"gpu_out_seg_{idx:03d}.json"
        if out.exists() and metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            print(f"skip existing segment {idx}: {out}")
        else:
            payload = {
                "source_url": data["source_url"],
                "target_url": seg["input_url"],
                "segment_index": idx,
                "mode": args.mode,
                "source_face": args.source_face,
                "target_face": args.target_face,
                "fps": args.fps,
                "audio_offset_seconds": 0,
                "return_base64": True,
                "max_inline_mb": args.max_inline_mb,
            }
            resp = submit_job(payload, endpoint_id=args.endpoint_id, api_key=args.api_key, sync=True)
            output = extract_output(resp)
            if not output.get("ok"):
                raise SystemExit(f"RunPod segment {idx} failed: " + json.dumps(output, indent=2)[:4000])
            out.write_bytes(base64.b64decode(output["output_base64"]))
            out.chmod(0o644)
            metrics = output.get("metrics") or {}
            metrics["runpod_seconds_total"] = output.get("seconds_total")
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            metrics_path.chmod(0o644)
        out_paths.append(out)
        rec = {
            "index": idx,
            "output_path": str(out),
            "output_url": public_url(out),
            "metrics_url": public_url(metrics_path),
            "processed": metrics.get("processed"),
            "swapped": metrics.get("swapped"),
            "seconds_total": metrics.get("runpod_seconds_total") or metrics.get("seconds"),
        }
        records.append(rec)
        partial = job_dir / "gpu_partial.mp4"
        concat_mp4s(out_paths, partial)
        status = write_status(job_dir, name, records, total, partial=partial)
        print("STATUS=" + json.dumps(status | {"partial_url": public_url(partial)}, indent=2))

    final = job_dir / "gpu_final.mp4"
    concat_mp4s(out_paths, final)
    status = write_status(job_dir, name, records, total, partial=job_dir / "gpu_partial.mp4", final=final)
    print("DONE=" + json.dumps(status | {"final_url": public_url(final)}, indent=2))


if __name__ == "__main__":
    main()
