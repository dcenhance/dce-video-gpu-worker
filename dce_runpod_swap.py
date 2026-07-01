#!/usr/bin/env python3
"""One-command DCE RunPod GPU video swap.

Prepares public segment inputs, submits them to the configured RunPod Queue endpoint,
and publishes rolling partial/final outputs under dcenhancements.com.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_ENDPOINT_ID = "askvbho4tgu06s"
ROOT = Path(__file__).resolve().parent
PUBLIC_BASE = "https://dcenhancements.com/archive/outputs/video-gpu/runpod_jobs"


def run(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prepare and render a DCE video face-swap through RunPod GPU Queue."
    )
    ap.add_argument("source", type=Path, help="Source face image")
    ap.add_argument("video", type=Path, help="Target video")
    ap.add_argument("--name", help="Public job slug/name. Default derives from target filename.")
    ap.add_argument("--endpoint-id", default=os.getenv("RUNPOD_VIDEO_ENDPOINT_ID") or DEFAULT_ENDPOINT_ID,
                    help=f"RunPod video endpoint id. Default: RUNPOD_VIDEO_ENDPOINT_ID or {DEFAULT_ENDPOINT_ID}")
    ap.add_argument("--mode", default="source", help="Quality mode sent to worker; default: source")
    ap.add_argument("--source-face", default="largest")
    ap.add_argument("--target-face", default="center")
    ap.add_argument("--segment-seconds", type=float, default=30.0)
    ap.add_argument("--parallel", type=int, default=1, help="Total concurrent RunPod jobs")
    ap.add_argument("--per-endpoint-parallel", type=int, default=1)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--poll-seconds", type=float, default=10.0)
    ap.add_argument("--timeout-seconds", type=float, default=3600.0)
    ap.add_argument("--max-inline-mb", type=float, default=120.0)
    ap.add_argument("--limit", type=int, help="Smoke-test only first N segments")
    ap.add_argument("--reencode-input-segments", action="store_true", help="Re-encode input segments to 30fps CRF18 instead of stream-copy splitting")
    ap.add_argument("--no-resume", action="store_true", help="Ignore already completed GPU output segments")
    ap.add_argument("--prepare-only", action="store_true", help="Only create public input segments/manifest")
    ap.add_argument("--dry-run", action="store_true", help="Print orchestration payload without submitting RunPod jobs")
    args = ap.parse_args()

    require_file(args.source, "source image")
    require_file(args.video, "target video")
    if not os.getenv("RUNPOD_API_KEY"):
        raise SystemExit("RUNPOD_API_KEY is not set in this shell/session")

    prepare_cmd = [
        sys.executable,
        str(ROOT / "prepare_runpod_inputs.py"),
        str(args.source),
        str(args.video),
        "--segment-seconds",
        f"{args.segment_seconds:g}",
    ]
    if args.name:
        prepare_cmd += ["--name", args.name]
    if args.reencode_input_segments:
        prepare_cmd.append("--reencode")

    prepared = run(prepare_cmd, capture=True)
    # prepare script prints ffmpeg commands before the final JSON; parse last JSON object.
    text = prepared.stdout or ""
    start = text.rfind("{\n")
    if start < 0:
        print(text, file=sys.stderr)
        raise SystemExit("Could not parse prepare_runpod_inputs.py output")
    info = json.loads(text[start:])
    manifest_url = info["manifest_url"]
    manifest_path = Path("/srv/apps/website/public/archive/outputs/video-gpu/runpod_jobs") / manifest_url.rstrip("/").split("/runpod_jobs/", 1)[1]

    print("\nPREPARED")
    print(json.dumps(info, indent=2))
    print(f"Input page: {info['index_url']}")

    if args.prepare_only:
        return

    orchestrate_cmd = [
        sys.executable,
        str(ROOT / "runpod_orchestrate_job.py"),
        str(manifest_path),
        "--endpoint-id",
        args.endpoint_id,
        "--mode",
        args.mode,
        "--source-face",
        args.source_face,
        "--target-face",
        args.target_face,
        "--parallel",
        str(args.parallel),
        "--per-endpoint-parallel",
        str(args.per_endpoint_parallel),
        "--retries",
        str(args.retries),
        "--poll-seconds",
        str(args.poll_seconds),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--max-inline-mb",
        str(args.max_inline_mb),
    ]
    if args.limit:
        orchestrate_cmd += ["--limit", str(args.limit)]
    if args.no_resume:
        orchestrate_cmd.append("--no-resume")
    if args.dry_run:
        orchestrate_cmd.append("--dry-run")

    print("\nRUNPOD")
    print(f"Endpoint: {args.endpoint_id}")
    print(f"Status page will be under: {PUBLIC_BASE}/{manifest_path.parent.name}/runpod_outputs.html")
    run(orchestrate_cmd)


if __name__ == "__main__":
    main()
