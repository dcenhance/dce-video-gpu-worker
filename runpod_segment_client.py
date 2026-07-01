#!/usr/bin/env python3
"""DCE local client for one RunPod face-swap segment job.

This is the bridge the DCE server/orchestrator will use after it has split a
long video into 30s public segment URLs.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import time
from pathlib import Path
from typing import Any

import requests


def endpoint_url(endpoint_id: str, sync: bool) -> str:
    op = "runsync" if sync else "run"
    return f"https://api.runpod.ai/v2/{endpoint_id}/{op}"


def submit_job(payload: dict[str, Any], *, endpoint_id: str, api_key: str, sync: bool = True, timeout: int = 1800) -> dict[str, Any]:
    r = requests.post(
        endpoint_url(endpoint_id, sync),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"input": payload},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def extract_output(resp: dict[str, Any]) -> dict[str, Any]:
    # runsync shape usually has {status, output}; handler itself returns {ok,...}
    if "output" in resp and isinstance(resp["output"], dict):
        return resp["output"]
    return resp


def write_output(output: dict[str, Any], dest: Path) -> None:
    data = output.get("output_base64")
    if not data:
        raise SystemExit(f"No output_base64 in response. Response keys: {sorted(output)} warning={output.get('warning')}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(base64.b64decode(data))


def main() -> None:
    ap = argparse.ArgumentParser(description="Submit one 30s DCE face-swap segment to RunPod")
    ap.add_argument("--endpoint-id", default=os.getenv("RUNPOD_ENDPOINT_ID"), help="RunPod endpoint id, or RUNPOD_ENDPOINT_ID env")
    ap.add_argument("--api-key", default=os.getenv("RUNPOD_API_KEY"), help="RunPod API key, or RUNPOD_API_KEY env")
    ap.add_argument("--source-url", required=True)
    ap.add_argument("--target-url", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--segment-index", type=int, default=1)
    ap.add_argument("--mode", default="source")
    ap.add_argument("--source-face", default="largest")
    ap.add_argument("--target-face", default="center")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--crf", type=int)
    ap.add_argument("--audio-offset-seconds", type=float, default=0.0)
    ap.add_argument("--max-inline-mb", type=float, default=75.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.endpoint_id:
        raise SystemExit("Missing --endpoint-id or RUNPOD_ENDPOINT_ID")
    if not args.api_key:
        raise SystemExit("Missing --api-key or RUNPOD_API_KEY")

    payload: dict[str, Any] = {
        "source_url": args.source_url,
        "target_url": args.target_url,
        "segment_index": args.segment_index,
        "mode": args.mode,
        "source_face": args.source_face,
        "target_face": args.target_face,
        "fps": args.fps,
        "audio_offset_seconds": args.audio_offset_seconds,
        "return_base64": True,
        "max_inline_mb": args.max_inline_mb,
    }
    if args.crf is not None:
        payload["crf"] = args.crf

    if args.dry_run:
        safe = dict(payload)
        print(json.dumps({"endpoint_id": args.endpoint_id, "payload": safe}, indent=2))
        return

    t0 = time.time()
    resp = submit_job(payload, endpoint_id=args.endpoint_id, api_key=args.api_key, sync=True)
    output = extract_output(resp)
    if not output.get("ok"):
        raise SystemExit("RunPod job failed: " + json.dumps(output, indent=2)[:4000])
    write_output(output, args.output)
    print(json.dumps({
        "ok": True,
        "output": str(args.output),
        "bytes": args.output.stat().st_size,
        "seconds_client": time.time() - t0,
        "metrics": output.get("metrics"),
    }, indent=2))


if __name__ == "__main__":
    main()
