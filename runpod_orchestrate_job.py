#!/usr/bin/env python3
"""DCE RunPod queue orchestration for prepared long-video jobs.

Consumes prepare_runpod_inputs.py manifest, submits input segments to one or more
RunPod Queue endpoints, polls status, publishes each finished segment immediately,
refreshes a rolling partial MP4, and concatenates the final output when done.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from html import escape
from typing import Any

from runpod_segment_client import submit_job_async, get_job_status, extract_output

PUBLIC_ROOT = Path("/srv/apps/website/public/archive/outputs/video-gpu")
PUBLIC_URL = "https://dcenhancements.com/archive/outputs/video-gpu"
TERMINAL_OK = {"COMPLETED"}
TERMINAL_BAD = {"FAILED", "CANCELLED", "TIMED_OUT"}
ACTIVE_STATUSES = {"IN_QUEUE", "IN_PROGRESS", "RUNNING", "RETRYING"}


@dataclass
class EndpointState:
    endpoint_id: str
    active: int = 0
    submitted: int = 0
    completed: int = 0
    failed: int = 0


@dataclass
class InFlightJob:
    runpod_id: str
    endpoint_id: str
    segment: dict[str, Any]
    attempt: int
    submitted_at: float = field(default_factory=time.time)
    last_status: str = "SUBMITTED"


def sh(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE if capture else None, stderr=subprocess.STDOUT if capture else None)


def public_url(path: Path) -> str:
    rel = path.relative_to(PUBLIC_ROOT)
    return f"{PUBLIC_URL}/{str(rel).replace(os.sep, '/')}"


def concat_mp4s(inputs: list[Path], output: Path) -> None:
    ordered = [p for p in inputs if p.exists()]
    if not ordered:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    if len(ordered) == 1:
        output.write_bytes(ordered[0].read_bytes())
        output.chmod(0o644)
        return
    concat = output.with_suffix(".concat.txt")
    concat.write_text("".join(f"file '{p}'\n" for p in ordered), encoding="utf-8")
    sh(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", str(output)])
    output.chmod(0o644)


def record_for_output(job_dir: Path, idx: int, metrics: dict[str, Any]) -> dict[str, Any]:
    out = job_dir / f"gpu_out_seg_{idx:03d}.mp4"
    metrics_path = job_dir / f"gpu_out_seg_{idx:03d}.json"
    return {
        "index": idx,
        "output_path": str(out),
        "output_url": public_url(out),
        "metrics_url": public_url(metrics_path),
        "processed": metrics.get("processed"),
        "swapped": metrics.get("swapped"),
        "seconds_total": metrics.get("runpod_seconds_total") or metrics.get("seconds"),
        "endpoint_id": metrics.get("endpoint_id"),
        "runpod_id": metrics.get("runpod_id"),
        "attempt": metrics.get("attempt"),
    }


def write_status(
    job_dir: Path,
    name: str,
    records_by_idx: dict[int, dict[str, Any]],
    total: int,
    *,
    partial: Path | None = None,
    final: Path | None = None,
    inflight: dict[int, InFlightJob] | None = None,
    failed: dict[int, str] | None = None,
) -> dict[str, str | None]:
    records = [records_by_idx[i] for i in sorted(records_by_idx)]
    inflight_rows = []
    if inflight:
        for idx, job in sorted(inflight.items()):
            inflight_rows.append({
                "index": idx,
                "runpod_id": job.runpod_id,
                "endpoint_id": job.endpoint_id,
                "attempt": job.attempt,
                "status": job.last_status,
                "age_seconds": time.time() - job.submitted_at,
            })
    status = {
        "name": name,
        "completed_segments": len(records),
        "total_segments": total,
        "complete": len(records) >= total and not failed,
        "partial_url": public_url(partial) if partial and partial.exists() else None,
        "final_url": public_url(final) if final and final.exists() else None,
        "segments": records,
        "inflight": inflight_rows,
        "failed": failed or {},
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    js = job_dir / "runpod_outputs.json"
    js.write_text(json.dumps(status, indent=2), encoding="utf-8")
    js.chmod(0o644)

    rows = []
    for r in records:
        rows.append(
            f"<tr><td>{r['index']}</td><td><a href='{escape(r['output_url'])}'>output</a></td>"
            f"<td>{float(r.get('seconds_total') or 0):.1f}s</td><td>{r.get('swapped')}/{r.get('processed')}</td><td>{escape(str(r.get('attempt') or ''))}</td></tr>"
        )
    queue_rows = []
    for r in inflight_rows:
        queue_rows.append(
            f"<tr><td>{r['index']}</td><td>{escape(str(r['status']))}</td><td>{escape(str(r['attempt']))}</td><td>{float(r['age_seconds']):.0f}s</td></tr>"
        )
    failed_rows = []
    for idx, msg in sorted((failed or {}).items()):
        failed_rows.append(f"<tr><td>{idx}</td><td>{escape(str(msg)[:500])}</td></tr>")

    partial_link = f" · <a href='{escape(status['partial_url'])}'>partial</a>" if status.get("partial_url") else ""
    final_link = f" · <a href='{escape(status['final_url'])}'>final</a>" if status.get("final_url") else ""
    html = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>{escape(name)} RunPod Outputs</title><style>body{{background:#0b0d10;color:#e8edf2;font:14px system-ui,sans-serif}}main{{max-width:1100px;margin:auto;padding:24px}}a{{color:#7cc7ff}}td,th{{padding:8px;border-bottom:1px solid #242a33;text-align:left}}h2{{margin-top:28px}}</style></head>
<body><main><h1>{escape(name)} RunPod Outputs</h1><p>{len(records)}/{total} fertig{partial_link}{final_link} · <a href='runpod_outputs.json'>JSON</a></p>
<h2>Completed</h2><table><tr><th>#</th><th>Output</th><th>GPU seconds</th><th>Swapped</th><th>Attempt</th></tr>{''.join(rows)}</table>
<h2>Queue</h2><table><tr><th>#</th><th>Status</th><th>Attempt</th><th>Age</th></tr>{''.join(queue_rows)}</table>
<h2>Failed</h2><table><tr><th>#</th><th>Error</th></tr>{''.join(failed_rows)}</table>
</main></body></html>"""
    hp = job_dir / "runpod_outputs.html"
    hp.write_text(html, encoding="utf-8")
    hp.chmod(0o644)
    return {"status_url": public_url(hp), "status_json_url": public_url(js), "partial_url": status.get("partial_url"), "final_url": status.get("final_url")}


def parse_endpoint_ids(args: argparse.Namespace) -> list[str]:
    raw: list[str] = []
    if args.endpoint_id:
        raw.extend(args.endpoint_id)
    env = os.getenv("RUNPOD_VIDEO_ENDPOINT_IDS") or os.getenv("RUNPOD_ENDPOINT_IDS")
    if env:
        raw.extend([x.strip() for x in env.split(",") if x.strip()])
    if not raw and os.getenv("RUNPOD_ENDPOINT_ID"):
        raw.append(os.getenv("RUNPOD_ENDPOINT_ID", ""))
    ids: list[str] = []
    for item in raw:
        for part in item.split(","):
            part = part.strip()
            if part and part not in ids:
                ids.append(part)
    if not ids:
        raise SystemExit("Missing --endpoint-id or RUNPOD_VIDEO_ENDPOINT_IDS/RUNPOD_ENDPOINT_ID")
    return ids


def payload_for_segment(data: dict[str, Any], seg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "source_url": data["source_url"],
        "target_url": seg["input_url"],
        "segment_index": int(seg["index"]),
        "mode": args.mode,
        "source_face": args.source_face,
        "target_face": args.target_face,
        "fps": args.fps,
        "audio_offset_seconds": 0,
        "return_base64": True,
        "max_inline_mb": args.max_inline_mb,
    }


def choose_endpoint(endpoints: dict[str, EndpointState], per_endpoint_limit: int) -> EndpointState | None:
    available = [ep for ep in endpoints.values() if ep.active < per_endpoint_limit]
    if not available:
        return None
    return min(available, key=lambda ep: (ep.active, ep.submitted))


def decode_and_store_output(
    *,
    output: dict[str, Any],
    job: InFlightJob,
    job_dir: Path,
) -> dict[str, Any]:
    idx = int(job.segment["index"])
    data = output.get("output_base64")
    if not data:
        raise RuntimeError(f"No output_base64 returned for segment {idx}; warning={output.get('warning')}")
    out = job_dir / f"gpu_out_seg_{idx:03d}.mp4"
    metrics_path = job_dir / f"gpu_out_seg_{idx:03d}.json"
    out.write_bytes(base64.b64decode(data))
    out.chmod(0o644)
    metrics = output.get("metrics") or {}
    metrics["runpod_seconds_total"] = output.get("seconds_total")
    metrics["endpoint_id"] = job.endpoint_id
    metrics["runpod_id"] = job.runpod_id
    metrics["attempt"] = job.attempt
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metrics_path.chmod(0o644)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Run prepared DCE segments through RunPod Queue and publish outputs")
    ap.add_argument("manifest", type=Path, help="runpod_inputs.json from prepare_runpod_inputs.py")
    ap.add_argument("--endpoint-id", action="append", help="RunPod endpoint id. Repeat or comma-separate for load-balancing.")
    ap.add_argument("--api-key", default=os.getenv("RUNPOD_API_KEY"))
    ap.add_argument("--mode", default="source")
    ap.add_argument("--source-face", default="largest")
    ap.add_argument("--target-face", default="center")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--max-inline-mb", type=float, default=75.0)
    ap.add_argument("--limit", type=int, help="Submit only first N segments for smoke testing")
    ap.add_argument("--parallel", type=int, default=1, help="Total concurrent queued RunPod jobs")
    ap.add_argument("--per-endpoint-parallel", type=int, default=1, help="Concurrent jobs per endpoint")
    ap.add_argument("--poll-seconds", type=float, default=10.0)
    ap.add_argument("--timeout-seconds", type=float, default=3600.0)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    name = data["name"]
    job_dir = args.manifest.parent
    segments = data["segments"][: args.limit or None]
    total = len(segments)
    endpoint_ids = parse_endpoint_ids(args)
    endpoints = {eid: EndpointState(eid) for eid in endpoint_ids}

    print(json.dumps({
        "name": name,
        "segments_to_submit": total,
        "source_url": data["source_url"],
        "endpoints": endpoint_ids,
        "parallel": args.parallel,
        "per_endpoint_parallel": args.per_endpoint_parallel,
    }, indent=2))

    if args.dry_run:
        example = payload_for_segment(data, segments[0], args) if segments else {}
        print("DRY_RUN_PAYLOAD=" + json.dumps(example, indent=2))
        return
    if not args.api_key:
        raise SystemExit("Missing --api-key or RUNPOD_API_KEY")

    pending: list[dict[str, Any]] = []
    records_by_idx: dict[int, dict[str, Any]] = {}
    failed: dict[int, str] = {}
    attempts: dict[int, int] = {}
    out_paths_by_idx: dict[int, Path] = {}

    for seg in segments:
        idx = int(seg["index"])
        out = job_dir / f"gpu_out_seg_{idx:03d}.mp4"
        metrics_path = job_dir / f"gpu_out_seg_{idx:03d}.json"
        if not args.no_resume and out.exists() and metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            records_by_idx[idx] = record_for_output(job_dir, idx, metrics)
            out_paths_by_idx[idx] = out
            print(f"resume existing segment {idx}: {out}", flush=True)
        else:
            pending.append(seg)
            attempts[idx] = 0

    inflight: dict[int, InFlightJob] = {}
    partial = job_dir / "gpu_partial.mp4"
    write_status(job_dir, name, records_by_idx, total, partial=partial if partial.exists() else None, inflight=inflight, failed=failed)

    def publish_partial() -> None:
        if not out_paths_by_idx:
            return
        contiguous: list[Path] = []
        expected = 1
        for idx in sorted(out_paths_by_idx):
            if idx != expected:
                break
            contiguous.append(out_paths_by_idx[idx])
            expected += 1
        if contiguous:
            concat_mp4s(contiguous, partial)

    while pending or inflight:
        # Fill queue.
        while pending and len(inflight) < max(1, args.parallel):
            endpoint = choose_endpoint(endpoints, max(1, args.per_endpoint_parallel))
            if endpoint is None:
                break
            seg = pending.pop(0)
            idx = int(seg["index"])
            attempts[idx] = attempts.get(idx, 0) + 1
            payload = payload_for_segment(data, seg, args)
            try:
                resp = submit_job_async(payload, endpoint_id=endpoint.endpoint_id, api_key=args.api_key)
                runpod_id = str(resp.get("id") or resp.get("jobId") or "")
                if not runpod_id:
                    raise RuntimeError("RunPod async response had no id: " + json.dumps(resp)[:1000])
                job = InFlightJob(runpod_id=runpod_id, endpoint_id=endpoint.endpoint_id, segment=seg, attempt=attempts[idx], last_status=str(resp.get("status") or "SUBMITTED"))
                inflight[idx] = job
                endpoint.active += 1
                endpoint.submitted += 1
                print(f"queued segment={idx} endpoint={endpoint.endpoint_id} job={runpod_id} attempt={attempts[idx]} status={job.last_status}", flush=True)
            except Exception as exc:
                msg = f"submit failed attempt {attempts[idx]}: {exc}"
                print(f"ERROR segment={idx} {msg}", flush=True)
                if attempts[idx] <= args.retries:
                    pending.append(seg)
                else:
                    failed[idx] = msg
                    endpoints[endpoint.endpoint_id].failed += 1

        write_status(job_dir, name, records_by_idx, total, partial=partial if partial.exists() else None, inflight=inflight, failed=failed)
        if not inflight:
            if pending:
                time.sleep(args.poll_seconds)
                continue
            break

        time.sleep(args.poll_seconds)
        for idx, job in list(inflight.items()):
            endpoint = endpoints[job.endpoint_id]
            try:
                st = get_job_status(job.runpod_id, endpoint_id=job.endpoint_id, api_key=args.api_key)
                status = str(st.get("status") or "UNKNOWN")
                job.last_status = status
                age = time.time() - job.submitted_at
                print(f"poll segment={idx} endpoint={job.endpoint_id} job={job.runpod_id} status={status} age={age:.0f}s", flush=True)

                if status in TERMINAL_OK:
                    output = extract_output(st)
                    if "output" in st and isinstance(st["output"], dict):
                        output = st["output"]
                    if not output.get("ok"):
                        raise RuntimeError("handler returned not ok: " + json.dumps(output)[:2000])
                    metrics = decode_and_store_output(output=output, job=job, job_dir=job_dir)
                    out_paths_by_idx[idx] = job_dir / f"gpu_out_seg_{idx:03d}.mp4"
                    records_by_idx[idx] = record_for_output(job_dir, idx, metrics)
                    endpoint.active = max(0, endpoint.active - 1)
                    endpoint.completed += 1
                    del inflight[idx]
                    publish_partial()
                    status_urls = write_status(job_dir, name, records_by_idx, total, partial=partial, inflight=inflight, failed=failed)
                    print("SEGMENT_DONE=" + json.dumps({"index": idx, **status_urls}, indent=2), flush=True)
                elif status in TERMINAL_BAD:
                    raise RuntimeError("RunPod terminal failure: " + json.dumps(st)[:2000])
                elif age > args.timeout_seconds:
                    raise TimeoutError(f"RunPod job timed out locally after {age:.0f}s")
            except Exception as exc:
                endpoint.active = max(0, endpoint.active - 1)
                endpoint.failed += 1
                del inflight[idx]
                msg = str(exc)
                print(f"ERROR segment={idx} attempt={job.attempt} {msg}", flush=True)
                if job.attempt <= args.retries:
                    pending.append(job.segment)
                else:
                    failed[idx] = msg
                    write_status(job_dir, name, records_by_idx, total, partial=partial if partial.exists() else None, inflight=inflight, failed=failed)

    publish_partial()
    if failed:
        write_status(job_dir, name, records_by_idx, total, partial=partial if partial.exists() else None, inflight=inflight, failed=failed)
        raise SystemExit("Some segments failed: " + json.dumps(failed, indent=2)[:4000])

    final = job_dir / "gpu_final.mp4"
    final_inputs = [out_paths_by_idx[i] for i in sorted(out_paths_by_idx)]
    concat_mp4s(final_inputs, final)
    status = write_status(job_dir, name, records_by_idx, total, partial=partial if partial.exists() else None, final=final, inflight=inflight, failed=failed)
    print("DONE=" + json.dumps(status | {"final_url": public_url(final)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
