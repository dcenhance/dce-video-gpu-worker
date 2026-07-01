# DCE Video GPU Worker for RunPod

GPU segment renderer for the DCE segmented video-render pipeline.

The DCE server should keep doing uploads, public links, queue/status pages, partial MP4s, and final concat. RunPod should only render one 30s segment per job.

## Files

- `handler.py` — RunPod Serverless handler.
- `dce_gpu_faceswap.py` — GPU single-segment renderer using InsightFace + ONNXRuntime CUDA.
- `runpod_segment_client.py` — local DCE-side client for submitting one segment and decoding the returned MP4.
- `prepare_runpod_inputs.py` — DCE-side helper that publishes source + 30s input segment URLs.
- `runpod_orchestrate_job.py` — DCE-side orchestrator that submits prepared segments to RunPod, publishes GPU outputs, partial MP4, and final concat.
- `Dockerfile` — CUDA 11.8 runtime image for RunPod.
- `requirements.txt` — pinned runtime deps.

## Request shape

```json
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
    "return_base64": true,
    "max_inline_mb": 75
  }
}
```

Output includes metrics and either:

- `output_base64` for inline return, or
- `output_uploaded` if `output_put_url` is used.

For 30s 720p segments inline base64 may be large. If RunPod response limits bite, add a DCE upload endpoint/presigned PUT URL and send `output_put_url`.

## Build image

```bash
cd /srv/apps/dce-video-gpu-worker
docker build -t dce-video-gpu-worker:latest .
```

For RunPod, push to Docker Hub/GHCR, then create a Serverless template from that image.

## RunPod config recommendation

Start simple:

- GPU: RTX 3090 24GB or A4000/A4500 16GB
- Min workers: 0
- Max workers: 1 first, later 2-4
- Idle timeout: 180s while testing
- Execution timeout: at least 1800s for 30s source segments during early testing
- Container disk: 20-40GB

## Prepare public 30s inputs on DCE server

```bash
python3 prepare_runpod_inputs.py \
  SOURCE_IMAGE TARGET_VIDEO \
  --name my_long_video \
  --segment-seconds 30
```

This creates:

```text
https://dcenhancements.com/archive/outputs/video-gpu/runpod_jobs/my_long_video/index.html
https://dcenhancements.com/archive/outputs/video-gpu/runpod_jobs/my_long_video/runpod_inputs.json
```

## Orchestrate prepared segments through RunPod Queue

Single endpoint, one queued job at a time:

```bash
python3 runpod_orchestrate_job.py \
  /srv/apps/website/public/archive/outputs/video-gpu/runpod_jobs/my_long_video/runpod_inputs.json \
  --endpoint-id "$RUNPOD_ENDPOINT_ID" \
  --mode source \
  --parallel 1
```

Multiple endpoints / simple load-balancing:

```bash
python3 runpod_orchestrate_job.py \
  /srv/apps/website/public/archive/outputs/video-gpu/runpod_jobs/my_long_video/runpod_inputs.json \
  --endpoint-id ENDPOINT_A \
  --endpoint-id ENDPOINT_B \
  --mode source \
  --parallel 2 \
  --per-endpoint-parallel 1 \
  --retries 1
```

The orchestrator uses RunPod async Queue (`/run` + `/status/{id}`), so `IN_QUEUE` is handled correctly. It publishes status while jobs are running:

```text
gpu_out_seg_001.mp4
gpu_out_seg_002.mp4
gpu_partial.mp4 after contiguous finished segments
gpu_final.mp4 at the end
runpod_outputs.html/json status page with completed/inflight/failed sections
```

## DCE local one-segment client dry-run

```bash
python3 runpod_segment_client.py \
  --endpoint-id "$RUNPOD_ENDPOINT_ID" \
  --source-url "https://dcenhancements.com/source.jpg" \
  --target-url "https://dcenhancements.com/segment_001.mp4" \
  --output /tmp/segment_001_faceswap.mp4 \
  --mode source \
  --dry-run
```

## Actual one-segment submit

```bash
python3 runpod_segment_client.py \
  --endpoint-id "$RUNPOD_ENDPOINT_ID" \
  --source-url "https://dcenhancements.com/source.jpg" \
  --target-url "https://dcenhancements.com/segment_001.mp4" \
  --output /tmp/segment_001_faceswap.mp4 \
  --mode source
```

## Next integration step

Extend `the DCE server-side segmented pipeline` with:

```text
--gpu-provider runpod
--runpod-endpoint-id ENDPOINT
--runpod-parallel 1..4
```

Then the existing 30s segmentation can submit segment jobs to RunPod instead of calling local CPU `render()`.

## Quality defaults

Use:

```text
mode=source
source_face=largest
target_face=center
fps=30
crf=18
```

Do not use `preview240x4` for user-facing quality. That is TestOnly.
