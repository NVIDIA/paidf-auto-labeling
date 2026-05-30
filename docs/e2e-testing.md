<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# End-to-End Testing

Use this guide when you need more than unit tests: real media, the Docker runtime, GPU-backed SR/tracking, and live VLM/LLM calls.

Related: [Getting started](getting-started.md) (first run) · [Configuration reference](config-reference.md) (keys and endpoints).

---

## Validation levels

| Level | What it checks | Requires |
|-------|----------------|----------|
| Config dry-run | Config load, schema validation, path normalization, stage selection | No GPU, no endpoints |
| SR + tracking smoke | Docker image, GPU, checkpoint download, SR/tracking outputs | GPU |
| Endpoint smoke | VLM JSON or MCQ with SR/tracking disabled | VLM and/or LLM endpoint |
| Full pipeline smoke | One sample through all enabled stages | GPU + VLM + LLM |
| Full matrix | Image + video, all MCQ modes, multi-sample, optional multi-GPU | GPU + VLM + LLM |

**Recommended order**

1. Config dry-run
2. SR + tracking on one short clip or image
3. One full pipeline smoke
4. Full matrix — release checks, CI validation, or hardware bring-up only

---

## Quick smoke commands

Prep (paths are inside the container at `/workspace/...`):

```bash
mkdir -p input output/test
cp /path/to/clip.mp4 input/clip.mp4
```

### Config dry-run

```bash
./docker/deploy.sh shell -lc '
  uv run python modules/cli.py --dry-run --config configs/pipeline_example.yaml \
    super_resolution.enabled=false \
    detection_and_tracking.enabled=false \
    vlm_json.enabled=false \
    mcq_generation.enabled=false \
    data.0.inputs.video_path="/workspace/input/clip.mp4" \
    data.0.output.out_dir="/workspace/output/test/dry_run" \
    endpoints.vlm.url="" endpoints.vlm.model="" \
    endpoints.llm.url="" endpoints.llm.model=""
'
```

### SR + tracking smoke

```bash
./docker/deploy.sh shell -lc '
  uv run python modules/cli.py --config configs/pipeline_example.yaml \
    vlm_json.enabled=false \
    mcq_generation.enabled=false \
    data.0.inputs.video_path="/workspace/input/clip.mp4" \
    data.0.output.out_dir="/workspace/output/test/sr_tracking" \
    endpoints.vlm.url="" endpoints.vlm.model="" \
    endpoints.llm.url="" endpoints.llm.model=""
'
```

### Full pipeline smoke

Keep all stages enabled; set `endpoints.vlm.*` and `endpoints.llm.*`. Use `host.docker.internal` when VLM/LLM servers run on the host (see [Endpoints](config-reference.md#endpoints)).

```bash
./docker/deploy.sh shell -lc '
  uv run python modules/cli.py --config configs/pipeline_example.yaml \
    data.0.inputs.video_path="/workspace/input/clip.mp4" \
    data.0.output.out_dir="/workspace/output/test/full_pipeline" \
    endpoints.vlm.url="http://host.docker.internal:<VLM_PORT>/v1" \
    endpoints.vlm.model="<VLM_MODEL_ID>" \
    endpoints.llm.url="http://host.docker.internal:<LLM_PORT>/v1" \
    endpoints.llm.model="<LLM_MODEL_ID>"
'
```

---

## Full matrix

Broad regression coverage: dry-run, SR, detection/tracking, VLM JSON, all four MCQ modes, full pipeline on video and image, multi-sample runs, and optional multi-GPU SR.

Not a default first run. For a structured agent-driven matrix, use the repo **test-pipeline** skill; for manual commands, extend the smoke patterns above per [MCQ modes](mcq-modes.md).

---

## Expected runtime (SR)

SR dominates local GPU time. With default settings (`window_frames=128`, `overlap_frames=64`, stride 64), a 300-frame smoke clip yields **4 SR windows**: 0–127, 64–191, 128–255, and a final tail window.

Rough per-window averages (SR-only runs, including load, VAE, decode/encode, and write):

| Variant | Time per window |
|---------|-----------------|
| `seedvr2_3b` | ~1–2 min |
| `seedvr2_7b` | ~1.5× slower than 3B |

Tracking and VLM JSON are usually shorter than SR on this clip. MCQ time depends on endpoint latency, window count, retries, and VLM verify settings.

---

## Expected outputs

A successful run writes a **DAFT scene** under `out_dir`:

- `raw/`, `contextual/`, `task/`, `sidecars/`
- `logs/` and `config.yaml` (effective config snapshot)

Some MCQ modes also write `prompts/` (for example question-driven prompt artifacts).

**Smoke checks by stage**

| Stage | Look for |
|-------|----------|
| SR | `sidecars/sr_output.<ext>` |
| Tracking | `contextual/objects.json`, `contextual/instances.json` |
| VLM JSON | `contextual/video.json` + `contextual/events.json` (video), or `contextual/image.json` (image) |
| MCQ / full pipeline | At least one of `task/mcq.json`, `task/bcq.json`, `task/open_qa.json` |

---

## Automated checks

Host-side config tests (no GPU):

```bash
uv run pytest tests/pipeline/test_config_e2e.py -v
```

Tests marked `e2e_run` (when present) need Docker, GPUs, model downloads, and media under `input/`:

```bash
uv run pytest -m e2e_run -v
```
