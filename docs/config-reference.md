<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Configuration Reference

All configuration is driven by `configs/pipeline_example.yaml` plus OmegaConf dotlist overrides.

**Defaults in this reference match the blueprint YAML** unless noted; Pydantic schema defaults may differ — see `modules/al_utils/schema/`.

Run examples inside Docker (`./docker/deploy.sh shell` or a published image). Examples use `uv run python modules/cli.py` because the image ships `uv` and a prebuilt env. Host-side `uv run` is fine for local dev checks only.

```bash
uv run python modules/cli.py --config configs/pipeline_example.yaml [key=value ...]
```

Dry run (no stage execution): add `--dry-run` to the command above.

**Sections:** [Data I/O](#data-io) · [GPU](#gpu) · [Empty output policy](#empty-output-policy) · [SR](#super-resolution) · [Tracking](#detection--tracking) · [VLM JSON](#vlm-json) · [Endpoints](#endpoints) · [MCQ verify](#mcq-verification) · [Gotchas](#schema-gotchas) · [Env vars](#environment-variables) · [Checkpoints](#model-checkpoints) · [Docker](#docker)

---

## Stage enable/disable

```bash
super_resolution.enabled=false
detection_and_tracking.enabled=false
vlm_json.enabled=false
mcq_generation.enabled=false
```

---

## Data I/O

| Key | Required | Description |
|-----|----------|-------------|
| `data[*].inputs.video_path` | **yes** | Video or image path (local or remote `s3://`, `https://`, …). Supported images: `.jpg` `.jpeg` `.png` `.webp` `.bmp`. Supported videos: `.mp4` `.mov` `.m4v` (mp4-family containers handled by the LGPL-only FFmpeg mov demuxer; matroska, AVI, and WebM are intentionally unsupported). |
| `data[*].inputs.vlm_video_path` | optional | Override which media file VLM JSON (and MCQ) consumes. If omitted, the pipeline picks: `sidecars/<track_stem>_tracking_red_id.<ext>` → SR output → original. |
| `data[*].inputs.metadata_json_path` | for `metadata-llm` | Window captions sidecar JSON (`metadata.json` from a prior window MCQ run). `data[*].inputs.video_path` must still point to an existing media file for pipeline validation. |
| `data[*].output.out_dir` | **yes** | DAFT scene root. Stage outputs use the fixed layout under `raw/`, `contextual/`, `task/`, and `sidecars/`. Use `s3://…` to upload on success. |

Per-file output paths are not user-configurable. Setting `out_dir` is sufficient; the pipeline writes canonical paths such as `contextual/video.json`, `contextual/events.json`, and `task/{mcq,bcq,open_qa}.json`.

---

## GPU

| Key | Default | Description |
|-----|---------|-------------|
| `pipeline.gpu_ids` | `"all"` | GPU IDs for pipeline stages (`"all"` / `"0"` / `"2,3"`) |
| `pipeline.use_multi_gpu` | `false` | SR multi-GPU — uses all GPUs in `gpu_ids` (`sp_size` set automatically) |

- Tracking always uses the first GPU in `gpu_ids`.
- VLM JSON and MCQ call remote endpoints — no local GPU.
- Multi-GPU SR needs NVLink or PCIe P2P between GPUs.

**Capacity planning**

- Reserve a dedicated GPU for SeedVR2 SR at default settings (`res_h=720`, `res_w=1280`, `window_frames=128`, `overlap_frames=64`); `seedvr2_7b` needs more VRAM than `seedvr2_3b`.
- RF-DETR + BoostTrack ReID are much smaller than SR.
- Local VLM/LLM servers run outside the pipeline container; the shipped Qwen recipe expects one GPU per endpoint — exclude those GPUs from `pipeline.gpu_ids` unless you have verified headroom.

---

## Empty output policy

`pipeline.empty_output_policy` (`warn` | `fail`, default: `warn`)

- `warn`: if a stage fails or produces no output, log a warning and continue. SR specifically falls back to the original input video for downstream stages on failure (e.g. GPU OOM).
- `fail`: any stage failure or missing expected output is fatal.

When SR fallback happens under `warn`, `sidecars/pipeline_status.json` records `status=completed_degraded`, `degraded=true`, and fallback metadata. Under `fail`, the pipeline aborts before writing a completed scene status.

**Fallback chain**: SR → tracking → VLM JSON → MCQ. SR→tracking has an implicit fallback (original media on SR failure). When MCQ generation is enabled, VLM JSON failure is non-fatal so the pipeline can continue to the MCQ stage. All other missing upstream outputs are fatal unless `empty_output_policy=warn` allows the stage to continue without its optional artifacts.

---

## Super Resolution

| Key | Default | Description |
|-----|---------|-------------|
| `super_resolution.variant` | `seedvr2_3b` | Model variant (`seedvr2_3b` or `seedvr2_7b`) |
| `super_resolution.res_h` | `720` | Output height in pixels |
| `super_resolution.res_w` | `1280` | Output width in pixels |
| `super_resolution.seed` | `42` | Diffusion seed for reproducibility |
| `super_resolution.window_frames` | `128` | Temporal window size (frames) for diffusion |
| `super_resolution.overlap_frames` | `64` | Overlap between consecutive windows |
| `super_resolution.window_timeout` | `3600` | Wall-clock timeout for each SR window. On timeout, SR is treated as failed and `empty_output_policy` decides fallback vs abort. |
| `super_resolution.out_fps` | — | Output FPS (`null` = preserve input FPS) |

SR runs via `torchrun` and requires Docker (SeedVR2 is only available inside the image at `/opt/seedvr/`). Image inputs use SeedVR2 diffusion processing, and SR/tracking sidecar outputs use `.png` extensions for image scenes.

---

## Detection & Tracking

| Key | Default | Description |
|-----|---------|-------------|
| `detection_and_tracking.model` | `"rfdetr"` | Detector+tracker implementation |
| `detection_and_tracking.tracker` | `"boosttrack"` | Tracker backend: `boosttrack`, `deepocsort`, `bytetrack` |
| `detection_and_tracking.threshold` | `0.2` | Detection confidence threshold |
| `detection_and_tracking.iou_threshold` | `0.3` | IoU threshold for tracking association |
| `detection_and_tracking.classes` | car, truck, bus, motorcycle, bicycle, person (blueprint; schema default omits `person`) | COCO class names to detect |
| `detection_and_tracking.per_class` | `true` | Track each class independently |
| `detection_and_tracking.min_track_frames` | `5` | Minimum total frames for a track to appear in output |
| `detection_and_tracking.use_reid` | `true` | Enable appearance-based ReID (**BoostTrack only**). Set `false` to disable and skip weight download. |
| `detection_and_tracking.reid_weights` | `""` | ReID weights path. If empty with `use_reid=true`, downloads to `ckpts/reid/clip_vehicleid.pt`. |
| `detection_and_tracking.save_video` | `true` | Save `sidecars/<track_stem>_detection.<ext>` and `sidecars/<track_stem>_tracking.<ext>` |
| `detection_and_tracking.save_video_red_id` | `true` | Save extra red-box overlay media (`sidecars/<track_stem>_tracking_red_id.<ext>`) |
| `detection_and_tracking.save_vis` | `false` | Save per-frame visualization images |
| `detection_and_tracking.save_rgb` | `false` | Save extracted per-frame RGB JPEGs under `sidecars/rgb/` |
| `detection_and_tracking.copy_video` | `false` | Copy source video into tracking output folder |

---

## VLM JSON

| Key | Default | Description |
|-----|---------|-------------|
| `vlm_json.model` | `"vlm"` | VLM pipeline implementation |
| `vlm_json.scene_prompt_file` | — | Optional override for the video/image scene metadata prompt. When unset, shipped cookbook defaults are used. |
| `vlm_json.events_prompt_file` | — | Optional override for the video events prompt. Used for video inputs; when unset, shipped cookbook defaults are used. |
| `vlm_json.default_video_json_prompt_file` | `cookbooks/shared/prompts/vlm_json/video_json_prompt.md` | Shipped video metadata prompt default |
| `vlm_json.default_video_events_prompt_file` | `cookbooks/shared/prompts/vlm_json/video_events_prompt.md` | Shipped video events prompt default |
| `vlm_json.default_image_json_prompt_file` | `cookbooks/shared/prompts/vlm_json/image_caption_prompt.md` | Shipped image metadata prompt default |
| `vlm_json.split_json_calls` | `true` | Two VLM calls per video (video.json then events.json) — more stable |
| `vlm_json.structured_output` | `"openai"` | JSON forcing mode: `auto` (NIM→guided_json, else json_object), `nim`, `openai`, `off` |
| `vlm_json.temperature` | `0.0` | Sampling temperature — keep at 0 for deterministic output |
| `vlm_json.frame_fps` | `1.0` | Frame sampling rate |
| `vlm_json.resolution` | `360` | Input resolution |
| `vlm_json.max_frames` | `24` | Max frames per call |
| `vlm_json.max_tokens` | `8192` | Output token budget |
| `vlm_json.timeout` | `600` | Per-call API timeout in seconds |
| `vlm_json.rate_limit` | `0` | Min seconds between VLM calls (0 = no limit) |

**Input priority**: `data[*].inputs.vlm_video_path` (explicit override) → `sidecars/<track_stem>_tracking_red_id.<ext>` → SR output → original media. VLM JSON reads sampled frames only — it does **not** read `objects.json`/`instances.json`. If your media lacks overlaid bounding boxes, provide a matching prompt.

---

## Endpoints

| Key | Default | Description |
|-----|---------|-------------|
| `endpoints.vlm.url` | — | VLM base URL — accepts `http://host:port` or `http://host:port/v1` |
| `endpoints.vlm.model` | — | VLM model ID as served by the endpoint |
| `endpoints.vlm.retries` | `3` | Retry attempts per VLM call (exponential backoff) |
| `endpoints.vlm.retry_backoff_s` | `5.0` | Base backoff seconds; delay = `base × 2^attempt`, capped at 60s |
| `endpoints.llm.url` | — | LLM base URL |
| `endpoints.llm.model` | — | LLM model ID |
| `endpoints.llm.retries` | `3` | Retry attempts per LLM call |
| `endpoints.llm.retry_backoff_s` | `5.0` | Base backoff seconds |

**Endpoint URL/model resolution order** (first non-empty wins):
1. `VLM_BASE_URL` / `VLM_MODEL` env vars (override config)
2. `endpoints.vlm.url` / `endpoints.vlm.model` in config
3. NVCF auto-detect (`detect_nvcf_vlm_endpoint()`) — fills in only when both 1 and 2 are empty

Same pattern for LLM: `LLM_BASE_URL` / `LLM_MODEL` → config → NVCF auto-detect.

**API key priority:**
- VLM: `VLM_API_KEY` → `NVIDIA_API_KEY` → `OPENAI_API_KEY` → `"EMPTY"`
- LLM: `LLM_API_KEY` → `NVIDIA_API_KEY` → `OPENAI_API_KEY` → `"EMPTY"`

`"EMPTY"` works for local endpoints (e.g. vLLM) that require no auth.

**From inside Docker**, host-side servers are reachable via `host.docker.internal`, not `localhost`.

**Missing endpoint behavior**: On NVCF, if endpoints are still missing after auto-detection, the affected stages are gracefully disabled with a warning. On non-NVCF (local Docker), missing endpoints for enabled VLM/LLM stages are a hard error (exit code 2).

---

## Pipeline flags

| Key | Default | Description |
|-----|---------|-------------|
| `pipeline.model_cache_path` | `ckpts` | Checkpoint directory (overridden by `MODEL_CACHE_PATH`) |

---

## MCQ verification

| Key | Default | Description |
|-----|---------|-------------|
| `mcq_generation.window_metadata_extraction.vlm_verify_enabled` | `true` (blueprint) | Run an extra VLM pass over finalized per-window MCQ answers. The schema default is `false`, but the blueprint config enables verifier diagnostics |
| `mcq_generation.window_metadata_extraction.vlm_verify_prompt_file` | `null` | Optional cookbook verify template override. Leave unset to use `cookbooks/shared/prompts/mcq/vlm_verify/verify_prompt.md`; the runner injects the active verification policy |
| `mcq_generation.window_metadata_extraction.vlm_verify_apply_corrections` | `false` | When false, verifier only emits reasoning traces; set true only when the verifier may propose answer corrections under strict visual-evidence rules |

For per-mode VLM verify artifacts and prompt hashes, see [MCQ prompt artifacts](mcq-modes.md#prompt-artifacts).

---

## Schema gotchas

- Output paths are fixed by the DAFT layout — set only `data[*].output.out_dir`.
- `metadata-llm` requires `data[*].inputs.metadata_json_path` at schema validation (including `--dry-run`). You must still set `data[*].inputs.video_path` to an existing media file.
- `metadata-llm` is LLM-only; schema auto-disables `vlm_verify_enabled`.

---

## Environment variables

| Variable | Description |
|----------|-------------|
| `VLM_API_KEY` | Primary API key for VLM endpoint |
| `LLM_API_KEY` | Primary API key for LLM endpoint |
| `NVIDIA_API_KEY` | Fallback for both VLM and LLM (also read by NVIDIA SDK) |
| `OPENAI_API_KEY` | Fallback for both VLM and LLM (also read by OpenAI SDK) |
| `MODEL_CACHE_PATH` | Checkpoint directory (overrides `pipeline.model_cache_path`) |
| `HF_TOKEN` | HuggingFace token for gated model downloads |
| `VLM_BASE_URL` / `VLM_MODEL` | Override config `endpoints.vlm.*` (takes precedence over config file; also used for NVCF auto-detect). **Not forwarded by docker-compose** — use CLI overrides (`endpoints.vlm.url=...`) when running via Docker. |
| `LLM_BASE_URL` / `LLM_MODEL` | Override config `endpoints.llm.*` (takes precedence over config file; also used for NVCF auto-detect). **Not forwarded by docker-compose** — use CLI overrides (`endpoints.llm.url=...`) when running via Docker. |
| `HF_REPO_SEEDVR2_3B` / `_7B` | Override SeedVR2 HuggingFace repo IDs (mirrors/forks). **Not forwarded by docker-compose.** |

---

## Model checkpoints

Checkpoints are auto-downloaded on demand into `<model_cache_path>/`. Model choice comes from the config; Docker helpers do not decide which checkpoints to use.

- SR: selected by `super_resolution.variant`
- Tracking: RF-DETR weights downloaded automatically (~0.35 GB); ReID weights downloaded if `use_reid=true` (~0.63 GB)
- If a download fails:
  - `empty_output_policy=fail` → run aborts
  - `empty_output_policy=warn` → stage continues best-effort
- If downloads fail due to a gated HuggingFace repo: set `HF_TOKEN`

---

## Docker

The container starts as root; `docker/entrypoint.sh` uses **gosu** to remap the internal `nvidia` user to your host UID/GID (`PUID`/`PGID`), so bind-mounted directories are writable without rebuilds.

```bash
./docker/deploy.sh build          # build image (once)
./docker/deploy.sh shell          # interactive shell
./docker/deploy.sh shell -lc "cmd"  # non-interactive one-liner
```

`PUID`/`PGID` are injected automatically by `deploy.sh`. Manual override:
```bash
PUID="$(id -u)" PGID="$(id -g)" docker compose -f docker/docker-compose.yml run --rm --user 0:0 auto-labeling bash
```

The `docker-compose.yml` passes through `VLM_API_KEY`, `LLM_API_KEY`, `NVIDIA_API_KEY`, `OPENAI_API_KEY`, and `HF_TOKEN` from the host environment automatically.
