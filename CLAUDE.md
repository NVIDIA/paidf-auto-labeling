<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install runtime dependencies
uv sync
#
# Install dev dependencies (tests/lint)
uv sync --group dev

# Run all tests with coverage
uv run pytest --cov --cov-report=term --cov-report=xml:coverage_report.xml

# Run a single test file
uv run pytest tests/pipeline/test_config_e2e.py -v

# Run a single test function
uv run pytest tests/pipeline/test_config_e2e.py::test_config_load_validate_normalize_dry_run -v

# Lint
uv run ruff check

# Format check
uv run ruff format --check

# Auto-fix lint + format
uv run ruff check --fix && uv run ruff format
```

Note: Some tests are marked `@pytest.mark.e2e_run` — they require GPU and real video files in `./input/`.

## Architecture

This is a **4-stage auto-labeling pipeline** that generates synthetic annotations for video understanding:

```
Input Video → [SR] → [Detection & Tracking] → [VLM JSON] → [MCQ Generation] → Output JSONs
```

### Entry Points

- **`modules/cli.py`** — Main CLI entry point. Handles YAML config loading (via OmegaConf), dotlist overrides, remote I/O (S3/GCS via Multi-Storage-Client), environment variable resolution, and NVCF cloud execution.
- **`modules/pipeline.py`** — Orchestrates stage execution sequentially, manages GPU allocation, timing, error handling, and dry-run mode.

### Pipeline Stages (modules/)

1. **`sr_runner/`** — Super-resolution via SeedVR2 for video and image inputs. Image SR uses the real SeedVR2 subprocess but forces `sp_size=1`. Upscales input media before further processing. `factory.py` resolves GPU list via `resolve_gpu_list()` and passes `gpu_list: list[int]` to `SeedVR2Resolver`. `pipeline.use_multi_gpu` controls whether video SR uses all GPUs in the list (`sp_size = len(gpu_list)`) or only the first.

2. **`detection_and_tracking/`** — RF-DETR object detection (COCO-80 classes) + multi-object tracking. Tracker backends: BoostTrack (default, supports ReID), DeepOCSort, ByteTrack. `factory.py` resolves GPU list and passes `gpu_list: list[int]` to `RFDetrTracker`; tracking always uses `gpu_list[0]`. Outputs `instances.json` and `objects.json`.

3. **`vlm_json/`** — Calls OpenAI-compatible VLM endpoints to produce structured event/metadata JSON. Supports split (video.json + events.json) or single-call mode. Configurable frame sampling (default: 1 fps, 360 px input resolution, 24 max frames).

4. **`mcq_generation/`** — Generates task QA via 4 modes: `window-vlm-llm`, `window-direct-vlm`, `question-driven-vlm-llm`, `metadata-llm`. Outputs `task/mcq.json`, `task/bcq.json`, and `task/open_qa.json` as applicable.
   - `question-driven-vlm-llm` runs in **two pipeline stages**: (1) LLM-only prompt generation from a question bank (runs before SR/tracking); (2) window inference using the generated prompts (deferred until after tracking).

### DAFT Export (`modules/daft_export/`)

Converts in-memory pipeline data into DAFT v3.0-compliant scene directories:

- **`paths.py`** — Canonical `ScenePaths` layout: `raw/`, `contextual/`, `task/`, `sidecars/`.
- **`contextual.py`** — Builds `video.json`/`image.json` and `events.json` from VLM output.
- **`tracking.py`** — Builds `instances.json` and `objects.json` from detection/tracking output.
- **`task.py`** — Builds `mcq.json`, `bcq.json`, and `open_qa.json` from MCQ generation output.
- **`id_translator.py`** — Reconciles VLM-emitted event IDs with tracker instance keys; strips ungrounded `{id: <n>}` annotations from prose fields.
- **`common.py`** — Shared writer (`write_daft_json`) with shape-check guard; `set/get_scene_media_id` derives the DAFT `video_id`/`image_id` from the input filename stem.

Validate output: `tao-daft validate metropolis-v3.0 --path <scene_dir> --raw auto --strict` (requires opt-in `nvidia-tao-daft`; Docker/CI skip validation when the CLI is absent).

### Configuration System

- **`modules/config/`** — Config pipeline: `loader.py` (YAML + OmegaConf merge) → `normalize.py` (path resolution) → `validate.py` → `schema.py` (Pydantic models).
- **`configs/`** — A single blueprint config: `pipeline_example.yaml`. Use CLI dotlist overrides to enable/disable stages and switch MCQ modes.
- `modules/nvcf_msc_utils.py` — NVCF cloud execution and Multi-Storage-Client integration for remote I/O.

### Key Shared Utilities (`modules/al_utils/common.py`)

- **`resolve_gpu_list(gpu_ids)`** — Parses `pipeline.gpu_ids` (`str | int | None`) into `list[int]`. Handles `None`/`"all"` (auto-detect), `int 0`, `"2,3"`, whitespace; raises `ValueError` on non-numeric or negative IDs. Called by both factories before constructing a stage object.
- **`stage_log_file(name, log_dir)`** — Context manager that attaches a `FileHandler` to the root logger for the duration of an in-process stage, writing `<log_dir>/<name>.log` with START/END headers.
- **`run_cmd(...)`** — Runs a subprocess and tees stdout/stderr to console + per-stage log file.

### Key Dependencies

- **OmegaConf** — configuration merging and dotlist overrides
- **Pydantic** — config schema validation
- **OpenAI client** — VLM/LLM API calls (compatible with any OpenAI-spec endpoint)
- **av / ffmpeg-python** — video I/O
- **RF-DETR + tracking libs** — installed as core dependencies

### Docker

The recommended development environment is Docker (CUDA 12.8 + custom FFmpeg):

```bash
./docker/deploy.sh build    # Build image
./docker/deploy.sh shell    # Interactive shell
```

Model checkpoints are selected by config and resolved/downloaded by Python stages. Checkpoints and I/O are bind-mounted at `/workspace/ckpts`, `/workspace/input`, `/workspace/output`.
