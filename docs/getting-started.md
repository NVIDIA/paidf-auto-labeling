<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Getting Started

Runtime setup and first-run commands for the auto-labeling pipeline.

**Also see:** [Config reference](config-reference.md) · [E2E testing](e2e-testing.md) · [MCQ modes](mcq-modes.md) · [Remote I/O](remote-io.md)

---

## Requirements

**For a full pipeline run**

- Docker with NVIDIA GPU access
- Input video or image
- VLM and LLM endpoints (or NVCF auto-detect)
- Writable output directory

**GPU and capacity**

- SR and detection/tracking use `pipeline.gpu_ids` inside the container.
- VLM JSON and MCQ call remote endpoints — no pipeline GPU unless you host endpoints on the same machine.
- Reserve a dedicated GPU for SeedVR2 SR at default settings (`res_h=720`, `res_w=1280`, `window_frames=128`, `overlap_frames=64`); `seedvr2_7b` needs more VRAM than `seedvr2_3b`.
- Tracking footprint is smaller (~0.35 GB RF-DETR + ~0.63 GB default ReID).
- Local Qwen VLM/LLM recipe: one GPU per endpoint; keep them out of `pipeline.gpu_ids` unless you have verified headroom.
- Pin pipeline GPUs with `pipeline.gpu_ids`. Multi-GPU SR: `pipeline.use_multi_gpu=true` (all selected GPUs); tracking uses the first.
- Disk: Docker image plus on-demand checkpoints under `ckpts/` (7B SR checkpoint ~31 GB).
- Shared memory: repo `docker-compose` uses `shm_size: "8gb"`; published `docker run` examples use `--ipc=host --shm-size=32g`.

---

## Runtime image

| Situation | Image step | Run style |
|-----------|------------|-----------|
| Repo checkout | `./docker/deploy.sh build` or `./docker/deploy.sh check` | `./docker/deploy.sh shell -lc 'uv run python modules/cli.py ...'` |
| Published image | `docker pull <NGC_IMAGE>` | `docker run ... "${AUTO_LABELING_IMAGE}" bash -lc 'uv run python modules/cli.py ...'` |

- **Repo checkout** — develop from this tree; checkout mounted at `/workspace`; use `input/` and `output/` under the repo.
- **Published image** — run a release only; mount host paths explicitly (e.g. `/input`, `/output`).

Supported runtime is **Docker only**. The image ships `uv` and `/opt/venv`; `UV_NO_SYNC=1` prevents `uv run` from mutating the env at runtime. Host-side `uv run` is for dev checks (tests, lint) — not validated for pipeline execution.

---

## CLI shape

```bash
uv run python modules/cli.py --config configs/pipeline_example.yaml key=value key=value ...
```

| Pattern | Meaning |
|---------|---------|
| `data.0.inputs.video_path` | First sample input (video or image) |
| `data.0.output.out_dir` | First sample DAFT scene root |
| `data.1.*`, `data.2.*`, … | Additional samples |
| `<stage>.enabled=false` | Disable a stage |
| `host.docker.internal` | Reach host VLM/LLM from inside Docker |

---

## Workflow matrix

| Workflow | Endpoints | Key overrides |
|----------|-----------|---------------|
| Full pipeline | VLM + LLM | Default stages enabled |
| Dry run | None | `--dry-run`; disable stages as needed |
| SR + tracking | None | `vlm_json.enabled=false mcq_generation.enabled=false` |
| VLM JSON only | VLM | Disable SR/tracking/MCQ as needed |
| MCQ from frames | Mode-dependent | `mcq_generation.enabled=true mcq_generation.mode=<mode>` |
| MCQ from captions | LLM | `mcq_generation.mode=metadata-llm` |

---

## Prompts and cookbooks

Defaults live under `cookbooks/` — use-case folders own domain assets; `cookbooks/shared/` holds cross-cutting prompts.

| Asset | Location |
|-------|----------|
| VLM JSON | `cookbooks/shared/prompts/vlm_json/` |
| MCQ window/metadata | `cookbooks/traffic/prompts/mcq/` |
| Question-driven templates | `cookbooks/shared/prompts/mcq/question_driven_vlm_llm/templates/` |
| Question banks | `cookbooks/<use_case>/question_bank.json` |

New domain: copy a cookbook folder or add `cookbooks/<slug>/question_bank.json`, then set `mcq_generation.window_metadata_extraction.question_bank_file`. Override `scene_prompt_file` / `mcq_prompt_file` for fixed window prompts. Details: [MCQ modes](mcq-modes.md).

---

## First run (repo-local)

```bash
./docker/deploy.sh check    # or: ./docker/deploy.sh build
mkdir -p input output
cp /path/to/clip.mp4 input/clip.mp4
```

**Dry run** (config only — no GPU, no endpoints):

```bash
./docker/deploy.sh shell -lc '
  uv run python modules/cli.py --dry-run --config configs/pipeline_example.yaml \
    super_resolution.enabled=false \
    detection_and_tracking.enabled=false \
    vlm_json.enabled=false \
    mcq_generation.enabled=false \
    data.0.inputs.video_path="/workspace/input/clip.mp4" \
    data.0.output.out_dir="/workspace/output/smoke" \
    endpoints.vlm.url="" endpoints.vlm.model="" \
    endpoints.llm.url="" endpoints.llm.model=""
'
```

**Full pipeline** (all stages — replace endpoint placeholders):

```bash
./docker/deploy.sh shell -lc '
  uv run python modules/cli.py --config configs/pipeline_example.yaml \
    data.0.inputs.video_path="/workspace/input/clip.mp4" \
    data.0.output.out_dir="/workspace/output/full_run" \
    endpoints.vlm.url="http://host.docker.internal:<VLM_PORT>/v1" \
    endpoints.vlm.model="<VLM_MODEL_ID>" \
    endpoints.llm.url="http://host.docker.internal:<LLM_PORT>/v1" \
    endpoints.llm.model="<LLM_MODEL_ID>"
'
```

More smoke levels (SR+tracking only, full matrix, timings): [End-to-end testing](e2e-testing.md).

---

## Published image command shape

```bash
export AUTO_LABELING_IMAGE="<NGC_IMAGE>"
docker pull "${AUTO_LABELING_IMAGE}"

docker run --rm --gpus all --ipc=host --shm-size=32g \
  -e VLM_API_KEY \
  -e LLM_API_KEY \
  -e NVIDIA_API_KEY \
  -e OPENAI_API_KEY \
  -e HF_TOKEN \
  -v "$(pwd)/input:/input:ro" \
  -v "$(pwd)/output:/output" \
  -w /workspace \
  "${AUTO_LABELING_IMAGE}" \
  bash -lc '
    uv run python modules/cli.py --config configs/pipeline_example.yaml \
      vlm_json.enabled=false \
      mcq_generation.enabled=false \
      data.0.inputs.video_path="/input/clip.mp4" \
      data.0.output.out_dir="/output/sr_tracking" \
      endpoints.vlm.url="" endpoints.vlm.model="" \
      endpoints.llm.url="" endpoints.llm.model=""
  '
```

Swap overrides for a full pipeline run (enable all stages and set `endpoints.*`).

---

## Output

Each `out_dir` is a DAFT scene: `raw/`, `contextual/`, `task/`, `sidecars/`, plus `logs/` and `config.yaml`. Some MCQ modes also write `prompts/`.

---

## Development checks

```bash
uv sync --group dev
uv run pytest tests/pipeline/test_config_e2e.py -v
uv run ruff check
uv run ruff format --check
```

GPU/media `e2e_run` tests: see [End-to-end testing](e2e-testing.md).
