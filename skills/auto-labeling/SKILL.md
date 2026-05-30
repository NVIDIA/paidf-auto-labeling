---
name: auto-labeling
description: Build Docker-backed PAIDF auto-labeling commands for SR, tracking, VLM JSON, and MCQ runs. Use for pipeline planning or execution requests. Do NOT use for unrelated Docker tasks.
license: Apache-2.0
owner: NVIDIA
service: auto-labeling
version: 1.0.0
reviewed: 2026-05-20
metadata:
  author: NVIDIA
---

# PAIDF Auto-Labeling Pipeline Skill

You help users go from "I have videos or images" to a completed Docker-backed pipeline run for the NVIDIA Metropolis Physical AI Data Factory (PAIDF) auto-labeling pipeline (`modules/cli.py`). Build complete commands, then obtain explicit user approval before executing Docker commands.

The pipeline has 4 stages: Super Resolution (SR, SeedVR2) → Detection & Tracking (RF-DETR + BoostTrack) → VLM JSON → Multiple-Choice Question (MCQ) Generation. This repo ships a single blueprint config (`configs/pipeline_example.yaml`) meant to be adapted via CLI overrides.

## Routing (Read First)

The default flow is the **general 6-step pipeline run** below. Two shipped use-case examples are config-only specialisations of that same flow (they set `mcq_generation.mode=question-driven-vlm-llm`, `single_window=true`, and a question bank). When the request matches one of them, read the corresponding subdoc for the right overrides; otherwise stay on the general flow.

| Request looks like | Use the |
|--------------------|---------|
| "Run the pipeline", "process this video", "tracking + MCQ", batch runs, partial stages, NGC/published-image runs | General 6-step flow below |
| "Caption these images with my own questions", "label this batch with yes/no questions", "build a reusable question bank" | [`references/custom-caption.md`](references/custom-caption.md) |
| "Generate PAS captions", "person attribute caption", "describe what each person is wearing", "use the shipped person_attributes bank" | [`references/person-attribute-caption.md`](references/person-attribute-caption.md) |

## Instructions

Work through these steps in order. Combine steps into a single conversational turn when the user has already provided enough information — don't ask for things they've already told you.

---

### Step 1: Understand the Goal (Interview)

Ask these questions in one message (skip any already answered):

1. **Stages**: Which pipeline stages do they need? (SR / tracking / VLM JSON / MCQ — or full pipeline)
2. **Inputs**: How many videos? Do they already have intermediate outputs (e.g., existing `sidecars/metadata.json` for `metadata-llm`)?
3. **MCQ mode** (if MCQ is involved): See Step 2 and `references/mcq-mode-guide.md`.
4. **Endpoints**: Do they have VLM/LLM endpoints? What environment (local Docker, NVCF, OpenAI-compatible)?
5. **Checkpoints**: Model checkpoints are resolved from config and auto-downloaded by stage initialization; ask only if they need a custom `pipeline.model_cache_path` or `HF_TOKEN`.

---

### Step 2: Recommend a Preset Config

Use `configs/pipeline_example.yaml` for all workflows and select behavior via overrides:

- Full pipeline: defaults in the YAML (SR + tracking + VLM JSON + MCQ).
- SR-only: set `detection_and_tracking.enabled=false vlm_json.enabled=false mcq_generation.enabled=false`.
- Tracking-only: set `super_resolution.enabled=false vlm_json.enabled=false mcq_generation.enabled=false`.
- VLM JSON only: set `super_resolution.enabled=false detection_and_tracking.enabled=false mcq_generation.enabled=false`.
- MCQ modes: set `mcq_generation.enabled=true` plus `mcq_generation.mode=<mode>` (see the mode guide).

**Rule**: to disable a stage, add `<stage>.enabled=false` as a CLI override rather than switching config files.

For edge cases (partial pipelines, stage chaining across separate runs), read `references/config-decision-tree.md`.

---

### Step 3: Collect Required Parameters

Only ask for what their chosen config actually requires:

**Always required:**
- `data.0.inputs.video_path` — input media path (video **or image**) or remote (`s3://`, `msc://`, `https://`)
- `data.0.output.out_dir` — output root directory

**Any stage that calls a VLM** (e.g. `vlm_json`, or MCQ modes like `window-direct-vlm` / `window-vlm-llm` / `question-driven-vlm-llm`):
- `endpoints.vlm.url` — VLM base URL (e.g., `http://vlm-server:8000/v1`)
- `endpoints.vlm.model` — model ID string

> **Docker gotcha**: If the VLM/LLM servers are running on the **host machine**, use `host.docker.internal` instead of `localhost`. Example: `endpoints.vlm.url="http://host.docker.internal:18002/v1"`. Using `localhost` from inside Docker silently fails with connection errors.

For endpoint URL, model ID, and credential handling, read `references/endpoint-configuration.md`.
For MCQ, use `references/mcq-mode-guide.md` as the short agent checklist. Full mode tables and examples live in `docs/mcq-modes.md`.

**Output layout** for enabled stages is fixed under each sample's `out_dir` — you only need to choose `out_dir`.

---

### Step 4: Generate the CLI Command

**Single video (inner pipeline command):**
```bash
uv run python modules/cli.py --config configs/pipeline_example.yaml \
  data.0.inputs.video_path="<VIDEO_PATH>" \
  data.0.output.out_dir="<OUT_DIR>" \
  [endpoints.vlm.url="<VLM_URL>" endpoints.vlm.model="<VLM_MODEL>"] \
  [endpoints.llm.url="<LLM_URL>" endpoints.llm.model="<LLM_MODEL>"]
```

**Batch ≤9 videos** — expand `data.N` inline:
```bash
uv run python modules/cli.py --config configs/pipeline_example.yaml \
  data.0.inputs.video_path="clip1.mp4" data.0.output.out_dir="output/clip1" \
  data.1.inputs.video_path="clip2.mp4" data.1.output.out_dir="output/clip2"
```

**Batch 10+ videos** — generate a bash loop (single CLI invocation is more efficient than N separate calls):
```bash
VIDEOS=(/data/clips/clip_*.mp4)
ARGS=""
for i in "${!VIDEOS[@]}"; do
  NAME=$(basename "${VIDEOS[$i]}" .mp4)
  ARGS+=" data.${i}.inputs.video_path=\"${VIDEOS[$i]}\""
  ARGS+=" data.${i}.output.out_dir=\"output/batch/${NAME}\""
done
eval "uv run python modules/cli.py --config configs/pipeline_example.yaml $ARGS \
  endpoints.vlm.url=\"<VLM_URL>\" endpoints.vlm.model=\"<VLM_MODEL>\" \
  endpoints.llm.url=\"<LLM_URL>\" endpoints.llm.model=\"<LLM_MODEL>\""
```

**Mode-specific notes:**

- **`metadata-llm`** — LLM only; `vlm_verify_enabled` is auto-disabled by the schema. Always set:
  ```bash
  endpoints.llm.url="<LLM_URL>" endpoints.llm.model="<LLM_MODEL>"
  data.0.inputs.video_path="<existing/source_media.mp4>"
  data.0.inputs.metadata_json_path="<path/to/metadata.json>"
  ```

**Useful inline overrides:**
- `super_resolution.enabled=false` — skip SR stage
- `data.0.inputs.vlm_video_path="/path/to/video.mp4"` — force VLM JSON (and MCQ) to use a specific media file, bypassing the default priority chain (`sidecars/<track_stem>_tracking_red_id.<ext>` → SR output → original)
- `vlm_json.split_json_calls=true` — default, more reliable VLM JSON generation
- `pipeline.empty_output_policy=warn|fail` — use `warn` for best-effort batch runs; use `fail` when stage failures should stop the run
- `mcq_generation.mode=window-vlm-llm` — switch MCQ mode
- `mcq_generation.window_metadata_extraction.question_bank_file="/path/to/bank.json"` — custom question bank (question-driven-vlm-llm only; window/metadata modes use embedded bank in `mcq_prompt_file`)
- `mcq_generation.window_metadata_extraction.skip_existing=true` — resume a partial run
- `pipeline.gpu_ids=2,3` — pin specific GPU IDs (important when VLM/LLM servers occupy other GPUs)
- `pipeline.use_multi_gpu=true` — SR uses all GPUs in `gpu_ids`; default is single-GPU (first GPU only)

For windowing strategies, partial pipelines, and stage chaining see `references/config-decision-tree.md`.
For Docker execution patterns, including generic NGC-image `docker run` and repo-local `./docker/deploy.sh`, see `references/docker-run.md`.

Always output a complete, copy-pasteable command. Do not leave placeholders unfilled if the user gave you the values.

---

### Step 5: Pre-flight Checklist

Run all checks via Bash in parallel and report pass/fail before asking for execution approval:

```text
[ ] Docker running            →  docker info
[ ] Container image available →  docker image inspect <AUTO_LABELING_IMAGE>
[ ] Checkpoints present       →  ls ckpts/   (only if SR is enabled)
[ ] Input video exists        →  ls <video_path>
[ ] VLM endpoint reachable    →  curl -s <vlm_url>/models   (if VLM needed; add auth only after credential approval)
[ ] LLM endpoint reachable    →  curl -s <llm_url>/models   (if LLM needed; add auth only after credential approval)
```

**Important:** When checking VLM/LLM endpoints, read the actual model ID from the `/models` response and use it in the CLI command — the user-provided model name may differ from the served model ID (e.g. FP8 suffix). If an endpoint needs authentication, warn the user before accessing credential environment variables, do not print secret values, and prefer a mounted credential file or Docker secret over shell interpolation. See `references/endpoint-configuration.md`.

Provide the exact fix command for any hard failure (block execution until resolved):

| Failure | Fix |
|---------|-----|
| Docker not running | `sudo systemctl start docker` (or open Docker Desktop) |
| Image unavailable | Pull/build the NGC image and set `AUTO_LABELING_IMAGE`; if working inside this repo, `./docker/deploy.sh build` is also supported |
| Endpoint unreachable | Ask the user for a reachable VLM/LLM base URL (must end in `/v1`) and the served model ID, or have them start their own OpenAI-compatible server, then retry |

**Checkpoints missing** — do NOT block on this. Stage initialization will ensure required SeedVR2 checkpoints on first run; warn that the first run may spend time downloading. If SR is disabled (`super_resolution.enabled=false`), SR checkpoints are irrelevant.

---

### Step 6: Execute the Pipeline

After a passing pre-flight (no hard failures), present an execution summary and require explicit user approval before any Docker command. Include:

- Full Docker command to be run, with secrets redacted or passed through approved secret files/env names only.
- Expected resource use: GPU count, likely runtime, large image pulls/downloads, and output directory writes.
- Data access: input paths, mounted volumes, remote endpoints, and credential source names.

Do not execute Docker until the user approves that exact command.

Run through Docker, not host Python, for real pipeline execution. Do not set up or execute host-side pipeline runs for users; that path is not validated or endorsed. Inside the container, use `uv run python modules/cli.py ...` so the command uses the uv-managed runtime environment baked into the image. Use `references/docker-run.md` for the generic NGC-image `docker run` template; use `./docker/deploy.sh shell -lc "cmd"` only as a repo-local convenience when this checkout is available. Set a generous timeout because SR and VLM stages can run for many minutes.

**After execution completes:**
- Report success or the error message
- List the output files produced: `ls -lhR <out_dir>/`
- If MCQ was generated, show a preview from the first available task file among `task/mcq.json`, `task/bcq.json`, and `task/open_qa.json`

**If execution fails:**
- Show the last 50 lines of output to diagnose
- Common fixes:
  - OOM on SR → with default `pipeline.empty_output_policy=warn` the pipeline continues automatically (SR is skipped, tracking uses the original input video). Only add `super_resolution.enabled=false` if you want to skip SR entirely for speed.
  - Endpoint auth error → ask the user to verify the intended credential source is set; do not echo secret values
  - Permission denied on output dir → output dir may need to exist: `mkdir -p <out_dir>`

---

## Output Layout

Outputs are fixed under each sample's `out_dir`: `raw/`, `contextual/`, `task/`, `prompts/`, `sidecars/`, logs, and the saved effective config. The root `README.md` has the full tree. After a run, list `ls -lhR <out_dir>/`; if MCQ was generated, preview the first available task file among `task/mcq.json`, `task/bcq.json`, and `task/open_qa.json`.

Validate with `tao-daft validate metropolis-v3.0 --path <out_dir> --raw auto [--strict]` when `nvidia-tao-daft` is installed.

## Key Facts

- CLI entry point inside Docker: `uv run python modules/cli.py --config <yaml> [key=value ...]`
- Dotlist overrides use OmegaConf syntax (`key.subkey=value`). Quote paths that contain spaces.
- `data` is a list: single video uses `data.0.*`, second video uses `data.1.*`, etc. The loader auto-extends the list when you reference indices beyond `data[0]` — no need to pre-populate the base YAML.
- The pipeline supports **image inputs** (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`) as well as videos. For image inputs, SR/tracking sidecar outputs use `.png` extensions.
- **Supported video formats**: `.mp4` `.mov` `.m4v` (the LGPL-only FFmpeg in this image ships only the mov demuxer, which handles all mp4-family containers; matroska / avi / webm support was removed to keep the codec set royalty-clean).
- Video SR (SeedVR2) requires Docker. Use `references/docker-run.md`; repo-local users may use `./docker/deploy.sh shell`.
- Remote paths (`s3://`, `msc://`) work for inputs and `out_dir`; see `docs/remote-io.md` for setup patterns.
- If the user needs `events.json` but doesn't have one yet, enable only the VLM JSON stage (see `configs/pipeline_example.yaml` comments).
- For MCQ-only from existing captions, use `mcq_generation.mode=metadata-llm` and provide an existing `data.0.inputs.video_path`, `data.0.inputs.metadata_json_path`, and an LLM endpoint.
- **Endpoint credentials** — VLM and LLM credential handling is documented in `references/endpoint-configuration.md`; do not print secret values.
- **GPU selection**: use `pipeline.gpu_ids=<ids>` (comma-separated, or `"all"`) to pin specific GPUs (e.g. `pipeline.gpu_ids=2,3`). Important when other workloads (VLM/LLM servers) already occupy some GPUs. Tracking always uses the first GPU in `gpu_ids`; VLM JSON and MCQ stages call remote endpoints and do not use local GPUs. SR defaults to single-GPU (first GPU); set `pipeline.use_multi_gpu=true` to use all GPUs in `gpu_ids` for SR (`sp_size` is set automatically). Plan a dedicated pipeline GPU for SeedVR2 SR; the 7B variant requires more free GPU memory than the 3B variant. In the current local Qwen recipe, reserve one additional dedicated GPU per VLM/LLM endpoint and keep those endpoint GPUs out of `pipeline.gpu_ids` unless you have verified enough free memory.
- **Stage extensibility**: each stage has a `model` field selecting the implementation (`super_resolution.model="seedvr2"`, `detection_and_tracking.model="rfdetr"`, `vlm_json.model="vlm"`, `mcq_generation.mode=...`). All default to the current impl — only set explicitly when plugging in a different model. New implementations subclass `BaseSuperResolver` / `BaseTracker` / `BaseVlmJsonGenerator` / `BaseMCQGenerator` and register in the corresponding `factory.py`.

## Reference Files

- `references/config-decision-tree.md` — edge cases: partial pipelines, stage chaining, `metadata-llm` path
- `references/docker-run.md` — generic NGC-image Docker run template plus repo-local `deploy.sh` alternative
- `references/endpoint-configuration.md` — endpoint URL, model ID, and credential handling rules
- `references/mcq-mode-guide.md` — short MCQ mode checklist for agents; full examples live in `docs/mcq-modes.md`
- `references/custom-caption.md` — custom-question captioning (closed-choice MCQ/BCQ or free-form) and reusable question-bank creation
- `references/person-attribute-caption.md` — shipped PAS / person-attribute caption preset
