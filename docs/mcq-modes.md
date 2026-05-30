<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# MCQ Generation Modes

Four modes are available via `mcq_generation.mode=<mode>`. Run the examples from inside Docker (`./docker/deploy.sh shell -lc '...'` or a published image container), not directly on the host. Use `host.docker.internal` for VLM/LLM servers on the host.
Default prompts and question banks live under root-level `cookbooks/` so they can be found, copied, and shared independently from implementation modules.

Use this page after a first pipeline run when choosing a mode, inspecting shipped prompts, or adding a domain/question bank. Setup: [Getting started](getting-started.md). Keys and verify flags: [Config reference](config-reference.md#mcq-verification).

---

1. **Use-case cookbooks** (`cookbooks/traffic/`, `cookbooks/person_attributes/`, `cookbooks/robotics/`, `cookbooks/warehouse/`) own domain question banks, configs, and any domain-specific prompts.
2. **Shared cookbook assets** (`cookbooks/shared/`) hold prompt logic that is cross-use-case, such as VLM JSON prompts, VLM verification, and generic question-driven templates.
3. **Mode defaults** follow that ownership: shipped `window-*` / `metadata-llm` defaults are traffic blueprint prompts because they embed traffic questions, while `question-driven-vlm-llm` uses shared templates with whichever `question_bank_file` you provide.

## Mode summary

| Mode | Use when | Endpoints | Main prompt/question source |
|------|----------|-----------|-----------------------------|
| `question-driven-vlm-llm` *(default)* | New domains or reusable question banks. The bank drives both evidence extraction and final QA mapping. | VLM + LLM | `question_bank_file` plus shared QD templates |
| `window-vlm-llm` | You want a fixed VLM scene-caption prompt, then LLM mapping to task QA. Good for tuned domain workflows. | VLM + LLM | `scene_prompt_file` + `mcq_prompt_file` |
| `window-direct-vlm` | You want the VLM to answer task questions directly without an LLM mapper. Simpler stack, more model-dependent formatting. | VLM only | `mcq_prompt_file` |
| `metadata-llm` | You already have `sidecars/metadata.json` captions and want to regenerate task QA without re-running VLM/frame extraction. | LLM only | `metadata_json_path` + `mcq_prompt_file` |

## Shipped assets

Domain cookbooks:

- **Traffic** (`cookbooks/traffic/`): blueprint default domain. Includes `question_bank.json` plus fixed prompts for `window-vlm-llm`, `window-direct-vlm`, and `metadata-llm`.
- **Robotics** (`cookbooks/robotics/`): `question_bank.json` for robot manipulation with `question-driven-vlm-llm`.
- **Warehouse** (`cookbooks/warehouse/`): `question_bank.json` for warehouse, shelving, and floor-safety checks with `question-driven-vlm-llm`.
- **Person attributes** (`cookbooks/person_attributes/`): open-QA `question_bank.json` plus a domain VLM scene template for visible clothing/person descriptions.

Shared MCQ templates:

- `cookbooks/shared/prompts/mcq/question_driven_vlm_llm/templates/vlm_scene_prompt_template.md`
- `cookbooks/shared/prompts/mcq/question_driven_vlm_llm/templates/mcq_mapper_bank_injected_template.md`
- `cookbooks/shared/prompts/mcq/question_driven_vlm_llm/templates/mcq_mapper_rules_appendix_template.md`
- `cookbooks/shared/prompts/mcq/vlm_verify/verify_prompt.md`

VLM JSON prompts:

- `cookbooks/shared/prompts/vlm_json/video_json_prompt.md`
- `cookbooks/shared/prompts/vlm_json/video_events_prompt.md`
- `cookbooks/shared/prompts/vlm_json/image_caption_prompt.md`

VLM JSON prompts are separate from MCQ generation prompts.

---

## Adding prompts or question banks

Prefer `question-driven-vlm-llm` for a new domain. Create `cookbooks/<slug>/question_bank.json` by copying an existing bank, then run with:

```bash
mcq_generation.mode=question-driven-vlm-llm \
mcq_generation.window_metadata_extraction.question_bank_file="cookbooks/<slug>/question_bank.json"
```

The LLM generates the VLM scene prompt and MCQ mapper from the bank. To customize those generation templates, copy the shared templates and override:

```bash
mcq_generation.window_metadata_extraction.qd_vlm_scene_prompt_template_file="cookbooks/<slug>/prompts/mcq/question_driven_vlm_llm/templates/vlm_scene_prompt_template.md" \
mcq_generation.window_metadata_extraction.qd_mcq_mapper_prompt_template_file="cookbooks/<slug>/prompts/mcq/question_driven_vlm_llm/templates/mcq_mapper_bank_injected_template.md"
```

For fixed window modes (`window-vlm-llm`, `window-direct-vlm`, `metadata-llm`), copy the mode prompt file and override `scene_prompt_file` and/or `mcq_prompt_file`. In these modes, the question list is embedded in `mcq_prompt_file`, so update the prompt instructions and question list together.

VLM JSON prompts are separate from MCQ prompts. Override them with `vlm_json.scene_prompt_file` and, for video events, `vlm_json.events_prompt_file`.

---

## A) `window-direct-vlm` — VLM outputs MCQ directly (no LLM step)

```bash
uv run python modules/cli.py --config configs/pipeline_example.yaml \
  super_resolution.enabled=false detection_and_tracking.enabled=false vlm_json.enabled=false \
  mcq_generation.enabled=true mcq_generation.mode=window-direct-vlm \
  data.0.inputs.video_path="/path/to/video.mp4" \
  data.0.output.out_dir="/path/to/output" \
  endpoints.vlm.url="<VLM_BASE_URL>" endpoints.vlm.model="<VLM_MODEL_ID>" \
  endpoints.llm.url="" endpoints.llm.model=""
```

Default traffic prompt: `cookbooks/traffic/prompts/mcq/window_direct_vlm/mcq_prompt.md`

---

## B) `window-vlm-llm` — VLM caption → LLM MCQ

```bash
uv run python modules/cli.py --config configs/pipeline_example.yaml \
  super_resolution.enabled=false detection_and_tracking.enabled=false vlm_json.enabled=false \
  mcq_generation.enabled=true mcq_generation.mode=window-vlm-llm \
  data.0.inputs.video_path="/path/to/video.mp4" \
  data.0.output.out_dir="/path/to/output" \
  endpoints.vlm.url="<VLM_BASE_URL>" endpoints.vlm.model="<VLM_MODEL_ID>" \
  endpoints.llm.url="<LLM_BASE_URL>" endpoints.llm.model="<LLM_MODEL_ID>"
```

**Override traffic window prompts (optional):**
```bash
  mcq_generation.window_metadata_extraction.scene_prompt_file="cookbooks/traffic/prompts/mcq/window_vlm_llm/scene_prompt.md" \
  mcq_generation.window_metadata_extraction.mcq_prompt_file="cookbooks/traffic/prompts/mcq/window_vlm_llm/mcq_prompt.md"
```

> For `window-*` / `metadata-llm` modes the question bank is embedded directly inside `mcq_prompt_file` as a single unit — the file contains both the LLM instructions and the question list together. To customize, create a new prompt file containing both your updated instructions and your question list, then override `mcq_generation.window_metadata_extraction.mcq_prompt_file`.

---

## C) `question-driven-vlm-llm` — question bank → LLM prompt-gen → VLM → LLM MCQ

Runs in **two pipeline stages**: (1) LLM generates a VLM evidence-extraction prompt from your question bank (before SR/tracking); (2) the generated prompt drives the window VLM→LLM pipeline after tracking. The final prompts used by inference are saved under `out_dir/prompts/`.

Default generic templates live under `cookbooks/shared/prompts/mcq/question_driven_vlm_llm/templates/`.

```bash
uv run python modules/cli.py --config configs/pipeline_example.yaml \
  super_resolution.enabled=false detection_and_tracking.enabled=false vlm_json.enabled=false \
  mcq_generation.enabled=true mcq_generation.mode=question-driven-vlm-llm \
  data.0.inputs.video_path="/path/to/video.mp4" \
  data.0.output.out_dir="/path/to/output" \
  endpoints.vlm.url="<VLM_BASE_URL>" endpoints.vlm.model="<VLM_MODEL_ID>" \
  endpoints.llm.url="<LLM_BASE_URL>" endpoints.llm.model="<LLM_MODEL_ID>" \
  mcq_generation.window_metadata_extraction.question_bank_file="cookbooks/traffic/question_bank.json"
```

**Available cookbooks**:

| Path | Domain |
|------|--------|
| `cookbooks/traffic/question_bank.json` | Flattened traffic accident detection blueprint default |
| `cookbooks/robotics/question_bank.json` | Robot manipulation |
| `cookbooks/warehouse/question_bank.json` | Warehouse / shelving / floor safety |
| `cookbooks/person_attributes/question_bank.json` | Free-form person attribute captions; pair with `cookbooks/person_attributes/prompts/mcq/question_driven_vlm_llm/vlm_scene_prompt_template.md` |

For custom banks, see [Question bank format](../modules/mcq_generation/README.md#question-bank-format) for the schema, `open_qa` export, duplicate-option handling, BCQ rules, and aggregation options.

---

## D) `metadata-llm` — LLM MCQ from existing window captions

Use when you already have `metadata.json` (from a prior `window-vlm-llm`, `window-direct-vlm`, or `question-driven-vlm-llm` run) and want to regenerate MCQs without re-running VLM or frame extraction.

Default traffic prompt: `cookbooks/traffic/prompts/mcq/metadata_llm/mcq_prompt.md`

The unified pipeline still validates `data.0.inputs.video_path`; set it to an existing media file, usually the same source media that produced `metadata.json`.

```bash
uv run python modules/cli.py --config configs/pipeline_example.yaml \
  super_resolution.enabled=false detection_and_tracking.enabled=false vlm_json.enabled=false \
  mcq_generation.enabled=true mcq_generation.mode=metadata-llm \
  data.0.inputs.video_path="/path/to/source_video.mp4" \
  data.0.inputs.metadata_json_path="/path/to/sidecars/metadata.json" \
  data.0.output.out_dir="/path/to/output" \
  endpoints.llm.url="<LLM_BASE_URL>" endpoints.llm.model="<LLM_MODEL_ID>"
```

> `metadata-llm` is LLM-only. The schema auto-disables `vlm_verify_enabled` for this mode, so no VLM endpoint override is needed.

---

## Window / sampling knobs

These shared keys are used by window and metadata modes. Frame/VLM sampling rows apply to frame-extracting modes (`window-direct-vlm`, `window-vlm-llm`, `question-driven-vlm-llm`). In normal pipeline configs, `metadata-llm` reuses existing captions and does not re-extract frames, but it still records/inherits windowing metadata and the low-level runner uses sampling settings if optional VLM verify is enabled directly.

| Key | Default | Description |
|-----|---------|-------------|
| `mcq_generation.window_metadata_extraction.window_frames` | `60` ¹ | Frame-count-based windowing |
| `mcq_generation.window_metadata_extraction.window_seconds` | `4.0` | Time-based windowing (alternative to `window_frames`; used only when `window_frames=0`) |
| `mcq_generation.window_metadata_extraction.single_window` | `false` | Treat the whole media sample as one window |
| `mcq_generation.window_metadata_extraction.sampling_fps` | `2.0` | Frame sampling rate |
| `mcq_generation.window_metadata_extraction.resolution` | `480` | Input resolution |
| `mcq_generation.window_metadata_extraction.max_frames` | `100` | Max frames per window |
| `mcq_generation.window_metadata_extraction.skip_existing` | `false` | Resume a partial run |
| `mcq_generation.window_metadata_extraction.vlm_max_tokens` | `8192` | VLM output budget |
| `mcq_generation.window_metadata_extraction.llm_max_tokens` | `8192` | LLM output budget |
| `mcq_generation.window_metadata_extraction.vlm_temperature` | `0.0` | VLM sampling temperature |
| `mcq_generation.window_metadata_extraction.llm_temperature` | `0.0` | LLM sampling temperature |
| `mcq_generation.window_metadata_extraction.vlm_structured_output` | `"openai"` | JSON forcing mode for VLM: `auto`, `nim`, `openai`, `off` |
| `mcq_generation.window_metadata_extraction.llm_structured_output` | `"openai"` | JSON forcing mode for LLM: `auto`, `nim`, `openai`, `off` |
| `mcq_generation.window_metadata_extraction.rate_limit` | `0` | Min seconds between VLM/LLM calls (0 = no limit) |
| `mcq_generation.window_metadata_extraction.timeout` | `600` | API timeout (seconds) |
| `mcq_generation.window_metadata_extraction.aggregate_windows` | `true` ² | Combine multi-window outputs |

> ¹ Blueprint config default (`configs/pipeline_example.yaml`). Schema default is `0`. When not using the blueprint, set either `window_frames` > 0 or `window_seconds` > 0 — having both at `0` with `single_window=false` is a validation error. Use `single_window=true` to bypass windowing entirely.
>
> ² Blueprint config default. Schema default is `false`.

---

## Retry and VLM verify

**Retry missing questions** (`window-vlm-llm`, `window-direct-vlm`, `question-driven-vlm-llm`, `metadata-llm`):
- `retry_missing_questions` (default: `true`) — best-effort retry to fill required questions per window
- `retry_missing_max_rounds` (default: `2`) — max retry rounds per window

**VLM verify** (`window-vlm-llm`, `window-direct-vlm`, `question-driven-vlm-llm`):
- `vlm_verify_enabled` (schema default: `false`; **blueprint enables it by default**)
- Runs after per-window LLM finalization; requires `endpoints.vlm.*`
- `vlm_verify_prompt_file` (schema default: `null`) optionally overrides the generic verify prompt template. Leave unset to use `cookbooks/shared/prompts/mcq/vlm_verify/verify_prompt.md`.
- `vlm_verify_apply_corrections` (schema and blueprint default: `false`) — verification attaches reasoning traces only without changing answers; set `true` only to allow answer corrections under strict visual-evidence rules
- Writes sidecar files under the scene's `sidecars/` directory:
  - `sidecars/mcq.vlm_verify.json` — verdict + reasoning trace per question, plus a `summary` with `windows_total`, `windows_with_verify`, `questions_verified`, and `questions_corrected`
- Disable with `mcq_generation.window_metadata_extraction.vlm_verify_enabled=false`
- If MCQ generation completes task composition with zero task items, `sidecars/mcq.empty.json` may be written as a resume marker; it is not a DAFT task file. Resume accepts it only when it contains `mcq: []` and `_error: "zero_task_items"`.

`metadata-llm` is LLM-only in the CLI pipeline. Even though the blueprint enables VLM verify for other window modes, the schema auto-disables `vlm_verify_enabled` when `mcq_generation.mode=metadata-llm`; only direct/library use of the lower-level runner can enable metadata VLM verify, in which case it samples frames from the source video.

---

## Prompt artifacts

Window modes write prompt files under `out_dir/prompts/` for reproducibility. Exact files per mode:

| Mode | `scene_prompt.used.md` | `mcq_prompt.used.md` | `vlm_verify_prompt.used.md` when verify is enabled | `prompts.used.json` |
|------|------------------------|----------------------|------------------------------|---------------------|
| `window-direct-vlm` | — | ✓ | ✓ | ✓ |
| `window-vlm-llm` | ✓ | ✓ | ✓ | ✓ |
| `question-driven-vlm-llm` | ✓ (LLM-generated) | ✓ | ✓ | ✓ |
| `metadata-llm` | — | ✓ | — | ✓ |

For the three VLM verify-capable window modes, `vlm_verify_prompt.used.md` is the policy-expanded verify prompt skeleton; per-window MCQ answers are injected only at VLM call time. `prompts.used.json` includes `vlm_verify_prompt_sha256` alongside the other prompt hashes only when `vlm_verify_enabled=true`. Pipeline `metadata-llm` is LLM-only and persists only the MCQ prompt hash.

`question-driven-vlm-llm` also writes non-duplicate prompt-generation provenance: `scene_prompt.generated_by_llm.meta.json`, plus mapper rules and the bank-injected mapper prompt only when they differ from the final `mcq_prompt.used.md`.

Frame-extracting modes (`window-direct-vlm`, `window-vlm-llm`, `question-driven-vlm-llm`) use underscore-named scratch dirs such as `sidecars/_work/window_direct_vlm/`, `sidecars/_work/window_vlm_llm/`, and `sidecars/_work/question_driven_vlm_llm/` for per-window frame extraction and clean those frames on success. Pipeline `metadata-llm` reads existing captions from `metadata_json_path` and does not extract frames unless optional runner-level VLM verify is enabled directly.
