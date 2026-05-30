<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# MCQ generation module

This module contains:

- **Python library**: `mcq/`
  - `mcq/runners/`: mode implementations used by the unified CLI.
  - `mcq/utils/`: convenience import layer for common helpers (see below).
- **Use-case cookbooks**: root-level `cookbooks/<use_case>/` directories with question banks, configs, and domain-specific prompts colocated by domain.
- **Shared prompts**: root-level `cookbooks/shared/` for cross-use-case VLM JSON prompts, VLM verify prompts, and generic question-driven templates.

## Mode mapping (unified CLI)

For config/endpoints/auth usage, see the repo root `README.md`.

Set `mcq_generation.mode` to one of:

- **`window-vlm-llm`**: frames → VLM caption → LLM MCQ (`mcq_generation.mcq.runners.window_vlm_llm`)
- **`window-direct-vlm`**: frames → VLM direct MCQ (`mcq_generation.mcq.runners.window_direct_vlm`)
- **`question-driven-vlm-llm`**: question bank → LLM generates prompts (`mcq_generation.question_driven_vlm_llm`) → window inference (`mcq_generation.mcq.runners.window_vlm_llm`)
- **`metadata-llm`**: `metadata.json` captions → LLM MCQ (`mcq_generation.mcq.runners.metadata_llm`)

Prompt defaults follow cookbook ownership. For the user-facing mode/use-case map, see `../../docs/mcq-modes.md`.

### Related tools/stages

| Stage / purpose | Script |
|---|---|
| **VLM JSON stage** (separate module; video: `events.json` + `video.json`; image: `image.json`) | `modules/vlm_json/vlm_json_generator.py` (library: `modules/vlm_json/runners/video_pipeline.py`) |

## Question-driven window VLM+LLM (`question-driven-vlm-llm`)

This mode is for **coverage-driven MCQ generation**: instead of a fixed VLM prompt, you provide a **question bank**,
and an LLM generates:

- a **general VLM "scene evidence" prompt** that asks for structured information needed to answer the bank, and
- (optional) a **mapper rules appendix** to help the LLM map the structured scene description to bank answers.

Then the pipeline runs the normal **window VLM→LLM** pipeline using the generated prompts.

### Two-stage execution

This mode runs as **two separate pipeline stages**:

1. **Stage 1 — Prompt generation** (runs *before* SR/tracking, LLM-only, no video):
   - question bank → LLM → `prompts/scene_prompt.used.md` + `prompts/mcq_prompt.used.md`
   - if VLM verify is enabled: verify prompt template → `prompts/vlm_verify_prompt.used.md`
2. **Stage 2 — Window inference** (runs *after* tracking, deferred):
   - video → windows → frames → VLM caption → LLM MCQ (using the generated prompts)

### How to run (unified CLI)

Use the unified blueprint config and select the mode with `mcq_generation.mode=question-driven-vlm-llm`. Full user-facing commands and cookbook choices live in `../../docs/mcq-modes.md`; the sections below document the implementation-specific artifacts, merge behavior, and bank schema.

### Outputs & artifacts

Under `data[*].output.out_dir/` (preset defaults shown):

- `task/mcq.json`, `task/bcq.json`, and/or `task/open_qa.json` (final per-scene DAFT task files, written only when at least one qualifying item exists; see [Question bank format](#question-bank-format) for export rules)
- `sidecars/metadata.json` (per-window captions and intermediate fields)
- `sidecars/mcq.vlm_verify.json` (optional; VLM verification verdicts/reasoning per question)
- `sidecars/mcq.empty.json` (optional resume marker when task composition completed with zero task items; resume accepts it only with `mcq: []` and `_error: "zero_task_items"`)
- `sidecars/_work/window_vlm_llm/`, `sidecars/_work/window_direct_vlm/`, or `sidecars/_work/question_driven_vlm_llm/` (per-window frame extraction scratch; frames auto-cleaned on success)
- `prompts/` (prompt artifacts preserved for reproducibility; see `../../docs/mcq-modes.md#prompt-artifacts` for the per-mode matrix)

### How window answers are merged (aggregation defaults)

Window modes answer questions **per window**, then merge into one per-scene set of `task/mcq.json`, `task/bcq.json`, and/or `task/open_qa.json` files.

If `mcq_generation.window_metadata_extraction.vlm_verify_enabled=true` (schema default `false`;
`configs/pipeline_example.yaml` enables it by default), the runner performs one extra VLM verify pass
after each window's answer is finalized. Supported by `window-vlm-llm`, `window-direct-vlm`, and
`question-driven-vlm-llm`. Stores:
- per-window traces in `sidecars/metadata.json` (`windows[*].vlm_verify`)
- sidecar `sidecars/mcq.vlm_verify.json` with a `summary` (`windows_total`, `windows_with_verify`, `questions_verified`, `questions_corrected`)

When using `metadata-llm`:
- It is LLM-only for this mode; the schema auto-disables `vlm_verify_enabled`.
- No VLM endpoint override or manual `vlm_verify_enabled=false` override is needed.
- The low-level runner still has optional VLM verify support for direct use; if enabled there, it samples frames from the source video and writes the same verify sidecar shape.

Defaults (for custom/question-driven banks):
- **Yes/No questions** (options exactly `{Yes, No}`): `supermajority` (conservative for positives)
- **Multi-choice questions**: `majority_tie_first`
If you want different behavior, set per-question `aggregation` in the question bank (see below). No
domain-specific merge rules are hard-coded in the runners.

### Question bank format

Built-in question banks live inside use-case cookbooks (for example, `cookbooks/traffic/question_bank.json`); `cookbooks/person_attributes/question_bank.json` is a free-form/open-QA PAS preset that pairs with `cookbooks/person_attributes/prompts/mcq/question_driven_vlm_llm/vlm_scene_prompt_template.md`. Custom banks can live anywhere and are plain JSON:

```json
{
  "name": "my_bank",
  "questions": [
    {
      "id": "1_1",
      "question": "Is there a traffic accident taking place?",
      "options": ["Yes", "No"],
      "aggregation": "any"
    },
    {
      "id": "4_9",
      "question": "What is the type of traffic violation?",
      "options": ["A. Red light violation", "B. Illegal turn", "C. Other violation"],
      "include_if": { "4_1": "Yes" }
    },
    {
      "id": "9_1",
      "question": "Describe the road layout and traffic pattern.",
      "options": [],
      "aggregation": "majority"
    }
  ]
}
```

Notes:
- `id` and `question` are required per question.
- Missing `options` or `options: []` means an open-ended/free-form question. A non-empty answer is exported to DAFT `task/open_qa.json`; closed-choice questions are exported to `task/mcq.json` or `task/bcq.json`.
- Duplicate closed-choice options are removed when the question bank is loaded, before prompts are generated. For MCQ, duplicates are compared after stripping author-supplied letter prefixes such as `A. ` or `A)`.
- BCQ export is intentionally strict: only bare unique `["Yes", "No"]` or `["No", "Yes"]` options with a bare `Yes` or `No` answer become `task/bcq.json`. Prefixed values such as `["A. Yes", "B. No"]`, or extra choices such as `["No", "Yes", "unknown"]`, remain `task/mcq.json`.
- `aggregation` is optional. If set, it controls how per-window answers are merged into a per-video answer.
  - Allowed values: `majority`, `majority_tie_first`, `first`, `any`, `supermajority`
  - Defaults (if omitted):
    - Yes/No questions: `supermajority` (conservative)
    - Other multiple-choice: `majority_tie_first`
- `include_if` is optional; it is a mapping from **other question ids** → **required answer string**.
  The window MCQ runner uses it to conditionally include questions when aggregating.

### Config knobs (where they live)

All QD-specific knobs are under `mcq_generation.window_metadata_extraction.*`:

For window/direct/metadata modes, leave `scene_prompt_file` / `mcq_prompt_file` as `null` to use the shipped prompt under that mode's directory. Override these only when you want a custom prompt.

- **Question bank**:
  - `question_bank_file` (blueprint value: `cookbooks/traffic/question_bank.json`)
- **Prompt generation (LLM used to write prompts)**:
  - `prompt_gen_llm_base_url` / `prompt_gen_llm_model` (default to `endpoints.llm.*` when unset)
  - `prompt_gen_llm_max_tokens` (default: 8192)
  - `prompt_gen_seed` (default: `null`; controls LLM sampling seed for reproducibility; `null` = no seed)
  - Retries: inherited from `endpoints.llm.retries` (no separate knob)
  - `qd_vlm_scene_prompt_template_file` (QD template file for generating the VLM scene prompt; shipped under `cookbooks/shared/prompts/mcq/question_driven_vlm_llm/templates/`)
  - `qd_mcq_mapper_prompt_template_file` (QD template file for mapping caption -> MCQ, with bank injection; shipped under `cookbooks/shared/prompts/mcq/question_driven_vlm_llm/templates/`)
  - `append_mapper_rules` (if true, generates an extra rules appendix and appends it to the mapper prompt)
- **Window inference** (same as `window-vlm-llm`):
  - `window_frames` (frame-count-based), `window_seconds` (time-based; alternative to `window_frames`), `single_window`, `sampling_fps`, `resolution`, `max_frames`
  - `vlm_max_tokens`, `llm_max_tokens`, `vlm_temperature`, `llm_temperature`, `timeout`, `rate_limit`
  - `aggregate_windows`, `retry_missing_questions`, `retry_missing_max_rounds`
  - `vlm_verify_enabled` (optional post-check pass)
  - `vlm_verify_prompt_file` (optional verify prompt override; leave `null` to use `cookbooks/shared/prompts/mcq/vlm_verify/verify_prompt.md`)
  - `vlm_verify_apply_corrections` (default `false`; set `true` only when verify may propose answer corrections)
  - `skip_existing` (resume behavior)

## Import paths

- **Pipeline config keys** stay under `mcq_generation.*` (this is the config schema name).
- **Python imports** use the package name `mcq_generation.mcq`:

```python
from mcq_generation.mcq.runners.window_vlm_llm import WindowVlmLlmRunner
from mcq_generation.mcq.utils.openai import call_chat_raw
```

- **Shared I/O helpers** (`read_text`, `write_json`, etc.) live under `al_utils/`:

```python
from al_utils.io import read_text, write_json, write_text, sha256_text
```

- **Prompt file loading** (config-relative path resolution) is in:

```python
from mcq_generation.mcq.utils.prompt_io import load_text, resolve_path
```
