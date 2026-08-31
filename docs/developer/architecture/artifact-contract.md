# Artifact Contract

Status: canonical for Auto-Labeling v1.1.0

Audience: Auto-Labeling engineers, pipeline owners, and reviewers who
need one definitive reference for what a run writes, when it reuses versus
regenerates, which stage owns which file, and how a partially-successful run is
represented on disk.

This document is the companion to
[Experiment Output Layout](../../user-guide/experiment-output-layout.md). That doc defines the
directory skeleton; this doc defines the *contract* over the files inside it.

## Scope

The contract covers the DAFT scene directory produced by
`super_resolution`, `detection_and_tracking`, `referring_expressions`,
`captioning`, `grounding_2d`, `visual_qa`, `reasoning`,
`person_attribute_search` (PAS), and `training_export`. It does not cover
model-endpoint wiring; see
[Model Client Architecture](./model-client-architecture.md).

## 1. Outputs Produced

### 1.1 DAFT scene layout

Every stage reads and writes inside one DAFT scene directory
(`DataEntry.data_path`). The shared scene skeleton and media handoff are defined
in `packages/core/src/core/scene.py`; reasoning-specific paths are centralized
in `packages/tasks/reasoning/src/reasoning/paths.py`.

```text
<scene>/
  raw/<media_id>.<ext>          # analyzed media (symlink for local inputs)
  contextual/                   # scene-level metadata (DAFT contextual)
  task/                         # question/answer + description tasks (DAFT task)
  sidecars/                     # non-DAFT diagnostics + pipeline_state.json
```

DAFT files are validated against the DAFT v3 type set on write
(`core.formats.daft.write_daft_json`). Sidecars are free-form JSON that the DAFT
type set cannot host (for example PAS v3 attributes or the anomaly verdict).

### 1.2 Outputs by stage

The table lists each stage's owned outputs, the on-disk path (relative to the
scene root), and the output kind. `DAFT contextual` and `DAFT task` are
validated DAFT files; `non-DAFT` entries are sidecars outside the DAFT type
set.

| Stage | Output | Path | Kind |
|---|---|---|---|
| super_resolution | enhanced media | `sidecars/sr_output.<ext>` | non-DAFT media |
| detection_and_tracking | tracked-object catalogue | `contextual/instances.json` | DAFT contextual |
| detection_and_tracking | per-frame 2D detections | `contextual/objects.json` | DAFT contextual |
| detection_and_tracking | 4D MOT (opt-in) | `contextual/tracking.json` | DAFT contextual |
| detection_and_tracking | analyzed media | `raw/<media_id>.<ext>` | media |
| detection_and_tracking | PAS track seam (opt-in) | `sidecars/detection_and_tracking/tracks.json` | non-DAFT |
| detection_and_tracking | PAS track crops (opt-in) | `sidecars/tracks/crops/` | non-DAFT media |
| referring_expressions | region phrases | `sidecars/referring_expressions/referring_expressions.json` | non-DAFT |
| referring_expressions | phrase diagnostics | `sidecars/referring_expressions/` | non-DAFT |
| captioning | scene description | `task/scene_description.json` | DAFT task |
| captioning | whole-video summary | `task/video_summarization.json` | DAFT task |
| captioning | per-window dense captions | `task/temporal_description.json` | DAFT task |
| captioning | dense caption windows | `contextual/chunks.json` | DAFT contextual |
| grounding_2d | grounded expressions | `sidecars/grounding_2d/grounding_2d.json` | non-DAFT |
| grounding_2d | extraction and grounding diagnostics | `sidecars/grounding_2d/` | non-DAFT |
| visual_qa | multi-choice questions | `task/mcq.json` | DAFT task |
| visual_qa | binary (Yes/No) questions | `task/bcq.json` | DAFT task |
| visual_qa | open-ended questions | `task/open_qa.json` | DAFT task |
| visual_qa | window/item diagnostics | `sidecars/visual_qa/` | non-DAFT |
| reasoning | temporal events (video) | `contextual/events.json` | DAFT contextual |
| reasoning | event characterization (opt-in) | `contextual/msted.json` | DAFT contextual |
| reasoning | query-driven event grounding | `task/temporal_localization.json` | DAFT task |
| reasoning | MCQ with explanations | `task/mcq_openended.json` | DAFT task |
| reasoning | Yes/No with explanations | `task/bcq_openended.json` | DAFT task |
| reasoning | open-ended QA (opt-in) | `task/open_qa.json` | DAFT task |
| reasoning | causal grounding (opt-in) | `task/causal_linkage.json` | DAFT task |
| reasoning | anomaly verdict (opt-in) | `sidecars/reasoning/anomaly.json` | non-DAFT |
| person_attribute_search | structured v3 attributes | `sidecars/person_attribute_search/attributes.json` | non-DAFT |
| person_attribute_search | tiered retrieval queries | `sidecars/person_attribute_search/queries.json` | non-DAFT |
| person_attribute_search | HITL preannotation | `sidecars/person_attribute_search/hitl.json` | non-DAFT |
| person_attribute_search | per-chunk PAS (video seam) | `sidecars/person_attribute_search/pas.json` | non-DAFT |
| person_attribute_search | per-chunk queries (video seam) | `sidecars/person_attribute_search/chunk_queries.json` | non-DAFT |
| person_attribute_search | contextual PAS mirror (opt-in) | `contextual/person_attributes.json` | DAFT contextual |
| person_attribute_search | merged anomaly record (opt-in) | `sidecars/person_attribute_search/pas_anomaly.json` | non-DAFT |
| training_export | TAO-VL-Reason export | export target (outside scene) | export format |
| training_export | Cosmos-Reason export | export target (outside scene) | export format |

Notes:

- After a media-transforming task succeeds, the shared pipeline interface
  promotes its output to `sidecars/active.<ext>`. That handoff is owned by
  `core`, not by the super-resolution task.
- `sidecars/referring_expressions/referring_expressions.json` and
  `sidecars/grounding_2d/grounding_2d.json` are stage-specific JSON. They are
  not registered DAFT v3 task types and must not be written under `task/`.
- The reasoning source lists its owned outputs in
  `ReasoningTask._owned_outputs`
  (`packages/tasks/reasoning/src/reasoning/task.py`).
- Closed-form `mcq.json` / `bcq.json` / `open_qa.json` are produced by
  `visual_qa`, which routes each generated item into exactly one of the three
  via `core.formats.daft.converters.tasks.to_daft_tasks`. The `*_openended`
  variants under `task/` are produced by `reasoning` and are distinct files.
- Composite EPAS runs use separate diagnostic namespaces such as
  `sidecars/visual_qa_anomaly/` and `sidecars/visual_qa_per_track/`; the
  standalone Visual QA service owns `sidecars/visual_qa/`.
- Reasoning traces inside `mcq.json` / `bcq.json` come from the `visual_qa`
  stage running with `--include-reasoning`, not from a separate reasoning pass.

## 2. Overwrite / Reuse Behavior

### 2.1 Cookbook selection versus in-stage skip

Reuse across a product run is expressed by which cookbook nodes you enable.
Visual Attribute Search (`person_attribute_search`) is assembly-only: it does
not wrap captioning or Visual QA. Those producers are separate services and
separate cookbook nodes. The PAS service has no `--overwrite` flag. When the
`person_attribute_search` node runs, it writes its owned sidecars.

A few stages skip work when their own deliverable already exists. For example
`grounding_2d` reuses `sidecars/grounding_2d/grounding_2d.json` unless
`force_reprocess` is set. To regenerate a producer, re-run that producer node
rather than expecting PAS to own it.

### 2.2 File-level write behavior

When a stage does run, its writes are last-writer-wins, not merges:

- DAFT files are written atomically with `overwrite=True` by default
  (`write_daft_json`): a temp file is written and renamed into place, so a
  partial file is never visible, and any prior file at that path is replaced.
- The reasoning LLM emitters have **no** built-in reuse gate. Every enabled
  emitter re-renders and overwrites its owned file on each run
  (`emit_stage`). To preserve a prior reasoning run, point the run at a fresh
  output directory rather than relying on reuse.
- `pipeline_state.json` is the exception: it is merged, not overwritten (see
  section 4.3).

## 3. Reasoning Write Scope

This section defines exactly what the `reasoning` stage may create or overwrite
versus what it must treat as read-only upstream evidence.

### 3.1 Reasoning may create or overwrite

- `contextual/events.json`, `contextual/msted.json`
- `task/temporal_localization.json`, `task/mcq_openended.json`,
  `task/bcq_openended.json`, `task/causal_linkage.json`
- `task/open_qa.json` (only when the reasoning `open_qa` emitter is enabled)
- `sidecars/reasoning/anomaly.json`
- `raw/<media_id>.<ext>` — created **only if absent**, as a symlink to the
  analyzed media, so a reasoning-only scene is self-contained for
  `training_export`. Reasoning never rewrites existing raw media
  (`stage_raw_media` in `core.scene`, called from `ReasoningTask.run`).

### 3.2 Reasoning must not overwrite (read-only inputs)

- Raw media content in `raw/` (symlink-if-missing only; never a rewrite).
- Detection outputs: `contextual/instances.json`, `contextual/objects.json`,
  `contextual/tracking.json`.
- Captioning outputs: `contextual/chunks.json`,
  `task/scene_description.json`, `task/video_summarization.json`,
  `task/temporal_description.json`, and the scene metadata
  `contextual/video.json` / `contextual/image.json`.
- Closed-form VQA outputs: `task/mcq.json`, `task/bcq.json`.
- PAS outputs under `sidecars/person_attribute_search/` — reasoning reads
  `pas.json` as anomaly-prompt evidence but never writes PAS files.

### 3.3 Shared-filename caution: `task/open_qa.json`

`task/open_qa.json` can be written by **either** `visual_qa` (closed/open QA
split) **or** the reasoning `open_qa` emitter. Enabling both against the same
scene makes the later stage's file win. Enable `open_qa` on only one of the two
stages per pipeline, or route them to separate scenes, to avoid a silent
overwrite.

## 4. Degraded-State Semantics

A "degraded" scene is one that completed but is missing one or more owned
outputs. Auto-Labeling is fallback-friendly by design; a missing
optional label is expected, not an error. Degradation is expressed through two
layers.

### 4.1 Pipeline failure policy

`LinearPipeline` runs each task under an `EmptyOutputPolicy`
(`core.policy`), default `WARN`:

- `WARN` (default): a task that raises is recorded as a failed `StageOutcome`,
  its outputs are skipped, and the pipeline continues with downstream stages.
- `FAIL`: a task that raises aborts the whole pipeline with
  `StagePolicyError`.

The active policy is logged per stage (`... (policy=warn|fail)`).

### 4.2 Best-effort LLM emitters

Reasoning and QA emitters never fail the pipeline return code. In `emit_stage`:

- A missing LLM endpoint, prompt-registry failure, `PromptError`,
  `DaftConvertError`, or write error is logged at `WARNING` and the stage is
  skipped (no file written).
- An emitter returning `None` (for example empty caption windows, or no items
  after filtering) means "nothing to write" and is not an error.
- Image scenes short-circuit temporal-only stages
  (`requires_temporal_axis`) with no output.

Additional per-sample degradation inside a stage:

- LLM responses truncated with `finish_reason='length'` or returned empty are
  dropped for that sample; the stage still completes with the remaining
  samples. The token-budget expansion in `core.llm.clients`
  (`_maybe_expand_token_budget`) mitigates this by growing `max_tokens` on the
  first length truncation.

### 4.3 How degradation is recorded

- `sidecars/pipeline_state.json` (`ScenePipelineState`) tracks the outcome of
  each owned emitter under `annotation_export.emitters` (name, success,
  artifact). `update_annotation_export_state` replaces **only** the emitters a
  stage owns and preserves all others, so an independently-run stage never
  erases a sibling stage's inventory.
- `annotation_export.success` is `true` when **any** owned emitter succeeded,
  so it signals "produced something", not "produced everything". Consumers that
  need completeness must check individual emitter entries and/or file presence.

### 4.4 Downstream expectations

- Consumers must tolerate missing optional `task/` and `contextual/` files; a
  degraded scene is valid.
- `training_export` (TAO-VL-Reason) has one hard requirement: an analyzed media
  file under `raw/<media_id>`. A scene missing raw media fails that scene's
  export (this is why the reasoning stage self-stages raw media in 3.1).

## Open Items

- Formalize a machine-readable per-scene completeness report (beyond the
  any-succeeded `annotation_export.success` flag) so reviewers can distinguish a
  fully-labeled scene from a degraded one without inspecting files.
- Decide whether `task/open_qa.json` should be split into distinct filenames for
  the `visual_qa` and `reasoning` producers to remove the shared-filename
  overwrite risk in 3.3.
