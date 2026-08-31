# Reasoning

`reasoning` is the task package that produces DAFT v3 JSON artifacts from an
existing scene directory. The Python import package is `reasoning`.

This package owns opt-in DAFT-export behavior:

- prompt registry and bundled prompt variants for optional LLM stages
- LLM adapters for DAFT export stages
- stage emitters that compose existing contextual inputs into LLM-derived DAFT outputs
- reasoning enrichment and reasoning stripping for DAFT-export outputs

Shared file-contract primitives live in `core.formats.daft`: version literals,
envelope construction, type routing, deterministic captioning/Visual-QA pivots,
and atomic writes. Pipeline-level DAFT validation lives in the separate
`daft-validation` task package. Keeping those primitives in `core` lets services
write DAFT-compatible files without pulling in DAFT export prompts or LLM
adapters.

## Architecture

The package separates four concerns:

- Shared utilities load scene inputs, prompts, paths, and timecodes.
- Pure converters produce DAFT payload dictionaries.
- LLM adapters obtain structured results but do not write files.
- Stage emitters apply policy and write through the shared atomic DAFT writer.

Task-specific prompts and export orchestration remain in this package. Reusable
DAFT mechanics and deterministic captioning/Visual-QA pivots remain in
`core.formats.daft`.

## Data Flow

DAFT export runs over a scene directory with the standard layout:

```text
<scene>/
  raw/
  contextual/
  task/
  sidecars/
```

The service loads existing contextual files and task-owned sidecars, then runs
LLM-driven stages when a `DaftExportConfig` enables them. Deterministic
captioning and Visual-QA pivots are written by their respective services.

Input sources:

- `contextual/video.json` or `contextual/image.json`
- `contextual/events.json` and `contextual/instances.json` when present
- captioning sidecars under `sidecars/metadata*.json` or
  `sidecars/captioning/`
- artifact references in `sidecars/pipeline_state.json`

Output files are DAFT annotation files under `contextual/` and `task/`.
Task-internal intermediate files must stay under `sidecars/`; structural scene
validation is performed by the separate `daft-validation` package.

## Optional Exports

LLM-backed stages run only when their config block is present and enabled:

- Event aggregation writes `contextual/events.json`.
- MSTED writes `contextual/msted.json`.
- Temporal localization writes `task/temporal_localization.json`.
- Open-ended QA writes `task/open_qa.json`, `task/mcq_openended.json`, and
  `task/bcq_openended.json`.
- Causal linkage writes `task/causal_linkage.json`.
- Reasoning enrichment adds optional reasoning fields to selected task outputs.

Binary yes/no questions route only to BCQ; non-binary closed-choice questions
route only to MCQ.

All writes are structurally validated and atomic. Full pipeline validation
belongs to the separate `daft-validation` task.

## Extending Reasoning

Add a pure converter first. Add an LLM adapter only when the stage calls a
model, then add a stage emitter and opt-in config. Keep prompt logic out of
`core`, and cover converter, adapter, and filesystem behavior independently.

## Testing

```bash
make lint-check
make mypy
make test
```
