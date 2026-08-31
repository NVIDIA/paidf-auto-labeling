# Reasoning Service

Service wrapper for the `reasoning` task package.

The service runs DAFT exporters over existing scene directories. Each input
`DataEntry` should point at:

- `media_path`: original media path, used to derive `SceneContext`
- `data_path`: local DAFT scene directory to read/write

Example JSONL payload:

```json
{"media_path": "data/input_media/videos/traffic_video_analytics/traffic_sample_000.mp4", "data_path": "output/video1/"}
```

Stage that traffic clip from NGC before using the sample payload. See
[Samples and Cookbooks](../../docs/user-guide/samples-and-cookbooks.md#download-vss-sample-clips).

## Running

```bash
make run SCRIPT=reasoning-service:main
make run SCRIPT=reasoning-service:main ARGS='--input-file payloads/simple.jsonl'
make run SCRIPT=reasoning-service:main ARGS='--help'
```

`make run SCRIPT=reasoning-service:main` offers a `simple payload` preset.
Pass `ARGS=...` with your input manifest and `DaftExportConfig` file to enable
optional LLM stages.

Important options:

- `--config-file`: JSON/YAML `DaftExportConfig` for optional LLM stages.
- `--llm-provider`: provider adapter for reasoning LLM requests. Currently only
  `openai-compatible` is supported.
- `--llm-endpoint-url`, `--llm-model`: endpoint overrides for enabled LLM
  stages. OpenAI-compatible auth uses `NVIDIA_API_KEY`.
- `--reasoning-mode {config,keep,strip}`: preserve or strip task reasoning fields.

## Input Contract

The service expects the standard scene skeleton:

```text
<scene>/
  raw/
  contextual/
  task/
  sidecars/
```

It creates missing top-level directories, then reads canonical contextual files
and task sidecars. The service runs `daft-validation` before and after reasoning
so unknown DAFT filenames and media contract mismatches fail before any stage
work and again after new outputs are written. Task-internal artifacts belong
under `sidecars/`.

Captioning inputs for opt-in LLM stages are discovered from:

- `sidecars/metadata.json`
- `sidecars/metadata_chunk.json`
- `sidecars/captioning/metadata.json`
- `sidecars/captioning/metadata_chunk.json`
- `sidecars/captioning/video_captions.json`
- `sidecars/captioning/image_caption.json`
- `sidecars/captioning/image_captions.json`
- `sidecars/pipeline_state.json` entries under `task_artifacts["captioning"]`

## Outputs

Enabled LLM stages may write:

- `contextual/events.json`
- `contextual/msted.json`
- `task/open_qa.json`
- `task/temporal_localization.json`
- `task/mcq_openended.json`
- `task/bcq_openended.json`
- `task/causal_linkage.json`

Deterministic captioning outputs (`contextual/chunks.json`,
`task/temporal_description.json`, `task/scene_description.json`,
`task/video_summarization.json`) are written by `captioning-service`.
Deterministic Visual-QA outputs (`task/mcq.json`, `task/bcq.json`,
`task/open_qa.json`) are written by `visual-qa-service`.

After each scene, `sidecars/pipeline_state.json` is updated with
`annotation_export` emitter outcomes for files emitted by the current run. Stale
canonical files that already existed are left in place but are not counted as
current-run successes.

## Training Exports

Reasoning writes per-scene DAFT annotations only. To aggregate completed scenes
into a supported training format, run `training-export-service` as a separate
batch step after reasoning completes.

## Config Shape

Config may be a bare DAFT export object or nested under `reasoning`:

```yaml
reasoning:
  events:
    enabled: true
    prompt_variant: events_from_chunks
  msted:
    enabled: true
    prompt_variant: msted_default
```

Every LLM-driven stage is opt-in through `enabled: true`. The service resolves
the LLM endpoint from `--llm-endpoint-url` and `--llm-model`; stages with no
resolved endpoint log a warning and skip writing.

## Testing

Run service tests with a clean `PYTHONPATH` so ambient pytest plugins from the
host environment do not affect collection:

```bash
env PYTHONPATH= uv run --package reasoning-service python -m pytest services/reasoning_service/tests
```

The paired task package tests are:

```bash
env PYTHONPATH= uv run --package reasoning python -m pytest packages/tasks/reasoning/tests
```

## Docker

Registered image:

```bash
make build IMAGE=reasoning-service:build
```

The reasoning task operates on scene metadata rather than decoding or encoding
media, so its container does not install OpenCV or PyAV. It retains the
checksum-pinned, restricted input-only FFmpeg build for core media telemetry.
