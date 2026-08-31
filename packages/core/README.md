# Auto-Labeling Core

Shared abstractions and utilities used by all task and service packages in this
workspace. Every task and service depends on this package. Core sits at the
bottom of the dependency graph and must not import tasks or services.

## Main Surfaces

- `core.interfaces` — shared service, pipeline, and task execution contracts.
- `core.scene` — DAFT scene skeleton, raw/active media handoff, and pipeline
  state.
- `core.formats.daft` — DAFT envelopes, validation, conversion, and atomic
  writes.
- `core.model_clients` and `core.llm` — provider-neutral VLM/LLM requests.
- `core.media` — approved video probing, decoding, frame iteration, and output.
- `core.utils.multistorage` — local and remote storage handling.
- `core.utils.telemetry` — OpenTelemetry setup for services and tasks.

See the [documentation index](../../docs/README.md) for architecture, remote
storage, observability, and artifact contracts.

## DAFT File Contract

`core.formats.daft` owns the shared DAFT file-contract primitives: envelope
creation, canonical type routing, validation, and atomic JSON writes under
`contextual/` or `task/`. It also owns deterministic service pivots that turn
captioning and Visual-QA sidecars into DAFT payloads. Task packages own prompts
and model-specific sidecar generation.

The default validator catches writer-shape errors. For stricter schema checks,
compose it with `JsonSchemaDaftPayloadValidator` pointed at a metropolis-v3.0
schema directory, or use `NvidiaTaoDaftPayloadValidator` when the
`nvidia-tao-daft` package is installed.

## Media utilities

`core.media` owns approved H.264, VP9, and MPEG-4 Part 2 video probing and
decoder selection, RGB frame streaming, and VP9 output policy. PyAV is optional
because most core consumers do not encode video. Install `core[video-output]`
to use `Vp9VideoWriter`; the writer imports PyAV lazily when it is opened.

Controlled service containers must still exclude the resolved PyAV wheel and
use `scripts/media_toolchain.py` to build PyAV from source against the
repository's restricted FFmpeg build.
