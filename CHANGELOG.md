# Changelog

All notable changes to this repository should be recorded here.

The version number below reflects the current workspace version declared in
`pyproject.toml`.

## [1.1.0] - 2026-08-15

Auto-Labeling supersedes the legacy `pseudo-labeling` repository
(v1.0.0), which ran one fixed pipeline — super resolution, detection and
tracking, VLM JSON, and MCQ generation — through a single Docker container and
CLI. This release re-architects that pipeline as a monorepo of independent
services and adds annotation capabilities legacy did not have.

### Added — new capabilities beyond legacy v1.0.0

- 2D grounding: caption-to-box/mask grounding with SAM3
  (`grounding_2d_service`)
- Referring expressions: per-box discriminative phrase generation
  (`referring_expressions_service`)
- Event and Person Attribute Search (EPAS/PAS): person attribute assembly plus
  tiered easy/medium/hard natural-language search query generation, including
  attribute-only and multi-view image PAS flows
  (`event_and_person_attribute_search_service`)
- Reasoning stages beyond legacy's MCQ generation: events, multi-step temporal
  event description (msted), causal linkage, and temporal localization, in
  addition to open-ended/MCQ/BCQ QA (`reasoning_service`)
- Native SAM 3.1 Object Multiplex tracking backend alongside RF-DETR +
  BoostTrack/ByteTrack and SAM3 (Transformers) detection/tracking
- Multi-format training export (`cosmos-reason-v1.0`, `tao-vl-reason-v1.0`) as
  a standalone batch stage or terminal workflow node
  (`training_export_service`)
- Multi-Storage Client-backed remote storage support for `media_path` and
  `data_path`
- Repository-wide observability assets (collector and dashboard
  configuration)

### Changed — architecture

- Re-architected from legacy's single monolithic pipeline and container into
  a uv-workspace monorepo of independently versioned, independently scalable
  services (`core` -> task packages -> service packages), composed by a
  cookbook-driven `workflow-runner` instead of one fixed CLI config
- Consolidated video codec handling into shared `core.media` utilities with
  one enforced policy across every stage: H.264 (NVIDIA CUVID hardware
  decode only), VP9, and MPEG-4 Part 2 inputs; VP9-only generated output
- Introduced a shared DAFT scene contract and `daft_validation` package
  enforcing identical artifact-ownership and validation behavior across every
  stage

### Added — documentation

- Operator user guide covering installation, getting started, per-service
  walkthroughs, model provisioning, VLM/LLM endpoint setup, cookbooks,
  remote storage, experiment output layout, and troubleshooting
- Package and service README index

### Known differences from legacy

See [Limitations](docs/user-guide/installation.md#limitations) for currently
disclosed gaps against the legacy `pseudo-labeling` pipeline.

## [1.0.0] - Legacy baseline

Kept here for reference only. This version was never released from this
repository — it is the `pseudo-labeling` repository
([NVIDIA/paidf-auto-labeling](https://github.com/NVIDIA/paidf-auto-labeling/tree/main)),
the predecessor this repository supersedes and that [1.1.0](#110---2026-08-15)
is written against as a baseline.

Legacy `pseudo-labeling` ran one fixed pipeline through a single Docker
container and CLI (`modules/cli.py`):

- **Super Resolution (SR)** — upscale input media
- **Detection & Tracking** — detect objects and assign track IDs
- **VLM JSON** — generate scene/event metadata with a VLM
- **MCQ Generation** — generate MCQ, BCQ, and open-QA task files with an LLM

All four stages ran in sequence by default, with partial runs supported when
earlier-stage outputs already existed. Every capability listed under
[1.1.0's "Added" section](#110---2026-08-15) is new relative to this
baseline.
