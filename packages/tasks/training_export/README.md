# Training Export Task

Batch task that exports completed DAFT scene annotations into supported
training formats.

The task reads one or more scene directories from `DataEntry.data_path` and
writes aggregate training datasets such as `cosmos-reason-v1.0` and
`tao-vl-reason-v1.0`. This package owns both task-level orchestration and
training-format conversion.

## Input Contract

Each scene must contain supported DAFT task files under `task/` and its analyzed
media under `raw/`. Unsupported or malformed task items are skipped with
warnings; a conversion with no supported items reports an error.

## Supported Formats

- `tao-vl-reason-v1.0` groups converted annotations by DAFT task type. Each
  task type is written as `<task-type>.json`; copied images and videos are
  stored under `images/` and `videos/`.
- `cosmos-reason-v1.0` writes one conversation JSON per sample under `text/`,
  a `meta.json` index, and copied media under `media/`.

Media copying is enabled by default. No-copy exports retain references to
source media; callers must preserve that media and its path relationship.

## Usage

Most callers should use the
[training export service](../../../services/training_export_service/README.md)
or enable the `training_export` stage in a workflow-runner cookbook.
