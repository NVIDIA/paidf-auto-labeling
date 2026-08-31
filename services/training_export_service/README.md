# Training Export Service

Batch service for exporting completed DAFT scene directories into training
datasets.

This service is intended to run after annotation services such as
`reasoning-service` have completed. It accepts one `DataEntry` per completed
scene and aggregates those scene directories into one or more training-format
outputs.

## Inputs

Each input record identifies one completed scene through `data_path`. Export
expects supported DAFT task files under `task/` and analyzed media under
`raw/`. Unsupported items or items whose media cannot be resolved are skipped
with warnings; the export fails if no supported items are converted. Use one
manifest per dataset, split, or shard.

## Run

```bash
make run SCRIPT=training-export-service:main \
  ARGS='--input-file <completed-scenes.jsonl> \
        --training-export-format cosmos-reason-v1.0 \
        --training-export-format tao-vl-reason-v1.0 \
        --training-export-dir output/training'
```

Repeat `--training-export-format` to emit multiple formats and
`--training-export-task` to select task types. Use `--help` for metadata,
media-copy, and remote-storage options.

## Outputs

Each format is written under the configured export directory:

```text
<training-export-dir>/
├── cosmos-reason-v1.0/
└── tao-vl-reason-v1.0/
```

Format-specific files and media-copy behavior are documented by the
[training export task](../../packages/tasks/training_export/README.md).

For thousands of videos, schedule annotation at video scale first, then run one
export per dataset, split, or shard.

## Container

```bash
make build IMAGE=training-export-service:build
```

The container includes the checksum-pinned, restricted input-only FFmpeg build
for core media telemetry. Training export copies media without decoding or
encoding it, so the image does not install OpenCV or PyAV.
