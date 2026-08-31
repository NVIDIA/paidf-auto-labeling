# Training Export

## What it does

The training export service is a **batch** step that packages a directory of
completed DAFT scenes into a training-ready dataset. It runs after annotation
services (typically [Reasoning](reasoning.md)) have finished, and it can emit
more than one training format from the same set of scenes in one pass.

## Before you start

- A manifest listing the completed scenes to export — one `DataEntry` per
  scene, identified by `data_path`. Scenes without supported `task/` files or
  resolvable media are skipped with a warning; the export fails only if
  **nothing** in the manifest converts.
- No GPU, VLM, or LLM endpoint is required — this service copies and
  repackages existing scene data; it does not decode or encode media.

## Step by step

### 1. Check the full option list

```bash
make run SCRIPT=training-export-service:main ARGS='--help'
```

### 2. Export one or more formats

```bash
make run SCRIPT=training-export-service:main \
  ARGS='--input-file <completed-scenes.jsonl> \
        --training-export-format cosmos-reason-v1.0 \
        --training-export-format tao-vl-reason-v1.0 \
        --training-export-dir output/training'
```

Repeat `--training-export-format` for each format you need in the same run.
Use `--training-export-task` to restrict which DAFT task types are included.

### 3. (Optional) Run it as a workflow-runner terminal stage

Instead of a standalone batch call, a cookbook can configure
`training_export:` as its last node — see
[Operations: Workflow Runner](../operations-workflow-runner.md#training-export).

## Verify it worked

- The command exits `0` with no traceback.
- Each requested format has its own subdirectory under
  `<training-export-dir>`, for example:

  ```text
  output/training/
  ├── cosmos-reason-v1.0/
  └── tao-vl-reason-v1.0/
  ```

- Check the run log for skip warnings — a scene silently missing from the
  export usually means it lacked a supported `task/` file or its media could
  not be resolved.

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| Export fails immediately | No item in the manifest had a supported task file or resolvable media — verify the manifest points at completed scenes |
| One scene is missing from the output | Check the run log; unsupported items are skipped with a warning rather than failing the whole export |
| Exporting a very large dataset is slow/heavy | Shard the manifest — one export per dataset, split, or shard — instead of one call over the entire corpus |

For error output you don't recognize, see
[Troubleshooting](../troubleshooting.md).

## Full argument reference

[`services/training_export_service/README.md`](../../../services/training_export_service/README.md)

## Next

This is the terminal stage of a pipeline — the export directory is the final
product handed off for model training.
