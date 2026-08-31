# Auto-Labeling Example Service

> **This package is a template.** Copy it when adding a new annotation service
> and replace the placeholder implementation with product logic.

A service composes one or more task packages into a runnable annotation
pipeline. Services depend on `core` and their task packages, and register an
entrypoint so `make run` can discover them.

## Run

```bash
make run SCRIPT=example-service:main
make run SCRIPT=example-service:main \
  ARGS='--input-file payloads/simple.jsonl'
make build IMAGE=example-service:build
```

Each input is a standard `DataEntry` containing `media_path` and `data_path`.
The template creates the scene skeleton and writes:

```text
sidecars/example_task.json
sidecars/example_task_manifest.json
sidecars/pipeline_state.json
```

The template container includes the checksum-pinned, restricted input-only
FFmpeg build for core media telemetry. The example task does not decode or
encode media, so the image does not install OpenCV or PyAV.
