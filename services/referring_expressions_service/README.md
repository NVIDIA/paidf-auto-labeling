# Referring Expressions Service

This image-only product stage generates short, discriminative VLM phrases for
known DAFT boxes. It does not detect objects; run detection and tracking or 2D
grounding first.

## Inputs

Each `DataEntry` must point to an image and a scene containing
`contextual/objects.json`. `contextual/instances.json` is optional. Configure a
reachable VLM with `--vlm-endpoint-url` and `--vlm-model`.

Authentication depends on the endpoint provider. Hosted NVIDIA endpoints
typically use `NVIDIA_API_KEY`; self-hosted endpoints use their own
authentication policy.

## Run

```bash
make build IMAGE=referring-expressions-service:main
make run SCRIPT=referring-expressions-service:main \
  ARGS='--input-file payloads/simple.jsonl --vlm-endpoint-url http://localhost:8000/v1'
```

Use `--help` for the complete CLI, including overlay drawing, frame selection,
IoU matching, retry, and force-reprocessing options.

## Outputs

```text
<data_path>/
└── sidecars/referring_expressions/
    ├── referring_expressions.json       # final stage deliverable
    ├── step0_region_expressions.json
    └── marked_boxes.jpg                 # default; omit with --no-draw-box-overlay
```

Final deliverable lives under `sidecars/` because it is stage-specific JSON,
not a registered DAFT `task/` type.

This image is slim (no SAM3/CUDA bake). Pair with
`detection-and-tracking-sam3-service` in cookbooks when you need boxes first.
Rebuild after task/service code changes; do not rely on `PYTHONPATH` mounts.

See the [task contract](../../packages/tasks/referring_expressions/README.md)
and [cookbook](../../cookbooks/README.md#referring-expressions) for pipeline
composition.
