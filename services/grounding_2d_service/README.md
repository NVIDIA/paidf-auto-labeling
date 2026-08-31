# 2D Grounding Service

This image-only service extracts groundable expressions from a caption with a
VLM, then uses SAM3 to produce boxes and masks for those expressions.

The task reuses the `detection_and_tracking` **library** (SAM3) in-process so
prompts can be dynamic from the VLM step. Packaging does **not** depend on the
detection service image.

## Images

- `grounding-2d-service` is the slim CPU image for CI, CLI, and packaging
  checks. It does not contain the SAM3 runtime needed for product grounding.
  Build it with `make build IMAGE=grounding-2d-service:main`.
- `grounding-2d-sam3-service` is the end-to-end GPU image. Build it with
  `make build IMAGE=grounding-2d-service:sam3`.

The workflow runner defaults to the slim image. A live grounding workflow must
override the stage image:

```yaml
container:
  images:
    grounding_2d: grounding-2d-sam3-service
```

Both GPU flavors (`detection-and-tracking-*:sam3` and `grounding-2d-service:sam3`)
build from **their own** service `docker/` Dockerfiles with independent
`PACKAGE_NAME` / tags. Rebuild after task or service code changes; do not use
`PYTHONPATH` bind-mounts for product runs.

## Run

```bash
make build IMAGE=grounding-2d-service:main
make build IMAGE=grounding-2d-service:sam3

make run SCRIPT=grounding-2d-service:main ARGS='--help'
make run SCRIPT=grounding-2d-service:main \
  ARGS='--disabled --input-file <image-input.jsonl>'

VLM_URL=http://localhost:8000/v1
ARGS="--input-file <image-input.jsonl> --vlm-endpoint-url ${VLM_URL} \
      --sam3-model-cache-path <model-cache>" \
  make run SCRIPT=grounding-2d-service:main
```

The final command is a full host-environment run and requires a GPU-capable
SAM3 installation and local checkpoints. For a product container run, use the
GPU image through the
[grounding cookbook](../../cookbooks/README.md#2d-grounding).

Use `--help` for caption precedence, filtering, SAM3, threshold, endpoint, and
retry options.

## Inputs

Each input record must be a standard `DataEntry`; `media_path` and `data_path`
are required, while `id` is generated when omitted. The service supports images
only. Captions are resolved from an explicit `--caption`, per-scene input
metadata, or upstream captioning artifacts. Per-scene metadata can be written to
`<data_path>/sidecars/input.json`:

```json
{"caption": "A traffic camera frame showing vehicles and road lanes."}
```

The GPU image requires SAM3 checkpoints and a reachable VLM endpoint.
Authentication depends on the endpoint provider; never store credentials in
payloads or cookbooks.

## Outputs

```text
<data_path>/
└── sidecars/grounding_2d/
    ├── grounding_2d.json                # final stage deliverable
    ├── step0_expressions.json
    └── step1_grounding.json
```

Final deliverable lives under `sidecars/` because it is stage-specific JSON,
not a registered DAFT `task/` type.

For orchestration, treat each stage image as an independently scalable worker.
Mount SAM3 weights read-only at `/models/sam3`, or set `SAM3_MODEL_PATH` to the
mounted directory or checkpoint. Keep the VLM on an external endpoint so the
two runtimes can scale separately.

See the [task contract](../../packages/tasks/grounding_2d/README.md) and
[cookbook](../../cookbooks/README.md#2d-grounding).
