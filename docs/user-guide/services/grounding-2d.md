# 2D Grounding

## What it does

The 2D grounding service links a caption to specific regions of an image. It
first asks a VLM to extract groundable phrases from a caption (for example,
"the red car" from "a red car is parked next to a blue truck"), then uses SAM3
to produce a box and mask for each phrase. It is image-only.

## Before you start

- SAM3 checkpoints. See [Model Provisioning](../model-provisioning.md).
- A reachable VLM endpoint. See [VLM and LLM Endpoints](../vlm-llm-endpoints.md).
- A caption per scene, provided one of three ways: `--caption` on the command
  line, a `sidecars/input.json` file with a `caption` key, or upstream
  captioning artifacts already in the scene.
- The right container image: `grounding-2d-service` (the default, CPU-only,
  no SAM3 runtime) is for CI/packaging checks only. Real grounding needs
  `grounding-2d-sam3-service` — see step 3.

## Step by step

### 1. Check the full option list

```bash
make run SCRIPT=grounding-2d-service:main ARGS='--help'
```

### 2. Smoke-test without SAM3 or a VLM endpoint

```bash
make run SCRIPT=grounding-2d-service:main \
  ARGS='--disabled --input-file <image-input.jsonl>'
```

### 3. Run for real (requires GPU + SAM3 checkpoints)

```bash
make run SCRIPT=grounding-2d-service:main \
  ARGS="--input-file <image-input.jsonl> \
        --vlm-endpoint-url http://localhost:8000/v1 \
        --sam3-model-cache-path <model-cache>"
```

For a containerized product run instead of a bare host run, build and use the
SAM3-capable image:

```bash
make build IMAGE=grounding-2d-service:sam3
```

Then drive it through the [workflow runner](../operations-workflow-runner.md)
with the grounding [cookbook](../samples-and-cookbooks.md), overriding the
stage image to `grounding-2d-sam3-service`.

## Verify it worked

- The command exits `0` with no traceback.
- `<data_path>/sidecars/grounding_2d/step0_expressions.json` contains
  extracted expressions.
- `<data_path>/sidecars/grounding_2d/grounding_2d.json` and
  `<data_path>/sidecars/grounding_2d/step1_grounding.json` contain **at least
  one grounded instance**. A file with expressions but zero grounded
  instances means grounding failed even though the command exited
  successfully — check the SAM3 checkpoint and prompts.

Do not look under `task/`: `grounding_2d.json` is not a DAFT task type, and
DAFT validation rejects it there. `contextual/objects.json` is an optional
SAM3 signal when boxes were written; grounding-only success does not require
that file.

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| No expressions extracted | The scene has no caption — pass `--caption`, add `sidecars/input.json`, or run [Captioning](captioning.md) first |
| Expressions extracted but zero grounded instances | Wrong or missing SAM3 checkpoint path, or the phrase doesn't correspond to anything visible — verify `--sam3-model-cache-path`/`SAM3_MODEL_PATH` |
| Command works locally but not in the container | You built/ran the slim `grounding-2d-service` image instead of `grounding-2d-sam3-service` |

For error output you don't recognize, see
[Troubleshooting](../troubleshooting.md).

## Full argument reference

[`services/grounding_2d_service/README.md`](../../../services/grounding_2d_service/README.md)

## Next

Nothing downstream is required —
`sidecars/grounding_2d/grounding_2d.json` is the final stage deliverable
for this workflow. Referring phrases are a separate cookbook node that
reads `contextual/objects.json`.
