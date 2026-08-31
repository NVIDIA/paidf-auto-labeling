# Referring Expressions

## What it does

The referring expressions service generates a short phrase for each already
detected box that uniquely identifies it in language (for example, "the person
in the red jacket near the door"). It does not detect objects itself — run
[Detection and Tracking](detection-and-tracking.md) or
[2D Grounding](grounding-2d.md) first so the scene already has boxes.

## Before you start

- A scene that already contains `contextual/objects.json` (and optionally
  `contextual/instances.json`) from an earlier detection stage.
- A reachable VLM endpoint. See [VLM and LLM Endpoints](../vlm-llm-endpoints.md).
- This image is slim — no SAM3/CUDA baked in. Pair it with
  `detection-and-tracking-sam3-service` in a cookbook when you need boxes
  first, rather than expecting this service to produce them.

## Step by step

### 1. Build the image

```bash
make build IMAGE=referring-expressions-service:main
```

### 2. Check the full option list

```bash
make run SCRIPT=referring-expressions-service:main ARGS='--help'
```

### 3. Run against a scene that already has boxes

```bash
make run SCRIPT=referring-expressions-service:main \
  ARGS='--input-file payloads/simple.jsonl --vlm-endpoint-url http://localhost:8000/v1'
```

Point `--input-file` at your own manifest once you've confirmed the sample
payload runs. The scene at each entry's `data_path` must already contain
`contextual/objects.json`.

## Verify it worked

- The command exits `0` with no traceback.
- `<data_path>/sidecars/referring_expressions/referring_expressions.json`
  exists and contains one region per authoritative box in
  `contextual/objects.json`. A partial match — fewer regions than boxes — is a
  failure even though the process exits successfully; check the run log for a
  skipped-box warning.
- Every entry has a non-empty `description` and a valid source object ID.
- If overlay drawing is enabled (the default), a marked-up image exists at
  `<data_path>/sidecars/referring_expressions/marked_boxes.jpg`. Disable it
  with `--no-draw-box-overlay`.

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| "No boxes found" / empty output | The scene has no `contextual/objects.json` yet — run detection and tracking or 2D grounding first |
| Fewer regions than boxes | Check the run log for a `Skipping invalid data entry` warning; a box may have failed IoU matching against the source frame |
| Endpoint auth error | Confirm the correct API key env var is exported for your endpoint provider (`NVIDIA_API_KEY` for most hosted NVIDIA endpoints) |

For error output you don't recognize, see
[Troubleshooting](../troubleshooting.md).

## Full argument reference

[`services/referring_expressions_service/README.md`](../../../services/referring_expressions_service/README.md)

## Next

Nothing downstream is required — `referring_expressions.json` is the final
stage deliverable for this workflow.
