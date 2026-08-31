# Referring Expressions Task

Given **known instances** (boxes), generate discriminative referring phrases
with a VLM. Domain-agnostic: object `type` is open vocabulary (no traffic enum).

Auto-Labeling port of `2d-data-engine/referring-data-engine` (boxes → language), not
`grounding-data-engine` (caption → boxes).

## Pipeline

```
DataEntry media (image)
  + contextual/objects.json   (boxes per frame)
        │
        ▼
Step 0  Region expressions (VLM + SoM-lite marks)
        → short phrases linked to DAFT object_ids
```

Shipped surface is **Step 0**:

- Numbered box overlay sent to the VLM
- Free-form `type` / `color` / `description` (types normalized to slugs)
- Link by mark id first, then greedy IoU
- Keep authoritative DAFT boxes on output regions

## Inputs

| Source | Role |
|--------|------|
| `media_path` | Image |
| `contextual/objects.json` | Per-frame boxes (`bounding_box_2d_tight`, `object_id`) |
| `contextual/instances.json` | Optional DAFT metadata |

Upstream: `detection_and_tracking` or `grounding_2d`.

## Outputs

```
sidecars/referring_expressions/referring_expressions.json
sidecars/referring_expressions/step0_region_expressions.json
```

Final deliverable and Step 0 diagnostics both live under
`sidecars/referring_expressions/` (not under DAFT `task/`).

## Auth

Auth depends on the configured VLM provider/endpoint (`vlm_provider`,
`vlm_endpoint_url`, `vlm_model` via `create_endpoint_client` / `ChatRequest`).
Hosted NVIDIA endpoints typically need `NVIDIA_API_KEY`; self-hosted or
OpenAI-compatible endpoints follow that endpoint's auth instead.

## Cookbook composition

```yaml
stages:
  - detection_and_tracking   # or grounding_2d
  - referring_expressions
```

Images stay independent: GPU SAM3 service for boxes, slim referring service for
VLM phrasing (Osmo/Airflow-friendly).
