# Visual Attribute Search Task

> Product name: **Visual Attribute Search** (person attribute search +
> event/anomaly retrieval-query generation + cross-pass packaging). The package
> import path, distribution name (`person-attribute-search`), workflow-runner
> stage key, and `sidecars/person_attribute_search/` output dir intentionally
> stay `person_attribute_search` as a stable internal contract.

Terminal task that turns explicit attributes or upstream producer sidecars into
search-ready artifacts. Dedicated captioning, Visual QA, and
detection/tracking services create media-derived evidence before this task; the
PAS service only runs this assembly task. This package owns typed assembly,
schema validation, event/anomaly + caption retrieval-query buckets, pipeline
state, optional DAFT mirror, and cross-pass packaging.

## Table of Contents

- [Role in the pipeline](#role-in-the-pipeline)
- [Responsibilities](#responsibilities-one-module-per-concern)
- [Prompt Assets](#prompt-assets)
- [Inputs / Outputs](#inputs--outputs)
- [Failure semantics](#failure-semantics)
- [Parity](#parity)
- [Testing](#testing)

## Role in the pipeline

`PersonAttributeSearchTask` is an **assembly** task. It does not detect people,
caption media, or run visual question answering itself. It accepts either
upstream Visual QA sidecars or an explicit attribute JSON, normalizes them into
the PAS domain model, and writes the product-facing search artifacts.

```text
detection_and_tracking  -> tracks/crops
captioning              -> scene or image captions
visual_qa               -> person attributes, hard queries, anomaly votes
person_attribute_search -> PAS sidecars, query buckets, optional DAFT mirrors
```

Two text-LLM paths are individually configurable:

- `llm_query_generation` — per-person medium/hard retrieval queries.
- `bucket_query_generation` — chunk-level Anomaly and Caption query buckets.

Both use the same configured text-LLM endpoint client. They are independent of
the upstream VLM calls used to create captions and Visual QA evidence.

## Responsibilities (one module per concern)

| Module | Concern |
|---|---|
| `schema.py` | Typed `PersonAttributes` model + v3 vocabulary + strict-v3/video field split + tolerant validation |
| `attributes.py` | Adapt upstream Visual QA `items` → `PersonAttributes` |
| `sources.py` | Sidecar discovery, track-seam loading, and adaptive attribute-source resolution |
| `identity.py` | Aggregate many images/crops into one person identity |
| `tracks.py` | Per-track aggregation: `TrackRecord` → one `people[]` entry (video flow) |
| `track_inputs.py` | Assembler that merges crops + per-track model outputs into the seam |
| `templates.py` | Deterministic easy/medium query templates (slot fill + stop-words) |
| `queries.py` | Assemble `QuerySet`, extract hard queries, and flatten chunk queries |
| `llm_queries.py` | Optional per-person LLM tiered-query generation |
| `bucket_queries.py` | Optional chunk-level PAS/Anomaly/Caption query buckets |
| `prompts.py` | Legacy PAS prompt assets (`data/prompts/`) + placeholder rendering |
| `outputs.py` | Pipeline state, DAFT mirror, and merged anomaly deliverable writes |
| `export/hitl.py` | Data-factory HITL preannotation documents |
| `export/benchmark.py` | Legacy `generated_*_v3` JSON envelopes |
| `export/chunk_pas.py` | Per-chunk video `pas.json` + flat `queries.json` documents |
| `config.py` | `PersonAttributeSearchConfig` source paths, output paths, LLM knobs, export toggles |
| `task.py` | Thin `PersonAttributeSearchTask` coordinator |

## Prompt assets

The legacy generative prompts are version-pinned under `data/prompts/` and
consumed by `visual_qa`/`captioning` via their `prompt_file` seam (or rendered
through `prompts.py`):

| Asset | Drives |
|---|---|
| `pas_attributes.txt` | VLM attribute extraction (v3 schema, `primary (fine)` colors) |
| `pas_queries.txt` | LLM tiered query generation (`{attributes}`, `{caption}`) |
| `pas_image_queries.txt` | Per-image VLM caption + hard queries (`{attributes}`) |

## Inputs / Outputs

### Image / single-identity flow

Reads (from the scene `sidecars/`):
- `visual_qa/items.json` — structured attribute answers (PAS question bank)
- or an explicit `attribute_json` file; this source overrides scene sidecars

Writes (to `sidecars/person_attribute_search/`):
- Legacy generation: `attributes.json`, `queries.json`, `hitl.json`
- Explicit-attribute bundle generation: `bundle_attributes.json`,
  `bundle_queries.json`, and optional `bundle_hitl.json`; all contain one matching
  `people` entry per source image

### Video per-track flow

When a per-track seam exists the task switches to the multi-person flow:

Reads:
- `person_attribute_search/track_inputs.json` — one record per track
  (`{chunk_id, crop_root, source_annotation, anomaly_labels, caption_queries,
  tracks: [{track_id, detection_score, crop_dir, items: [...], ...}]}`). This
  seam may be supplied directly; normal video cookbooks instead let PAS join
  the canonical detection/tracking and per-track Visual QA sidecars.

Writes (to `sidecars/person_attribute_search/`):
- `pas.json` — `{chunk_id, pas: {n_people, people: [...]}, crop_root}`
- `chunk_queries.json` — flat, de-duplicated `{queries: [{query}, ...]}` combining
  every person's easy/medium/hard tiers, anomaly labels, and caption queries

These are **non-DAFT sidecars** by design. DAFT is a closed-enum system and the
PAS v3 attribute schema / tiered retrieval queries are not DAFT
`contextual`/`task` types, so — like the anomaly sidecar — PAS writes
under `sidecars/`, not `contextual/`/`task/`. There is no separate
`daft_export` stage on this repo; an optional lossy DAFT projection (e.g.
queries → `open_qa`) can be added later if a consumer needs it.

## Failure semantics

The core PAS sidecars are the source of truth; optional derived outputs are
fail-soft:

- DAFT contextual mirror failures are logged and do not remove PAS sidecars.
- Merged anomaly deliverable failures are logged and do not remove PAS sidecars.
- Bucket query LLM failures fall back to the PAS bucket only.

The task records success/failure in scene pipeline state under
`task_artifacts["person_attribute_search"]`. Additive-output degradation is
recorded there in `optional_failures`, so a run that lost only an optional
mirror, merged deliverable, or query bucket is distinguishable from a core PAS
write failure.

LLM-backed query generation has two independent retry budgets. `llm_retries`
and `llm_retry_backoff_s` cover transient endpoint failures;
`llm_response_retries` and `llm_response_retry_backoff_s` cover successful calls
that return empty, malformed, or contract-invalid query output. All four apply
to bundle, per-person tiered, and chunk-bucket generation.

## Parity

`tests/test_video_parity.py` locks the per-chunk video **output contract**:
given a seam reconstructed from a golden `sample_output` (`tests/fixtures/video/`),
the per-track assembly reproduces the golden `pas.json` exactly and the golden
query list exactly. Generative *content* (LLM query wording) is produced by the
live per-track fan-out and is out of scope for the offline harness.

## Testing

```bash
PYTHONPATH=packages/tasks/person_attribute_search/src:packages/core/src \
  uv run pytest packages/tasks/person_attribute_search/tests -q
```

For end-to-end wiring, use one of the
[Visual Attribute Search cookbooks](../../../cookbooks/visual_attribute_search/README.md).
The video flows declare detection, captioning, and each Visual QA pass as
separate nodes before `person_attribute_search`; use `--container-dry-run` first
to inspect the generated Docker commands.
