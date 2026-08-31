# Visual Attribute Search

## What it does

**Visual Attribute Search** assembles a person's attributes (clothing
color, accessories, and similar) from upstream evidence, then asks an LLM to
generate tiered natural-language search queries (`easy`, `medium`, `hard`) for
retrieving that person later. It is an assembly stage: it does not caption
media, run Visual QA, copy person images, or build tracking data itself.

The service and CLI names are unchanged:
`event-and-person-attribute-search-service` runs the internal
`person_attribute_search` stage. The cookbook family is now
`cookbooks/visual_attribute_search/`.

## Before you start

- Either an explicit `--attribute-json` file describing one person, **or** a
  scene that already contains Visual QA sidecars
  (`sidecars/visual_qa/items.json` for a single identity, or
  `sidecars/visual_qa/windows.normalized.json` plus
  `sidecars/detection_and_tracking/tracks.json` for video tracks).
- A reachable LLM endpoint for query generation. See
  [VLM and LLM Endpoints](../vlm-llm-endpoints.md).
- A query prompt file (the repo ships one under the Visual Attribute Search
  cookbook directory).

This service intentionally has no video codec, OpenCV, or PyAV dependency —
it only reasons over already-extracted attribute JSON.

## Step by step

### 1. Check the full option list

```bash
make run SCRIPT=event-and-person-attribute-search-service:main ARGS='--help'
```

### 2. Generate queries directly from an attribute JSON file

Use this path when you already have one person's attributes as JSON (for
example: `{"person_key": "person_1", "attributes": {"top_outer_color": "red"}}`):

```bash
make run SCRIPT=event-and-person-attribute-search-service:main ARGS="\
  --input-file payloads/simple.jsonl \
  --attribute-json /data/input_attributes.json \
  --llm-provider openai-compatible \
  --llm-endpoint-url http://localhost:8082/v1 \
  --llm-model <served-llm-model> \
  --query-prompt-file /cookbook/pas_synonymous_query_prompt.json \
  --query-count 3"
```

### 3. Generate queries from Visual QA evidence already in the scene

```bash
make run SCRIPT=event-and-person-attribute-search-service:main ARGS="\
  --input-file <input.jsonl> --config-file <pas.yaml>"
```

`--config-file` accepts a `person_attribute_search` block directly.

## Verify it worked

- The command exits `0` with no traceback.
- For single-person legacy runs:
  `<data_path>/sidecars/person_attribute_search/attributes.json` and
  `.../queries.json` exist, and `queries.json` has non-empty `easy`, `medium`,
  and `hard` entries.
- For bundle-mode runs (explicit attribute JSON with bundle generation):
  `bundle_attributes.json` and `bundle_queries.json` exist instead, keyed by
  person.
- Generated queries only reference attributes that were actually present in
  the input — PAS does not invent missing attributes.

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| "No attribute source" error | Neither `--attribute-json` nor a scene-local Visual QA sidecar was found — provide one or the other |
| Queries missing a tier | The LLM response didn't satisfy the tiered-query contract; the service retries automatically (`--llm-response-retries`, default `2`) before falling back |
| More than one attribute source configured | An explicit `--attribute-json` always wins over scene sidecars — remove it if you intended to use scene evidence |

For error output you don't recognize, see
[Troubleshooting](../troubleshooting.md).

## Full argument reference

[`services/event_and_person_attribute_search_service/README.md`](../../../services/event_and_person_attribute_search_service/README.md)
and the
[Visual Attribute Search cookbook contracts](../../../cookbooks/visual_attribute_search/README.md).

## Next

[Training Export](training-export.md) if this run is part of a cookbook that
terminates in a training dataset.
