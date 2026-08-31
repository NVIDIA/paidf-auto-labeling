# event-and-person-attribute-search-service

PAS-only service for assembling person attributes and generating tiered retrieval
queries. The service does not caption media, inspect images, run Visual QA, copy
person images, or synthesize tracking manifests.

## Inputs

The service accepts either of two attribute sources:

1. Scene-local Visual QA sidecars produced by `visual-qa-service`.
2. An explicit local or remote `--attribute-json` file.

An explicit attribute file always wins. Because it represents one person, it may
only be used with one service input `DataEntry`. Supported shapes are:

```json
{"person_key":"person_1","attributes":{"top_outer_color":"red"}}
```

```json
[{"image_id":"front.jpg","attributes":{"top_outer_color":"red"}}]
```

```json
{"entries":[{"image_id":"front.jpg","attributes":{"top_outer_color":"red"}}]}
```

For legacy query generation, multiple entries are treated as views of the same
person: scalar attributes are merged by majority vote with input-order
tie-breaking and accessories are unioned. In bundle mode, queries are generated
separately for every image entry so view-specific attributes remain available.
Unknown, null, empty, and placeholder values are ignored.

Without `--attribute-json`, PAS uses the existing scene contracts:

- `sidecars/visual_qa/items.json` for a single identity;
- `sidecars/visual_qa/windows.normalized.json` plus
  `sidecars/detection_and_tracking/tracks.json` for video tracks; or
- `sidecars/person_attribute_search/track_inputs.json` for a preassembled track seam.

Caption, anomaly, crop, and tracking inputs are optional or flow-specific
sidecars produced by dedicated upstream services. The checked-in
[Visual Attribute Search cookbook contracts](../../cookbooks/visual_attribute_search/README.md)
show the exact producer for every configured input.

## Outputs

Single-person legacy runs write PAS records under
`sidecars/person_attribute_search/`:

- `attributes.json` — identity, dataset, images, and normalized attributes;
- `queries.json` — identity, attributes, and easy/medium/hard legacy queries;
- `hitl.json` — optional HITL preannotation output.

With `bundle_query_generation` and an explicit attribute JSON, `attributes.json`,
`queries.json`, and `hitl.json` are replaced by `bundle_attributes.json`,
`bundle_queries.json`, and optional `bundle_hitl.json`. All use the
`{chunk_id, n_people, people: {key: {...}}}` shape. There is one
`people` entry and one model call per input image. Unique person keys are kept as
map keys; repeated person keys use `image_id` to prevent overwrites. The enclosing
key identifies the person/image and matches across bundle documents, so source
identity, image, attribute, and diagnostic metadata is omitted from query entries.

Attribute bundle entries contain only `track_id` and normalized `attributes`;
query bundle entries contain only `image_filename` and `queries`.
HITL bundle entries contain `track_id`, `image_url`, and per-image `preannotations`.

In legacy template/per-tier generation, easy and medium queries are
`[query, attributes_used]` pairs and hard queries are strings. With
`bundle_query_generation` enabled, all three tiers are string lists, matching
the bundle prompt response and avoiding empty `attributes_used` placeholders.
`natural_caption` is included only when the selected input actually provides a
nonempty caption.

Existing per-track runs continue to write `pas.json`, `chunk_queries.json`, and
the aggregated `bundle_queries.json` when bundle generation is enabled.

## Usage

Generate queries directly from an attribute artifact:

```bash
main --input-file payloads/simple.jsonl \
  --attribute-json /data/input_attributes.json \
  --llm-provider openai-compatible \
  --llm-endpoint-url http://localhost:8082/v1 \
  --llm-model google/gemma-4-31B-it \
  --query-prompt-file /cookbook/pas_synonymous_query_prompt.json \
  --query-count 3 \
  --temperature 0.2 \
  --top-p 0.9
```

Consume Visual QA artifacts already present in each scene:

```bash
main --input-file payloads/simple.jsonl --config-file /data/pas.yaml
```

`--config-file` accepts direct `person_attribute_search` configuration or the
legacy nested `event_and_person_attribute_search.person_attribute_search` shape.
Legacy captioning and Visual QA blocks are ignored. There is no `image_pas`
service mode; attribute-only and multi-view image PAS are compositions expressed
by the workflow nodes in their respective cookbooks.

## LLM retries

PAS query generation uses separate retry budgets for endpoint failures and
unusable model output:

```yaml
llm_retries: 2
llm_retry_backoff_s: 2.0
llm_response_retries: 2
llm_response_retry_backoff_s: 2.0
```

`llm_retries` handles transient network failures, timeouts, rate limits, and
server errors. `llm_response_retries` repeats a successful request when its text
is empty, malformed, or does not satisfy the applicable bundle, tiered-query, or
bucket contract. Response retries use exponential backoff capped at 60 seconds.
The same settings are available as `--llm-retries`, `--llm-retry-backoff-s`,
`--llm-response-retries`, and `--llm-response-retry-backoff-s`.

The budgets are independent. With both retry counts set to `2`, one logical
generation can make up to nine endpoint requests in the worst case. After
exhaustion, strict bundle generation fails, legacy tiered generation uses its
existing deterministic fallback, and bucket generation emits PAS-only queries.

## Container

The image contains only `core`, `person-attribute-search`, and the service
wrapper. It intentionally has no video codec, OpenCV, PyAV, captioning, or Visual
QA runtime dependencies.

```bash
make build IMAGE=event-and-person-attribute-search-service:build
```
