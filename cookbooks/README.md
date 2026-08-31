# Workflow Runner Cookbooks

Cookbooks are scenario-level workflow-runner configurations. Keep scenario
defaults, prompt files, and question banks here instead of hard-coding them in
the runner.

## Table of Contents

- [Recommended Layout](#recommended-layout)
  - [`configs/`](#configs)
  - [`prompts/`](#prompts)
  - [`question_bank.json`](#question_bankjson)
- [Included Examples](#included-examples)
- [Common Run Pattern](#common-run-pattern)
- [Scenario Runbooks](#scenario-runbooks)
- [Cookbook Contract](#cookbook-contract)
- [Authoring Optimized Recipes](#authoring-optimized-recipes)

Each cookbook is self-contained: it ships its own configs, prompts, and
`question_bank.json`, so a run only needs `--cookbook-file <the cookbook>`.

## Recommended Layout

```text
cookbooks/<scenario>/
  configs/
    pipeline_video.yaml
    pipeline_image.yaml
  prompts/
    ...
  question_bank.json
```

### `configs/`

Pipeline YAML files live here. They select the media pipeline, stage sequence,
runtime defaults, endpoints, and per-stage cookbook fields.

### `prompts/`

Prompt files live under stage-specific folders and are referenced from
`workflow.nodes.<node>.args` with paths relative to the cookbook config.

### `question_bank.json`

Each cookbook ships its own `question_bank.json` beside its config. Keep banks
self-contained rather than cross-referencing another cookbook.

## Included Examples

Each tracked config ships sample media paths and default model names so
`--container-dry-run` produces a concrete plan from a fresh checkout. Traffic,
warehouse, and intersection-frame files are not in git; stage them from NGC
before a real run
([Samples and Cookbooks](../docs/user-guide/samples-and-cookbooks.md#download-vss-sample-clips)).
Copy the config to `*.local.yaml` (gitignored) in the same `configs/`
directory and fill in your model-cache, SAM3 weights mount, served VLM/LLM
model names, and optional production media path.

- `video_data_augmentation/configs/pipeline_video.yaml` provides end-to-end
  traffic-safety video auto-labeling.
- `visual_attribute_search/configs/` provides four explicit visual-attribute
  workflows for attribute JSON, multi-view images, video reasoning, and
  video attribute search. See the
  [Visual Attribute Search contracts](visual_attribute_search/README.md).
- `image_spatial_grounding/configs/pipeline_grounding.yaml` provides caption
  to SAM3 regions; `pipeline_referring.yaml` provides regions to referring
  phrases.
- `smart_spaces/configs/pipeline_warehouse_event_reasoning.yaml` provides a
  warehouse event-verification and reasoning workflow.

## Common Run Pattern

Never edit a tracked cookbook with machine-specific paths or credentials. Copy
the closest config to a gitignored `*.local.yaml`, then set output,
model-cache, mount, endpoint, model, and optional production media values.

Inspect the container plan before running:

```bash
CONFIG=<cookbook.local.yaml>
ARGS="--cookbook-file ${CONFIG} --container-dry-run" \
  make run SCRIPT=workflow-runner:main
```

After reviewing the plan, run every stage as the invoking user:

```bash
CONFIG=<cookbook.local.yaml>
ARGS="--cookbook-file ${CONFIG} --container-user auto" \
  make run SCRIPT=workflow-runner:main
```

Export secret variables in the host environment, then pass only their names
with `--container-env`, for example `--container-env NVIDIA_API_KEY`. Never put
a secret value in a cookbook or pass it as `KEY=VALUE` on the CLI. Non-secret
runtime overrides may use `KEY=VALUE`; for example,
`--container-env CUDA_VISIBLE_DEVICES=0` selects a GPU.

Cookbook `container.env` values use list form. Use a bare name such as
`NVIDIA_API_KEY` to pass through a host variable and `NAME=value` only for
non-secret fixed configuration.

## Scenario Runbooks

### Video data augmentation

The traffic-safety pipeline runs:

```text
super_resolution -> detection_and_tracking -> captioning -> visual_qa
  -> reasoning
```

The tracked config uses
`data/input_media/videos/traffic_video_analytics/` after you stage NGC
traffic clips there. Copy
`video_data_augmentation/configs/pipeline_video.yaml` to
`pipeline_video.local.yaml` when you need a different media/output path,
SeedVR2 model cache, SAM3 mount, or VLM/LLM endpoints. The scenario uses
`question_bank.json` and
`prompts/dense_caption/traffic_scene_prompt.md`; captioning and Visual QA use
aligned four-second windows.

### Visual Attribute Search

Exactly four configs are checked in:

- `pipeline_image_attributes_pas.yaml` generates query bundles from one explicit
  attribute JSON document and needs only a text LLM.
- `pipeline_image_multiview_pas.yaml` jointly captions and questions all views of
  one identity before PAS assembly. Download RSTPReid first; images are not in
  git ([dataset](https://github.com/NjtechCVLab/RSTPReid-Dataset)).
- `pipeline_video_pas_reasoning.yaml` combines event-verification reasoning,
  per-track attributes, PAS, and training export.
- `pipeline_video_epas.yaml` produces captions, anomaly evidence, per-track
  attributes, PAS query buckets, and contextual PAS output.

The terminal `event-and-person-attribute-search-service` is PAS-only; the service
name is unchanged. Captioning and Visual QA are explicit upstream nodes in
workflows that require them. For inputs, node flow, endpoints, mounts, dry-run
commands, and expected artifacts, see the
[Visual Attribute Search runbook](visual_attribute_search/README.md).

### Image spatial grounding

The image spatial grounding scenario runs:

```text
captioning -> grounding_2d -> referring_expressions
```

Two configs cover the two genuinely distinct entry points:

- `pipeline_grounding.yaml` runs `captioning -> grounding_2d` — caption the
  scene, then ground visible objects with SAM3 boxes and masks. Sample media
  is `data/input_media/images/traffic_intersection_frames/` after you stage
  NGC traffic clips and extract one still per clip.
- `pipeline_referring.yaml` runs `referring_expressions` only, for phrase
  generation over the grounding scenes. Run `pipeline_grounding.yaml`
  first so each scene already has `contextual/objects.json`.

Live grounding must override the default slim image:

```yaml
container:
  images:
    grounding_2d: grounding-2d-sam3-service
```

An explicit caption can be supplied in `<data_path>/sidecars/input.json`;
otherwise upstream captioning output is used. Deployment-specific tuning
belongs in `workflow.nodes.grounding_2d.args`; inspect the service `--help`
output for supported flags.

Configure image/output paths, VLM endpoint, SAM3 mount, and detection prompts
for production media. The referring stage itself uses the slim
`referring-expressions-service` image; only detection and grounding require SAM3
images.

`pipeline_referring.yaml` uses the same frames directory and the same
`out_dir` as `pipeline_grounding.yaml`
(`output/auto_labeling/image_spatial_grounding/grounding`). The runner
expands each image to `grounding/<filename>/`, which is where grounding
wrote `contextual/objects.json`.

### Smart Spaces

The Smart Spaces warehouse workflow runs:

```text
detection_and_tracking -> captioning -> event_verification_visual_qa -> reasoning
```

Use `smart_spaces/configs/pipeline_warehouse_event_reasoning.yaml` for warehouse
safety and operational-event verification on
`data/input_media/videos/warehouse_safety/` after you stage NGC warehouse
clips there. The
cookbook carries its own warehouse event-verification question bank and does not
run `person_attribute_search` or `training_export`.

## Cookbook Contract

Cookbook YAML is not passed to stage services directly. The runner reads
`data[*].inputs.media_path` plus `data[*].output.out_dir`/`data_path`, preserves
an optional `data[*].id`, converts those entries into the core `DataEntry`
schema, writes a temporary JSONL manifest, and passes that manifest to every
container with `--input-file`.

Each config can choose `pipeline: video` or `pipeline: image`. Shared
container runtime knobs such as `model_cache_path` and `gpu_ids` belong
under `runtime:`. The default configs use repo-root-style sample paths so dry-run plans stay
readable. Stage NGC clips before a real run; replace the output, endpoint,
checkpoint, and optional media values before production runs. Example paths
include
`data/input_media/videos/traffic_video_analytics/traffic_sample_000.mp4`,
`data/input_media/images/traffic_intersection_frames/traffic_sample_000.jpg`,
`ckpts`, and `output/auto_labeling/<scenario>`.
Without an explicit stage list, pipelines use these minimal default subsets:

```text
video: super_resolution -> detection_and_tracking -> captioning -> reasoning
image: captioning -> reasoning
```

The shipped video-data-augmentation cookbook also enables `visual_qa` between
captioning and reasoning. Scenario configs can select stages with `stages:` or
per-stage `enabled:` flags; the runner still applies the canonical relative
order documented in the
[services overview](../docs/developer/architecture/services-overview.md). For DAG-style
recipes, use `workflow.nodes` with one node per stage and `needs:`
dependencies:

```yaml
workflow:
  nodes:
    tracking:
      stage: detection_and_tracking
    captions:
      stage: captioning
      needs: [tracking]
    export:
      stage: reasoning
      needs: [captions]
```

The runner validates those dependencies and executes the resulting topological
order sequentially. Nodes may repeat a stage, and each execution keeps its node
ID in the container plan, results, and logs. Keep each node's service CLI
arguments under `workflow.nodes.<id>.args`. Path-valued node arguments are
resolved and mounted automatically. Use repo-root paths for media, output, and
model caches. Use `../` paths for assets beside the scenario config, such as
prompts and question banks.

For mixed-resolution video datasets, include `super_resolution` but gate it
with node args so high-resolution clips are skipped and low-resolution clips
promote `sidecars/active.*` to the SR output:

```yaml
workflow:
  nodes:
    super_resolution:
      stage: super_resolution
      args:
        - --resolution-policy
        - auto
        - --min-input-short-side
        - "720"
        - --min-input-long-side
        - "1280"
```

SeedVR source/runtime is baked into the SR service image. Configure
`runtime.model_cache_path` to a cache containing `seedvr2/ema_vae.pth` and
`seedvr2/seedvr2_ema_3b.pth`, or use the SR service download mode when network
access is allowed.

When `visual_qa.enabled: true` and a question bank is configured, the runner
passes the resolved question bank plus VLM/LLM endpoint settings to the
`visual_qa` container. Use the Visual QA node's `args` to choose a generation
mode such as `window-vlm-llm`; otherwise the visual-QA service defaults to
normalizing existing `sidecars/visual_qa/windows.json` evidence.

## Authoring Optimized Recipes

Start from the closest legacy source cookbook and port the pieces that are
specific to the use case: the question bank, dense-caption prompt, detector
class list, reasoning task sections, and stage budgets. Do not ship empty
question-bank sections.

When adding a new domain cookbook, use the repo-local PAIDF skills for prompt
and detector design. Domain prompts should be evidence-first, parser-aware, and
explicit about uncertainty. Detector settings should separate directly mapped
runner fields from detailed service flags kept with the detector node's `args`.

For a novel use case, ask for the missing decisions before writing the
cookbook: media type, target annotation families, objects/classes of interest,
VQA questions, whether SR is allowed, model endpoints, GPU/model-cache paths,
and whether outputs should prioritize speed or maximum recall. If the user
does not know, offer concrete options such as `traffic safety video`,
`generic image annotation`, `warehouse operations`, or `security surveillance`
and select the nearest existing source cookbook.

For video recipes, make the windowing explicit. Use `--window-frames 0` with
`--window-seconds` for time-based windows, or choose frame windows with
`--window-frames` and `--remainder-threshold`. Mirror the same windowing values
onto `visual_qa` when it generates QA evidence directly from media.

For full reasoning coverage, wire every relevant section from the same bank:
`open_qa.question_file`, `mcq_openended.item_file`,
`bcq_openended.question_file`, and `temporal_localization.query_file` for video.
Run `--container-dry-run` before the real experiment and verify the generated
commands include the expected stage order, endpoints, model cache, windowing,
question bank, prompt files, and writable output mounts.
