# Getting Started

Follow this page start to finish for your first labeling run. You will copy a
cookbook, fill in a few machine-specific paths, dry-run it, then run it.

How a run works: a cookbook YAML lists stages, paths, and endpoints. The
workflow runner starts one container per stage, in order. Every stage reads
and writes the same scene directory under your `out_dir`. You do not call
services yourself unless you are debugging one stage.

If you are changing Auto-Labeling source code, still do this once so you can run the
product. Contributor checks live in
[Local Development](../developer/local-development.md).

## Before you start

You need:

- Python 3.12+, `uv`, GNU Make, and Docker or Podman.
  [Installation](installation.md)
- GPU checkpoints for detection and, if you use it, super resolution.
  [Model Provisioning](model-provisioning.md)
- A reachable VLM and LLM endpoint, plus an API key in your shell.
  [VLM and LLM Endpoints](vlm-llm-endpoints.md)

### Step 1 — Confirm the runner is installed

```bash
git clone <repo-url>
cd paidf-auto-labeling
make sync
make run SCRIPT=workflow-runner:main ARGS='--help'
```

If `--help` fails, stop and fix [Installation](installation.md) before
continuing.

### Step 2 — Provision the models you need

Path A needs GPU checkpoints for detection and (optionally) super resolution.
Follow [Model Provisioning](model-provisioning.md) to download them before
continuing — skip this step only if you already have a populated model cache.

### Step 3 — Set up a VLM/LLM endpoint

Path A's captioning, Visual QA, and reasoning stages call an external model
endpoint. Follow [VLM and LLM Endpoints](vlm-llm-endpoints.md) to either
deploy one yourself or point at a hosted one, then verify it responds before
moving on.

### Step 4 — Dry-run the cookbook

A dry run validates cookbook parsing, stage order, image selection, and mount
resolution without running any containers. Always do this before a real run —
see the Path A commands below.

### Step 5 — Run for real

Run the same cookbook without `--container-dry-run`. This builds/pulls the
required images and actually executes each stage.

### Step 6 — Confirm the output

A successful run produces one scene directory per input under your configured
output root, containing `raw/`, `contextual/`, `task/`, and `sidecars/`. See
[Experiment Output Layout](experiment-output-layout.md) for what belongs in
each folder.

Once you've completed all six steps for Path A, you know the full shape every
other workflow in this repo follows — only the cookbook file and the stages it
chains change.

## Five Paths

| Path | What | Notes |
| --- | --- | --- |
| **A — Video auto-labeling** | Super resolution, tracking, captioning, visual QA, and reasoning | Best first cookbook for the general video flow |
| **B — Smart Spaces warehouse reasoning** | Tracking, captioning, event QA, and reasoning | Best use-case-specific warehouse sample |
| **C — Video visual attribute search + reasoning + export** | Tracking, captioning, event QA, reasoning, person QA, visual attribute search, training export | Best end-to-end example with a downstream export |
| **D — Image spatial grounding** | Captioning, 2D grounding with SAM3, and referring phrases | Image-only |
| **E — Attribute-only or multi-view visual search** | Visual attribute search over explicit attributes or one identity image group | Best attribute-search-specific validation |

Prerequisites: Python 3.12+, `uv`, GNU Make, Docker or Podman, and endpoint or
GPU prerequisites for the selected workflow. Full matrix:
[Installation](installation.md#requirements).

Path A is the worked example below. Paths B–E use the same copy, edit,
dry-run, run pattern with a different cookbook.

## Path A — Video Auto-Labeling

Cookbook:
`cookbooks/video_data_augmentation/configs/pipeline_video.yaml`

The tracked file uses placeholders for media, SAM3 weights, and endpoint model
names. Copy it, then edit only the values for this machine:

```bash
cp cookbooks/video_data_augmentation/configs/pipeline_video.yaml \
  cookbooks/video_data_augmentation/configs/pipeline_video.local.yaml
```

Set these fields in `pipeline_video.local.yaml`:

| Field | What to put there |
| --- | --- |
| `data[*].inputs.media_path` | Your video, or a staged NGC traffic clip under `data/input_media/videos/` |
| `data[*].output.out_dir` | A writable output directory, for example `output/auto_labeling/video_data_augmentation` |
| `runtime.model_cache_path` | The checkpoint cache from [Model Provisioning](model-provisioning.md) |
| `container.mounts` | Read-only mount of the SAM3 weights into the detection image |
| `endpoints.vlm` and `endpoints.llm` | URL and model name from [VLM and LLM Endpoints](vlm-llm-endpoints.md) |

Stage the NGC traffic clips first (see
[Samples and Cookbooks](samples-and-cookbooks.md#download-vss-sample-clips)).
Path A then uses
`data/input_media/videos/traffic_video_analytics/traffic_sample_000.mp4`.

Export the API key in the same shell you will use to run (never put the value
in the cookbook):

```bash
export NVIDIA_API_KEY="<your-key>"
# Local unauthenticated endpoints can use: export NVIDIA_API_KEY="EMPTY"
```

Dry-run first. This compiles the stage plan without starting models:

```bash
CONFIG=cookbooks/video_data_augmentation/configs/pipeline_video.local.yaml
make run SCRIPT=workflow-runner:main \
  ARGS="--cookbook-file ${CONFIG} --container-dry-run"
```

Check the printed plan for stage order, image names, the model-cache path,
SAM3 mounts, and endpoint URLs. Then run for real:

```bash
make run SCRIPT=workflow-runner:main \
  ARGS="--cookbook-file ${CONFIG} --container-user auto --container-ensure-images --container-env NVIDIA_API_KEY"
```

`--container-user auto` keeps output files writable across stages.
`--container-ensure-images` builds any missing stage images.
`--container-env NVIDIA_API_KEY` passes the key into every stage container
without writing the secret into YAML.

Success: one scene directory appears under your `out_dir` with `raw/`,
`contextual/`, `task/`, and `sidecars/`. See
[Experiment Output Layout](experiment-output-layout.md) for what belongs in
each folder. If the run fails, start at [Troubleshooting](troubleshooting.md).

## Path B — Smart Spaces Warehouse Reasoning

Use the Smart Spaces cookbook when you want a use-case-specific warehouse
event-verification and reasoning workflow:

- `cookbooks/smart_spaces/configs/pipeline_warehouse_event_reasoning.yaml`

It runs:

`detection_and_tracking -> captioning -> event_verification_visual_qa -> reasoning`

Stage the NGC warehouse clips first. Copy the cookbook to `*.local.yaml`,
set the media path, output path, model cache, SAM3 mount, and endpoint
URLs, then run:

```bash
CONFIG=cookbooks/smart_spaces/configs/pipeline_warehouse_event_reasoning.local.yaml
make run SCRIPT=workflow-runner:main \
  ARGS="--cookbook-file ${CONFIG} --container-dry-run"
```

Success signal: the per-scene directory contains event-verification Visual QA
sidecars plus reasoning `task/` outputs.

## Path C — Video Visual Attribute Search, Reasoning, And Training Export

This is the most complete visual-attribute cookbook shipped in the repo:

- `cookbooks/visual_attribute_search/configs/pipeline_video_pas_reasoning.yaml`

It runs:

```text
detection_and_tracking
  -> captioning
  -> event_verification_visual_qa
  -> reasoning
  -> person_attribute_visual_qa
  -> person_attribute_search
  -> training_export
```

Copy it to `pipeline_video_pas_reasoning.local.yaml`. Set the media path,
output path, model cache, SAM3 mount, and endpoint URLs the same way as Path A.
Stage the NGC warehouse clips first. Path C then uses
`data/input_media/videos/warehouse_safety/warehouse_safety_0000.mp4`.

```bash
CONFIG=cookbooks/visual_attribute_search/configs/pipeline_video_pas_reasoning.local.yaml
export NVIDIA_API_KEY="<your-key>"

make run SCRIPT=workflow-runner:main \
  ARGS="--cookbook-file ${CONFIG} --container-dry-run"

make run SCRIPT=workflow-runner:main \
  ARGS="--cookbook-file ${CONFIG} --container-user auto --container-ensure-images --container-env NVIDIA_API_KEY"
```

Success:

- the per-scene directory contains `contextual/`, `task/`, and PAS sidecars
- the configured training-export directory contains a `tao-vl-reason-v1.0/`
  dataset

Video EPAS without export uses
`cookbooks/visual_attribute_search/configs/pipeline_video_epas.yaml`
with the same run commands.

## Path D — Image Spatial Grounding

Cookbook: `cookbooks/image_spatial_grounding/configs/pipeline_grounding.yaml`

Copy it to `pipeline_grounding.local.yaml`, then set:

- image `media_path` (sample:
  `data/input_media/images/traffic_intersection_frames/` after you stage NGC
  traffic clips and extract one still per clip)
- output directory
- VLM endpoint
- SAM3 weights mount

Dry-run and run through the workflow runner as in Path A, including
`--container-env NVIDIA_API_KEY`. Real runs need the SAM3-capable grounding
image. `pipeline_referring.yaml`, in the same
`cookbooks/image_spatial_grounding/configs/` directory, runs
`referring_expressions` only. Run grounding first, then point referring at
the same grounding `out_dir` so it can read `contextual/objects.json`.

## Path E — Visual-Attribute-Only Image Flows

Two visual-attribute image contracts are checked in:

- `cookbooks/visual_attribute_search/configs/pipeline_image_attributes_pas.yaml`
  — PAS over one attribute JSON file. The attribute-only flow never opens
  media. Change the JSON path in both `media_path` and `--attribute-json`, and
  keep exactly one data entry.
- `cookbooks/visual_attribute_search/configs/pipeline_image_multiview_pas.yaml`
  — one identity, several views. Person images are not in git. Download
  [RSTPReid](https://github.com/NjtechCVLab/RSTPReid-Dataset)
  ([Google Drive archive](https://drive.google.com/file/d/1HTeDZUVrZr6nL56ZlkYBNqjSWh3IGV2X/view?usp=sharing)),
  copy one identity into
  `data/input_media/images/image_attribute_augmentations/`, then keep
  `media_path` as a single representative file and pass the folder through
  `--image-group-dir`. Change both `--image-group-dir` paths and
  `media_path` together.

Copy to `*.local.yaml`, set the LLM endpoint, then dry-run and run as in
Path A.

## What to read next

- [Samples and Cookbooks](samples-and-cookbooks.md) — every sample file and
  what you must change
- [Operations: Workflow Runner](operations-workflow-runner.md) — more runner
  flags
- [Services](services/README.md) — run one stage by itself
- [Troubleshooting](troubleshooting.md) — failed dry-runs, images, endpoints,
  and missing outputs
