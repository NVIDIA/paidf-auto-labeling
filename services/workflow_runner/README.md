# workflow-runner

Auto-Labeling-specific recipe compiler and local launcher for Dockerized annotation
stage services.

`workflow-runner` reads CLI arguments or cookbook YAML, resolves Auto-Labeling-specific
inputs, prompts, question banks, stage args, mounts, and optional
`workflow.nodes` dependencies, then runs the selected stage containers
sequentially. It is not a general-purpose orchestrator: scheduling,
distributed execution, worker queues, retry policy, and resource placement
belong to platform tooling such as OSMO or Airflow.

The runner does not import task packages directly; each stage runs in its own
service container against the same `DataEntry` JSONL manifest and shared scene
directory. The repository-level `scripts/run.py` and `scripts/build.py`
entrypoints still own local script launching and stage Docker image discovery
through package `pyproject.toml` metadata. This service uses those repo
conventions at the boundary, but keeps stage ordering, mounts, GPU/runtime
options, and shared sidecar dataflow in one Auto-Labeling-specific command-line tool.

```text
DataEntry JSONL
  -> paidf-super-resolution-service                   optional, video
  -> paidf-detection-and-tracking-{rfdetr,sam3}-service optional
  -> paidf-referring-expressions-service              optional, image
  -> paidf-captioning-service
  -> paidf-grounding-2d-service                       optional, image
  -> paidf-visual-qa-service                          optional
  -> paidf-reasoning-service
  -> paidf-training-export-service                    optional
  -> paidf-event-and-person-attribute-search-service  optional (PAS)
```

The default image and build target names match the stage-service packages:

- `super_resolution` — image `paidf-super-resolution-service`; build target
  `super-resolution-service:build`.
- `detection_and_tracking` — image
  `paidf-detection-and-tracking-rfdetr-service`; build target
  `detection-and-tracking-service:rfdetr`.
- `referring_expressions` — image `paidf-referring-expressions-service`; build target
  `referring-expressions-service:main`.
- `captioning` — image `paidf-captioning-service`; build target
  `captioning-service:main`.
- `grounding_2d` — image `paidf-grounding-2d-service`; build target
  `grounding-2d-service:main`.
- `visual_qa` — image `paidf-visual-qa-service`; build target
  `visual-qa-service:build`.
- `reasoning` — image `paidf-reasoning-service`; build target
  `reasoning-service:build`.
- `training_export` — image `paidf-training-export-service`; build target
  `training-export-service:build`.
- `person_attribute_search` — image
  `paidf-event-and-person-attribute-search-service`; build target
  `event-and-person-attribute-search-service:build`.

`reasoning` replaces the former `daft_export` stage: on `main`, DAFT export was
folded into per-task DAFT validation plus the `reasoning` service. The legacy
name `daft_export` is still accepted in `--stages` and cookbook sections as an
alias for `reasoning`, so older configs keep working.

`person_attribute_search` is an opt-in PAS assembly stage. It consumes an
explicit attribute JSON or sidecars produced by explicit captioning, Visual QA,
and detection/tracking nodes, then writes person attributes, retrieval queries,
and optional anomaly/contextual packaging. The PAS-only
`event-and-person-attribute-search-service` does not inspect media or generate
those upstream artifacts.

## Prerequisites

- Run commands from the repository root so the workflow runner can resolve
  workspace package metadata. Build its image with
  `make build IMAGE=workflow-runner:workflow-runner` when a containerized runner
  is required.
- Install Docker or Podman. GPU runs require a runtime that supports `--gpus`
  such as Docker with NVIDIA container toolkit.
- Install `make` and sync the repo dependencies before invoking the runner.
- Ensure any required container registry credentials, GPU access, and endpoint
  environment variables are available to the shell that launches `make run`.
  Pass endpoint-key variable names with `--container-env`, or declare them in a
  cookbook `container.env` block, so stage containers receive them.

### Model endpoints and secrets

Stage containers use host networking by default
(`--container-network host`). On Linux, an endpoint running on the same machine
can therefore use `http://localhost:<port>/v1`. If you select another container
network, use a hostname or address reachable from that network.

Endpoint authentication depends on the provider. OpenAI-compatible model
clients use `NVIDIA_API_KEY`; Gemini clients use `GEMINI_API_KEY`. Pass only the
environment-variable name:

```bash
make run SCRIPT=workflow-runner:main \
  ARGS='--cookbook-file <cookbook.local.yaml> \
        --container-env NVIDIA_API_KEY'
```

For local endpoints that do not require authentication, no key needs to be
passed. Never put secret values in a cookbook, payload, or command line.

## Run

Inspect the generated container commands:

```bash
make run SCRIPT=workflow-runner:main
make run SCRIPT=workflow-runner:main \
  ARGS='--input-file payloads/simple.jsonl --container-dry-run'
```

`make run SCRIPT=workflow-runner:main` offers presets for a checked-in
`payloads/simple.jsonl` dry run, the default video cookbook dry run, and help.

Run the default video pipeline:

```bash
make run SCRIPT=workflow-runner:main \
  ARGS='--pipeline video --input-file payloads/simple.jsonl'
```

Run the default image pipeline:

```bash
make run SCRIPT=workflow-runner:main \
  ARGS='--pipeline image --input-file payloads/simple.jsonl'
```

Include visual QA before reasoning:

```bash
make run SCRIPT=workflow-runner:main \
  ARGS='--input-file payloads/simple.jsonl \
        --stages detection_and_tracking captioning visual_qa reasoning \
        --question-bank-file /path/to/question_bank.json'
```

## Cookbooks

Scenario cookbooks live under repo-root `cookbooks/`, and the runner also
accepts external scenario cookbooks with `--cookbook-file`. Each scenario
typically provides one pipeline config plus endpoint defaults, detector
classes, prompt paths, question bank, and reasoning settings. The tracked
examples cover traffic video augmentation, Visual Attribute Search,
2D grounding, and referring expressions.

```text
cookbooks/<scenario>/
  configs/
    pipeline_video.yaml  # or pipeline_image.yaml
  prompts/
  question_bank.json
```

Use a cookbook directly:

```bash
CONFIG=cookbooks/video_data_augmentation/configs/pipeline_video.yaml
ARGS="--cookbook-file ${CONFIG} --container-dry-run" \
  make run SCRIPT=workflow-runner:main
```

The checked-in cookbooks use repo-root-style sample paths such as
`data/input_media/videos/traffic_video_analytics`,
`data/input_media/images/traffic_intersection_frames`, and
`output/auto_labeling/<scenario>`, so dry-run plans stay readable. Traffic
and warehouse files are not in git; stage them from NGC before a real
experiment. Replace `data[*].inputs.media_path`, `data[*].output.out_dir`,
and endpoint values as needed.

Cookbook configs choose the media pipeline with `pipeline: video` or
`pipeline: image`. Shared container runtime defaults such as
`model_cache_path` and `gpu_ids` live under `runtime:`.

The runner includes separate default pipelines:

```text
video: super_resolution -> detection_and_tracking -> captioning -> reasoning
image: captioning -> reasoning
```

Cookbooks can override the pipeline with `stages:` or by setting each
stage section's `enabled:` value. For forward-compatible graph-shaped
recipes, they can also declare a dependency-aware `workflow.nodes` block:

```yaml
workflow:
  nodes:
    sr:
      stage: super_resolution
    tracking:
      stage: detection_and_tracking
      needs: [sr]
    captions:
      stage: captioning
      needs: [tracking]
    qa:
      stage: visual_qa
      needs: [tracking, captions]
    reason:
      stage: reasoning
      needs: [captions, qa]
```

The current container executor still runs sequentially. It validates
`workflow.nodes`, rejects duplicate node IDs, unknown dependencies, and cycles,
and compiles the graph into a stable topological node order. A node may repeat a
stage and may add an `args` list:

```yaml
workflow:
  nodes:
    anomaly_qa:
      stage: visual_qa
      needs: [captions]
      args: [--question-bank-file, ../anomaly.json,
             --output-items-sidecar, visual_qa_anomaly/items.json]
    person_qa:
      stage: visual_qa
      needs: [anomaly_qa]
      args: [--question-bank-file, ../person.json,
             --output-items-sidecar, visual_qa_per_track/items.json]
```

The node ID is preserved in plans, results, and logs. Shared
`stage_args.visual_qa` values apply to both executions; node `args` are appended
after them, so node-specific options win. Local path values in node args are
resolved relative to the cookbook and mounted. This remains sequential workflow
composition, not parallel execution or per-node input branching. Local directory
`media_path` entries are expanded into one scene per image/video file by default.

When `visual_qa` is enabled and a question bank is configured, the runner
passes the resolved question bank plus VLM/LLM endpoint settings to the
`visual_qa` container. Cookbook `stage_args` file values such as prompt files
are resolved relative to the cookbook config file before they are passed into
containers.

## Container Plan

For an end-to-end run, the runner builds the container plan in this order:

1. Load the cookbook, resolve paths, choose `pipeline` plus `stages` or
   `workflow.nodes`, and convert `data[*].inputs.media_path` plus
   `data[*].output.out_dir` into `DataEntry` records.
2. Expand local media directories into one scene per media file unless
   `--no-expand-input-dirs` is set.
3. Resolve local media/output paths to absolute paths, create output
   directories, and write a temporary JSONL manifest.
4. Build one `docker run` command per selected stage. Plain `--stages`,
   `stages:`, and stage `enabled:` toggles use canonical dataflow order;
   `workflow.nodes` uses the graph's topological order. Each stage receives
   `--input-file <manifest>` plus its stage-specific arguments.
5. Add identity bind mounts for the manifest, media parents, output scene
   directories, model cache, question banks, prompt files, image-group and
   attribute paths from shared or node arguments, reasoning/PAS config, and any
   explicit `--container-mount` values.

Always run `--container-dry-run` first and inspect stage order, image names,
mount modes, model cache, endpoint URLs, question-bank path, and any
`stage_args`. For time-based video windows, the generated captioning or VQA
command must include both `--window-seconds <N>` and `--window-frames 0`; if
`--window-frames` is omitted, the captioning/VQA service uses frame-count
windows by default.

## Data Contract

The runner writes one temporary JSONL manifest and passes it to every
stage with `--input-file`. Local `media_path` and `data_path` values are
resolved to absolute paths and mounted into every container at the same
path. Remote URI-like values such as `msc://...` are left unchanged and
not mounted by the local runner.

Stage containers communicate through each scene's `data_path`, including
`sidecars/pipeline_state.json` and the DAFT `contextual/`, `task/`, and
`sidecars/` folders.

### Visual Attribute Search cookbook contracts

The four checked-in Visual Attribute Search configs demonstrate the
supported composition patterns:

| Config | Ordered nodes | Endpoints |
|---|---|---|
| `pipeline_image_attributes_pas.yaml` | PAS only over `--attribute-json` | LLM |
| `pipeline_image_multiview_pas.yaml` | captioning -> Visual QA -> PAS | VLM + LLM |
| `pipeline_video_pas_reasoning.yaml` | tracking -> captioning -> event VQA -> reasoning -> person VQA -> PAS -> export | VLM + LLM |
| `pipeline_video_epas.yaml` | tracking -> captioning -> anomaly VQA -> person VQA -> PAS | VLM + LLM |

See the [cookbook runbook](../../cookbooks/visual_attribute_search/README.md)
for inputs, mounts, expected producer and PAS artifacts, and a dry-run command
for every config.

## Training Exports

Training exports run as their own `training-export-service` stage/container.
When export formats are configured and `--stages` is not explicitly set, the
workflow runner appends this stage after reasoning. The source scene is the
per-entry `data_path` directory containing `raw/`, `contextual/`, `task/`, and
`sidecars/`.

```bash
make run SCRIPT=workflow-runner:main \
  ARGS='--input-file payloads/simple.jsonl \
        --training-export-format cosmos-reason-v1.0 \
        --training-export-format tao-vl-reason-v1.0 \
        --training-export-dir output/training'
```

Exports are written under one subdirectory per format, for example
`output/training/cosmos-reason-v1.0/` and
`output/training/tao-vl-reason-v1.0/`. Use `--training-export-task <type>` to
limit the Metropolis task files included in the training export.

Cookbooks can configure the same standalone export stage:

```yaml
training_export:
  enabled: true
  formats: [cosmos-reason-v1.0, tao-vl-reason-v1.0]
  output_dir: output/training/default_video
  task_types: [mcq, open_qa]
  copy_media: true
  metadata:
    description: Default video QA training split
    license: internal
    tags: [traffic, qa]
```

The exporter resolves media using each scene's DAFT `contextual/video.json` or
`contextual/image.json` `format` field and expects the corresponding file at
`raw/{media_id}.{format}`. Samples with missing contextual media metadata or
missing raw media are skipped and logged as warnings.

For real container runs, the runner also appends lightweight execution events
to `<data_path>/logs/workflow_runner.jsonl` for each local scene. These JSONL
records capture stage start/success/failure, return codes, image names, and
the generated container command. Dry-runs do not write scene logs, and
container stdout/stderr still stream live rather than being captured into this
file.

When running multiple local stage containers that use different default image
users, run as your own user so all stages can read and write the same mounted
scene directories. Pass `--container-user "$(id -u):$(id -g)"`, or set
`--container-user auto` (or `container.user: auto` in a cookbook) to resolve the
current `uid:gid` automatically.

### Rootless env is automatic

Whenever the runner runs as a non-root user, it auto-injects the environment
variables that rootless containers need (`USER`, `HOME`, `HF_HOME`,
`XDG_CACHE_HOME`, `TORCHINDUCTOR_CACHE_DIR`), so callers no longer list them by
hand. Any value you pass via `--container-env` (or a cookbook `container.env`)
takes precedence over the injected default of the same name.

### Cookbook `container:` block

A cookbook can declare its container wiring once so a run only needs
`--cookbook-file`:

```yaml
container:
  user: auto            # resolve to the invoking uid:gid
  images:               # pin per-stage images
    detection_and_tracking: paidf-detection-and-tracking-sam3-service
  env:                  # merged with (not replaced by) --container-env
    SAM3_MODEL_PATH: /models/sam3
  mounts:               # merged with (not replaced by) --container-mount
    - /host/models/sam3:/models/sam3:ro
```

Scalar fields (`user`, `images`) yield to explicit CLI flags; `env` and `mounts`
are additive (cookbook first, CLI last so CLI wins on conflicts).

## Useful Options

- `--container-dry-run` logs commands without launching containers.
- `--container-build-images` builds enabled stage images before running.
- `--container-ensure-images` builds only enabled stage images that are missing
  (existing images are left untouched; ignored when `--container-build-images`
  is set).
- `--container-mount HOST[:CONTAINER[:ro|rw]]` adds a bind mount.
- `--container-env NAME` passes an environment variable through.
- `--container-user UID:GID` runs each stage container as the same user
  (`auto` resolves the current `uid:gid`).
- `--stage-arg STAGE=ARG` forwards a raw argument to one stage.
- `--tracker sam3` switches the default tracking image/build target to
  the SAM3 service image.
