# Operations: Workflow Runner

`workflow-runner` reads a cookbook, resolves paths and stage arguments, then
runs the selected stage containers in order on your machine.

```bash
make run SCRIPT=workflow-runner:main ARGS='--help'
```

## Standard Run Pattern

1. Copy a tracked cookbook to `*.local.yaml`
2. Edit media, output, endpoints, model cache, and checkpoint mounts
3. Run `--container-dry-run`
4. Run the real workflow with `--container-user auto`
5. Inspect the scene outputs and pipeline state

```bash
CONFIG=<cookbook.local.yaml>
make run SCRIPT=workflow-runner:main \
  ARGS="--cookbook-file ${CONFIG} --container-dry-run"

export NVIDIA_API_KEY="<your-key>"
make run SCRIPT=workflow-runner:main \
  ARGS="--cookbook-file ${CONFIG} --container-user auto \
        --container-ensure-images --container-env NVIDIA_API_KEY"
```

`--container-user auto` keeps files writable across stages that may have
different image defaults. `--container-ensure-images` builds only missing
stage images; use `--container-build-images` to rebuild every enabled image.
`--container-env NVIDIA_API_KEY` forwards the key from your shell. Use
`GEMINI_API_KEY` instead when that is the provider on
[VLM and LLM Endpoints](vlm-llm-endpoints.md).

## Cookbook Shape

Most shipped cookbooks follow this shape:

```yaml
pipeline: video | image
workflow:
  nodes:
    <node_id>:
      stage: <stage_name>
      needs: [<upstream_node>]
      args: [--flag, value]
runtime:
  model_cache_path: <path>
  gpu_ids: all
container:
  user: auto
  images: {}
  env: []
  mounts: []
data:
  - id: <optional-scene-id>
    inputs:
      media_path: <path-or-uri>
    output:
      out_dir: <output-root>
endpoints:
  vlm: {url: ..., model: ...}
  llm: {url: ..., model: ...}
```

Copy tracked configs to `*.local.yaml` before editing machine-specific paths.
Keep `container.env` as variable names, not secret values. For the multiview
PAS flow, keep `media_path` as a file so the runner does not expand a directory
into one scene per image.

Which cookbook to copy: [Samples and Cookbooks](samples-and-cookbooks.md).

## Useful Flags

| Flag | When to use it |
| --- | --- |
| `--cookbook-file` | Load a scenario cookbook |
| `--stages ...` | Enable only selected stages |
| `--container-dry-run` | Print the stage plan without execution |
| `--container-user auto` | Run stage containers as the invoking user |
| `--container-env NAME` | Pass a host env var name through to every stage |
| `--container-mount HOST[:CONTAINER[:ro\|rw]]` | Add a bind mount to every stage |
| `--container-ensure-images` | Build only missing enabled stage images |
| `--container-build-images` | Rebuild every enabled stage image |
| `--container-network` | Override the default host network behavior |
| `--dev-data-root PATH` | Copy the source scene before writing, so samples are not mutated |
| `--stage-arg STAGE=ARG` | Add one extra flag to a single stage for this run |
| `--policy warn` | Continue after a failed stage so you can inspect a partial scene |

Common forwarded model options include `--model-cache-path`, `--gpu-ids`,
`--vlm-endpoint-url`, `--llm-endpoint-url`, `--question-bank-file`,
`--reasoning-config-file`, `--pas-config-file`, and
`--training-export-format` / `--training-export-dir`.

Typical env vars to pass with `--container-env`: `NVIDIA_API_KEY` or
`GEMINI_API_KEY` for endpoints, and `MSC_CONFIG` or
`MULTISTORAGECLIENT_CONFIGURATION` for remote paths.

`--dev-data-root` copies each source scene before a stage writes to it. Use it
when you want to keep an existing grounding `out_dir` unchanged while running
the referring-only config. See
[Getting Started](getting-started.md#path-d--image-spatial-grounding).

## Local Versus Remote Paths

- Local `media_path` and `data_path` values are mounted into every stage
  container.
- URI-like values such as `s3://...` are left unchanged and require
  [Remote Storage](remote-storage.md) configuration inside every stage
  container.

## Stage Images

Build a registered image from the repository root:

```bash
make build IMAGE=<package>:<target>
```

Registered names:

- `captioning-service:main`
- `detection-and-tracking-service:rfdetr`
- `detection-and-tracking-service:sam3`
- `event-and-person-attribute-search-service:build`
- `grounding-2d-service:main`
- `grounding-2d-service:sam3`
- `reasoning-service:build`
- `referring-expressions-service:main`
- `super-resolution-service:build`
- `training-export-service:build`
- `visual-qa-service:build`
- `workflow-runner:workflow-runner`

GPU images need a compatible NVIDIA driver and container GPU runtime. Mount
model checkpoints read-only when possible. How images are registered in
package metadata:
[Local Development](../developer/local-development.md#image-registration).

## Training Export

The training export stage can be driven from the workflow runner:

```bash
make run SCRIPT=workflow-runner:main \
  ARGS='--input-file payloads/simple.jsonl \
        --training-export-format tao-vl-reason-v1.0 \
        --training-export-dir output/training'
```

Cookbooks can also configure `training_export:` as a terminal stage.

## Scene Logging

For local non-dry runs, the runner appends lightweight execution records to
`<data_path>/logs/workflow_runner.jsonl`. Use that together with
`sidecars/pipeline_state.json` when diagnosing stage order, outputs, or
container return codes.
