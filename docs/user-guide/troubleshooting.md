# Troubleshooting

Work through a failed run in this order. Do not skip to a later stage until
the earlier one succeeds.

1. Dry-run the cookbook (below). Fix anything the plan prints wrongly:
   missing images, wrong mounts, wrong endpoint URLs.
2. If only one stage looks wrong, run that service alone from
   [Services](services/README.md).
3. Run the full cookbook with `--container-user auto` and
   `--container-env NVIDIA_API_KEY` (or `GEMINI_API_KEY`).
4. Inspect `<out_dir>/data/<entry-id>/sidecars/pipeline_state.json` and the
   scene folders in [Experiment Output Layout](experiment-output-layout.md).

If you changed Auto-Labeling source and the failure started after that edit, run the
contributor checks in [Local Development](../developer/local-development.md)
before chasing cookbook behavior.

## Dry Run First

A dry run validates cookbook loading and shows stage order, images, mounts,
endpoint URLs, model-cache paths, and forwarded arguments without running
models:

```bash
make run SCRIPT=workflow-runner:main \
  ARGS='--cookbook-file <cookbook.local.yaml> --container-dry-run'
```

## Missing Container Image

Symptom: Docker or Podman reports that a stage image does not exist.

```bash
make run SCRIPT=workflow-runner:main \
  ARGS='--cookbook-file <cookbook.local.yaml> --container-ensure-images --container-dry-run'
```

Remove `--container-dry-run` after reviewing the plan. Use
`--container-build-images` when every selected image must be rebuilt.

## Output Permission Errors

Symptom: one stage creates files that a later stage cannot update.

```bash
make run SCRIPT=workflow-runner:main \
  ARGS='--cookbook-file <cookbook.local.yaml> --container-user auto'
```

## Endpoint Is Unreachable

The runner uses host networking by default. On Linux, stage containers can
reach a model server on the same machine through
`http://localhost:<port>/v1`. If `--container-network` selects another network,
use a hostname or address reachable from that network.

Check:

1. The endpoint URL and served model name match the running server.
2. The endpoint responds from the selected container network.
3. The required API-key variable name is passed with `--container-env`.
4. The service timeout and retry settings are appropriate for model latency.

Do not print API key values while diagnosing endpoint access. See
[VLM and LLM Endpoints](vlm-llm-endpoints.md).

## Checkpoint Or GPU Failure

Check:

- The selected image flavor matches the backend, such as SAM3 versus RF-DETR.
- Model-cache and explicit checkpoint mounts exist and are mounted read-only.
- `--container-gpus` and service GPU arguments select available devices.
- Checkpoint download is enabled only when outbound access is permitted.

See [Model Provisioning](model-provisioning.md) and the owning service page.

## Scene Completes But Outputs Are Missing

The workflow runner defaults to `--policy fail`, which stops after a failed
stage container. To keep going so you can inspect a partial scene, rerun with
`--policy warn`. A successful container may still produce a degraded scene if
a stage treated empty output as a warning.

Inspect:

- `<data_path>/logs/workflow_runner.jsonl`
- `<data_path>/sidecars/pipeline_state.json`
- The selected cookbook's expected output list
- Container stdout and stderr for the original model or media error

Do not infer completeness from the process exit code alone.

## Media Decode Failure

Supported codecs are H.264, VP9, and MPEG-4 Part 2. H.264 requires NVIDIA
CUVID. If a GPU decoder rejects H.264, transcode the test asset to VP9 or
MPEG-4 Part 2 rather than enabling unapproved codec libraries in the image.
See [Media Policy](installation.md#media-policy).

## Common Failure Classes

| Symptom | First check |
| --- | --- |
| missing stage image | `--container-ensure-images` or `make build IMAGE=...` |
| permission denied in output tree | `--container-user auto` |
| endpoint auth or reachability error | URL, served model name, and `--container-env` |
| missing checkpoint | model cache path, explicit mount, and stage image flavor |
| missing output despite zero exit code | `sidecars/pipeline_state.json` and the cookbook contract |

To turn off telemetry noise in logs, set `OTEL_SDK_DISABLED=true`.
