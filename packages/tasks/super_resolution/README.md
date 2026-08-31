# Super Resolution

This task provides reusable super-resolution logic for Auto-Labeling. The runnable CLI
and container live in `services/super_resolution_service/`.

## Architecture

`SuperResolutionTask` receives a `DataEntry`, creates the canonical scene
skeleton, delegates model work to a registered resolver, and records the result
in `ScenePipelineState`.

The production `seedvr2` resolver remains a subprocess backend because SeedVR2
runs through `torchrun` and can use multiple GPUs. Model-runtime details are
isolated behind the resolver contract:

```python
Resolver.run(media_path: Path, scene_paths: ScenePaths) -> SrResult
```

## Data Flow

1. Probe the input and apply the configured resolution policy.
2. Skip inputs that already meet the configured minimum resolution.
3. Run the selected resolver when enhancement is required.
4. Write enhanced media to `sidecars/sr_output.<ext>`.
5. Record success or skip state in `sidecars/pipeline_state.json`.
6. Return successful output so the pipeline can promote it to
   `sidecars/active.<ext>`.

Downstream stages use active enhanced media when available and otherwise retain
the original source.

## Runtime Contract

SeedVR2 requires CUDA. It accepts MP4, MOV, and WebM video plus JPEG, PNG, and
WebP images. Supported video codecs are H.264, VP9, and MPEG-4 Part 2. Shared
decode and encode behavior comes from `core.media`.

Model-cache precedence is:

1. `MODEL_CACHE_PATH`
2. `SeedVR2Config.model_cache_path`
3. `/models`
4. `<cwd>/ckpts`

Expected checkpoints live under `<cache>/seedvr2/`:

```text
ema_vae.pth
seedvr2_ema_3b.pth
seedvr2_ema_7b.pth  # optional 7B variant
```

Checkpoint download is disabled by default. See
[Model Provisioning](../../../docs/user-guide/model-provisioning.md) for the product
layout and download policy.

## Development

```bash
make run SCRIPT=super-resolution-service:main ARGS='--help'
make build IMAGE=super-resolution-service:build
make lint-check
make mypy
make test
```

The service CLI `--help` output is the authoritative runtime argument
reference. Run the three quality gates before submitting changes.
