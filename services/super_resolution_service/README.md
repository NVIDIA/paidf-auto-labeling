# Super Resolution Service

Standalone service and container wrapper for `packages/tasks/super_resolution`.
The service owns CLI parsing, Docker image registration, and runtime packaging;
the task package owns the reusable `SuperResolutionTask` implementation.

```bash
make run SCRIPT=super-resolution-service:main ARGS='--help'
make run SCRIPT=super-resolution-service:main ARGS='--disabled --input-file payloads/simple.jsonl'
make build IMAGE=super-resolution-service:build
```

Running without `ARGS` selects the same disabled simple-payload preset in
non-interactive runs, or offers it from the workspace runner in an interactive
terminal. It validates service input parsing without requiring SeedVR2, GPUs,
checkpoints, or a real media file.

For a real SeedVR2 run, use a JSONL file whose `media_path` points at real media
and remove `--disabled`. The service image bakes the pinned SeedVR checkout into
`/opt/seedvr`, so normal users do not need to mount SeedVR source or pass
`--seedvr-root`:

```bash
make run SCRIPT=super-resolution-service:main ARGS='--input-file /path/to/sr-input.jsonl --gpu-ids 0 --model-cache-path /path/to/model-cache'
```

The container accepts `.mp4`, `.mov`, and `.webm` video suffixes, plus `.jpg`,
`.jpeg`, `.png`, and `.webp` image suffixes. Video streams must be H.264, VP9, or MPEG-4
Part 2. H.264 always uses NVIDIA CUVID. If CUVID cannot decode an H.264 input, use a VP9 or
MPEG-4 Part 2 input instead; VP9 decoding prefers NVIDIA CUVID and can fall back to FFmpeg's
native software decoder. MPEG-4 Part 2 uses FFmpeg's native software decoder. These fallbacks
apply only to decoding: SeedVR2
inference still requires a CUDA-capable GPU. All enhanced videos are encoded by
`libvpx-vp9` into `sr_output.mp4`.

For mixed-resolution datasets, enable the per-media resolution gate:

```bash
make run SCRIPT=super-resolution-service:main \
  ARGS='--input-file /path/to/sr-input.jsonl --resolution-policy auto --min-input-short-side 720 --min-input-long-side 1280 --gpu-ids 0 --model-cache-path /path/to/model-cache'
```

With `--resolution-policy auto`, the service probes each input and runs SeedVR2
only when the short side is below `--min-input-short-side` or the long side is
below `--min-input-long-side`. Inputs that already meet both thresholds are
recorded as skipped and downstream stages continue with the existing media.
The default policy is `always`, which preserves the previous behavior.

Example input:

```jsonl
{"id": "clip-001", "media_path": "/experiment/source/clip-001.mp4", "data_path": "/experiment/data/clip-001"}
```

All SR experiments should follow the workspace experiment layout:

```text
<experiment-root>/
  input.jsonl
  source/
    <entry-id>.<ext>
  logs/
    super_resolution_service.log
  data/
    <entry-id>/
      sidecars/
        raw.<ext>
        active.mp4                # enhanced videos; image suffixes are preserved
        sr_output.mp4             # VP9 video; image suffixes are preserved
        pipeline_state.json
        logs/sr.log
```

The full convention is documented in
[Experiment Output Layout](../../docs/user-guide/experiment-output-layout.md).

The real run requires valid media and GPU access. Operators can either
pre-populate SeedVR2 checkpoints under `<model-cache>/seedvr2` or run the
service with `--allow-checkpoint-download` to fetch the required checkpoint
files at startup. Download mode stores files in the same
`<model-cache>/seedvr2` directory, requires outbound network access to the
configured Hugging Face repositories, and still requires GPU access for the
actual SeedVR2 run. The Docker image can be built with:

```bash
make build IMAGE=super-resolution-service:build
```

The image installs `super-resolution[seedvr2]`, bakes a pinned SeedVR checkout
into `/opt/seedvr`, copies the legacy color-fix helper into SeedVR, and builds
from the NGC PyTorch 26.07 base image, which provides CUDA, PyTorch,
TorchVision, and `torchrun` on both amd64 and arm64. FlashAttention is installed
from source into the service venv during the image build; the `ninja`-based
compile uses up to eight jobs by default. The image does not include vLLM.
At runtime the resolver prepares
`<model-cache>/seedvr_runtime` and uses it as the SeedVR working directory,
matching the original pseudo-labeling SR container behavior.
