# Super Resolution

## What it does

The super resolution service upscales low-resolution video before the rest of
the pipeline runs, so that downstream detection, captioning, and QA stages see
sharper frames. It uses **SeedVR2**, a GPU video super-resolution model.

## Before you start

- A CUDA-capable GPU. Inference always requires a GPU; there is no CPU
  fallback.
- The SeedVR2 checkpoint files, either pre-downloaded (recommended) or fetched
  automatically at startup. See [Model Provisioning](../model-provisioning.md)
  for the download steps.
- Input video must be **H.264**, **VP9**, or **MPEG-4 Part 2**. H.264 decode
  requires the NVIDIA CUVID hardware decoder; if it is unavailable, use a VP9
  or MPEG-4 Part 2 file instead. This is a fixed repository-wide media policy,
  not a per-service option.

## Step by step

### 1. Smoke-test the service (no GPU, no checkpoint, no media required)

```bash
make run SCRIPT=super-resolution-service:main \
  ARGS='--disabled --input-file payloads/simple.jsonl'
```

`--disabled` exits after argument parsing so you can confirm the service is
wired up correctly before you need real hardware.

### 2. Check the full option list

```bash
make run SCRIPT=super-resolution-service:main ARGS='--help'
```

### 3. Run against real media

Write a `DataEntry` JSONL manifest pointing at your video:

```jsonl
{"id": "clip-001", "media_path": "/absolute/path/to/clip.mp4", "data_path": "/absolute/path/to/output/clip-001"}
```

Then run:

```bash
make run SCRIPT=super-resolution-service:main \
  ARGS='--input-file /absolute/path/to/manifest.jsonl --gpu-ids 0 --model-cache-path <model-cache>'
```

Replace `<model-cache>` with the directory that holds `seedvr2/` from
[Model Provisioning](../model-provisioning.md). If you have not downloaded the
checkpoint yet, you can instead add `--allow-checkpoint-download` to fetch it
on first run — this requires outbound network access and is not recommended
for offline or locked-down environments.

### 4. (Optional) Skip already-sharp media

If your dataset mixes low- and high-resolution clips, enable the resolution
gate so the service only upscales media that actually needs it:

```bash
make run SCRIPT=super-resolution-service:main \
  ARGS='--input-file /absolute/path/to/manifest.jsonl \
        --resolution-policy auto \
        --min-input-short-side 720 --min-input-long-side 1280 \
        --gpu-ids 0 --model-cache-path <model-cache>'
```

Clips that already meet both thresholds are recorded as skipped, and later
stages continue with the original media.

## Verify it worked

- The command exits `0` with no traceback.
- `<data_path>/sidecars/sr_output.mp4` exists.
- Its resolution is larger than the input's:

  ```bash
  ffprobe -v error -select_streams v:0 \
    -show_entries stream=codec_name,width,height \
    -of default=noprint_wrappers=1 \
    <data_path>/sidecars/sr_output.mp4
  ```

  `codec_name` reports `vp9` — every generated video in this repo is encoded
  as VP9, regardless of the input codec.
- `<data_path>/sidecars/active.mp4` now points at the enhanced media so
  downstream stages use it automatically.

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| Exits immediately with a codec error | Input is not H.264, VP9, or MPEG-4 Part 2 — re-encode or pick a different source clip |
| "CUVID decoder unavailable" on an H.264 input | The host has no working NVIDIA CUVID hardware decoder; convert to VP9 or MPEG-4 Part 2 instead of falling back to software H.264 decode, which is not permitted |
| Checkpoint not found | Point `--model-cache-path` at the directory containing `seedvr2/`, or pass `--allow-checkpoint-download` |
| Out of GPU memory | Lower resolution/duration of the input clip, or free the GPU of other workloads before running |

For error output you don't recognize, see
[Troubleshooting](../troubleshooting.md).

## Full argument reference

[`services/super_resolution_service/README.md`](../../../services/super_resolution_service/README.md)

## Next

[Detection and Tracking](detection-and-tracking.md) is the usual next stage in
a video pipeline.
