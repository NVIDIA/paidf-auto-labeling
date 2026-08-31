# Detection and Tracking

## What it does

The detection and tracking service finds objects in an image or video and, for
video, tracks each object across frames so it keeps a stable ID over time. It
supports three interchangeable backends:

- **RF-DETR + BoostTrack** (`rfdetr-boosttrack`) — the default production
  backend.
- **RF-DETR + ByteTrack** (`rfdetr-bytetrack`) — an alternate tracker for
  RF-DETR detections.
- **SAM3** (`sam3`) — text-prompted detection and tracking ("track every
  `person` and `vehicle`"), including the native **SAM 3.1 Object Multiplex**
  runtime for higher track counts.

A model-free `stub` backend exists purely for wiring smoke tests.

## Before you start

- A CUDA-capable GPU for `rfdetr-*` or `sam3` backends.
- The matching checkpoint: `rfdetr/rf-detr-base.pth` for RF-DETR, or the SAM3
  weights for SAM3. See [Model Provisioning](../model-provisioning.md).
- Input video must be H.264 (via NVIDIA CUVID), VP9, or MPEG-4 Part 2 — see
  [Media Policy](../installation.md#media-policy).

## Step by step

### 1. Smoke-test with the model-free stub backend

```bash
make run SCRIPT=detection-and-tracking-service:main \
  ARGS='--tracker stub --input-file payloads/simple.jsonl'
```

This exercises input loading, scene creation, and DAFT artifact writing
without needing checkpoints or a GPU.

### 2. Check the full option list

```bash
make run SCRIPT=detection-and-tracking-service:main ARGS='--help'
```

### 3a. Run with RF-DETR + BoostTrack (default production backend)

```bash
make run SCRIPT=detection-and-tracking-service:main \
  ARGS='--tracker rfdetr-boosttrack --input-file <input.jsonl> --model-cache-path <model-cache> --gpu-ids 0'
```

### 3b. Run with SAM3 (text-prompted)

```bash
make run SCRIPT=detection-and-tracking-service:main \
  ARGS='--tracker sam3 --sam3-prompts person vehicle --input-file <input.jsonl> --model-cache-path <model-cache> --gpu-ids 0'
```

`--sam3-prompts` (or `--classes` if omitted) is required for SAM3 — it tells
the model what to look for. For the higher-capacity native runtime, add
`--sam3-version sam3.1 --sam3-runtime native` and mount the
`sam3.1_multiplex.pt` checkpoint.

### 4. (Optional) Write overlay video for visual review

Add `--save-video` to either command above to write an annotated overlay video
under `sidecars/rfdetr/` (RF-DETR) or `sidecars/sam3/` (SAM3), useful when you
want to eyeball the detections before trusting the pipeline output.

## Verify it worked

- The command exits `0` with no traceback.
- `<data_path>/contextual/objects.json` and
  `<data_path>/contextual/instances.json` exist and contain at least one
  detected object.
- If `--save-video` was set, the overlay video probes as VP9:

  ```bash
  ffprobe -v error -select_streams v:0 -show_entries stream=codec_name \
    -of default=noprint_wrappers=1 <overlay-video-path>
  ```

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| SAM3 run exits asking for prompts | Pass `--sam3-prompts` or `--classes` |
| RF-DETR tries to download a checkpoint | Point `--model-cache-path` at your local `rfdetr/` directory, or explicitly pass `--allow-model-download` if that is intentional |
| Zero detections on a clip you expect objects in | Lower `--threshold` (default `0.2`), or confirm your `--classes`/`--sam3-prompts` match what's actually in frame |
| Tracks disappear quickly | Increase `--max-age` (frames a track survives without a hit) or decrease `--min-track-frames` |

For error output you don't recognize, see
[Troubleshooting](../troubleshooting.md).

## Full argument reference

[`services/detection_and_tracking_service/README.md`](../../../services/detection_and_tracking_service/README.md)

## Next

[Captioning](captioning.md) or [2D Grounding](grounding-2d.md)/[Referring
Expressions](referring-expressions.md), depending on whether your workflow
needs video captions or per-box language.
