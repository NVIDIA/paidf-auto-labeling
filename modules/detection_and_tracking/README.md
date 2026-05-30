<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Auto-Labeling: RFDETR + Tracking (Deep-OC-SORT / ByteTrack / BoostTrack)

A media processing pipeline that combines **RF-DETR** for object detection and multi-object tracking on video or image inputs. It generates structured annotations in JSON format for downstream training.

## Features

- **RF-DETR** detection with COCO-80 classes
- Multiple tracker options: `bytetrack`, `deepocsort`, `boosttrack`
- Optional **ReID embeddings** for BoostTrack (default weights: `ckpts/reid/clip_vehicleid.pt`, downloaded automatically)
- Structured output with `instances.json` and `objects.json`
- Optional visualization overlays and frames

## Usage

Run via the config-driven pipeline CLI (inside Docker):

The example below uses relative paths, so run it from the repository root; otherwise use absolute paths for both the CLI and config file.

```bash
uv run python modules/cli.py --config configs/pipeline_example.yaml \
  super_resolution.enabled=false vlm_json.enabled=false mcq_generation.enabled=false \
  detection_and_tracking.enabled=true \
  data.0.inputs.video_path="/path/to/video_or_image" \
  data.0.output.out_dir="/path/to/output"
```

### Key config knobs

| Config key | Default | Description |
|------------|---------|-------------|
| `detection_and_tracking.tracker` | `boosttrack` | Tracker backend: `bytetrack`, `deepocsort`, `boosttrack` |
| `detection_and_tracking.threshold` | `0.2` | Detection confidence threshold |
| `detection_and_tracking.iou_threshold` | `0.3` | IoU threshold for tracking |
| `detection_and_tracking.save_vis` | `false` | Save visualization frames |
| `detection_and_tracking.save_video` | `true` | Save annotated overlay media |
| `detection_and_tracking.min_track_frames` | `5` | Minimum frames for a valid track |
| `detection_and_tracking.use_reid` | `true` | Enable ReID embeddings (**BoostTrack only**) |
| `detection_and_tracking.reid_weights` | `""` | ReID weights path; auto-downloads if empty and `use_reid=true` |
| `detection_and_tracking.save_video_red_id` | `true` | Extra red-id overlay media (uniform red boxes + id). Not a ReID indicator |

## Output Structure

```text
<out_dir>/
├── contextual/
│   ├── instances.json              # Object definitions
│   └── objects.json                # Per-frame annotations
└── sidecars/
    ├── <track_stem>_detection.<ext>
    ├── <track_stem>_tracking.<ext>
    └── <track_stem>_tracking_red_id.<ext>
```

`<track_stem>` is the stem of the media passed into tracking. In the full pipeline this is usually `sr_output` when SR ran, or the original input stem when SR is disabled.
