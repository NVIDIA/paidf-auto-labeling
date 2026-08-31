# Detection And Tracking Service

Standalone service and container wrapper for
`packages/tasks/detection_and_tracking`. The service owns CLI parsing, Docker
image registration, and runtime packaging; the task package owns detection and
tracking behavior.

```bash
make run SCRIPT=detection-and-tracking-service:main
make run SCRIPT=detection-and-tracking-service:main ARGS='--help'
make run SCRIPT=detection-and-tracking-service:main ARGS='--tracker stub --input-file payloads/simple.jsonl'
make run SCRIPT=detection-and-tracking-service:main \
  ARGS='--tracker rfdetr-boosttrack --input-file <input.jsonl>'
make run SCRIPT=detection-and-tracking-service:main \
  ARGS='--tracker sam3 --sam3-prompts person vehicle --input-file <input.jsonl>'
make run SCRIPT=detection-and-tracking-service:main \
  ARGS='--disabled --input-file <input.jsonl>'
make build IMAGE=detection-and-tracking-service:rfdetr
make build IMAGE=detection-and-tracking-service:sam3
```

GPU flavors build from `services/detection_and_tracking_service/docker/Dockerfile` (same layout
as `grounding-2d-service:sam3`). Each service gets an independent image tag so
orchestration platforms can scale stages without sibling-image coupling.

Running without `ARGS` uses the default no-model command from this package's
`pyproject.toml`: `--tracker stub --input-file payloads/simple.jsonl`. That
exercises input loading, scene creation, and DAFT artifact writing without
requiring RF-DETR/SAM3 checkpoints or GPU access.

## Inputs

Each input record is a `DataEntry` JSON object with `media_path` and
`data_path`. `media_path` points to the image or video to annotate. `data_path`
is the scene output directory where the service writes DAFT artifacts.

Video streams must be H.264, VP9, or MPEG-4 Part 2. H.264 requires the NVIDIA CUVID decoder;
VP9 prefers CUVID and can fall back to software decoding, while MPEG-4 Part 2 uses FFmpeg's
native software decoder. Generated RF-DETR and
SAM3 videos are always encoded with `libvpx-vp9` into MP4 containers.

## CLI Options

General runtime options:

| Option | Default | Behavior |
|---|---:|---|
| `--tracker` | `rfdetr-boosttrack` | Selects the backend. Production choices are `rfdetr-boosttrack`, `rfdetr-bytetrack`, and `sam3`; `stub` supports model-free smoke tests. |
| `--disabled` | off | Exits after argument parsing. This is useful for smoke-testing service wiring without loading inputs or models. |
| `--model-cache-path` | backend default | Root directory for model assets. RF-DETR expects `rfdetr/rf-detr-base.pth`; SAM3 Transformers expects an HF-format `sam3/` directory; native SAM 3.1 expects `sam3/sam3.1_multiplex.pt` unless `SAM3_MODEL_PATH` overrides it. |
| `--gpu-ids` | `all` | Selects visible GPUs for model runtime, for example `0` or `0,1`. |
| `--classes` | empty | RF-DETR class filter. SAM3 uses these as prompts when `--sam3-prompts` is omitted. |
| `--copy-media` | off | Copies source media into the scene `raw/` folder instead of writing a symlink. |

RF-DETR and tracking options:

| Option | Default | Behavior |
|---|---:|---|
| `--threshold` | `0.2` | Detection confidence threshold passed to RF-DETR. Matches legacy pseudo-labeling. |
| `--iou-threshold` | `0.3` | IoU association threshold for tracker matching when supported by the selected tracker. |
| `--per-class` / `--no-per-class` | on | Runs tracker association independently per detected class. Default on matches legacy; disable with `--no-per-class`. |
| `--min-hits` | `3` | Minimum consecutive hits before a BoostTrack track is emitted. |
| `--max-age` | `60` | Maximum missed frames before a BoostTrack track is removed. |
| `--min-track-frames` | `5` | Drops RF-DETR tracks shorter than this many frames from the final DAFT output. Matches legacy. |
| `--bbox-expansion-ratio` | `0.1` | Expands RF-DETR tight boxes by this fraction when writing `bounding_box_2d_loose`. Matches legacy. |
| `--allow-model-download` | off | Allows RF-DETR to download its checkpoint when it is not present locally. Keep this off in offline or locked-down runtime environments. |

Diagnostic output options:

| Option | Default | Behavior |
|---|---:|---|
| `--save-video` | off | RF-DETR writes detection and tracking overlay videos under `sidecars/rfdetr/`. SAM3 writes its annotated video. |
| `--save-red-id-overlay` | off | RF-DETR writes a red-id tracking overlay. SAM3 treats this as a request for its id-label annotated video. |
| `--save-rgb` | off | RF-DETR writes sampled RGB frames under `sidecars/rfdetr/rgb/`. |

SAM3 options:

| Option | Default | Behavior |
|---|---:|---|
| `--sam3-prompts` | empty | Text prompts to track, for example `person vehicle`. Required unless `--classes` supplies prompts. |
| `--sam3-version` | `sam3` | Model family. `sam3` defaults to the Transformers runtime; `sam3.1` selects native Meta Object Multiplex. |
| `--sam3-runtime` | `auto` | Inference stack: `auto`, `transformers`, or `native`. `sam3.1` requires `native` (Transformers has no 3.1 multiplex integration). |
| `--sam3-tracking-mode` | `chunked` | `chunked` resets the video session every `--sam3-session-reset-s`; `continuous` keeps one session for the whole allowed clip. |
| `--sam3-target-fps` | `10.0` | Frame rate used when sampling video for SAM3 processing. |
| `--sam3-session-reset-s` | `10.0` | Chunk duration before SAM3 resets the video session (chunked mode only). |
| `--sam3-max-duration-s` | `30.0` | Maximum clip duration to process per input. `--sam3-max-clip-duration-s` is accepted as an alias. |
| `--sam3-write-annotated-video` | off | Writes a SAM3 mask-contour annotated video under `sidecars/sam3/`. |
| `--sam3-annotated-video-trails` | off | Draws object trails in the SAM3 annotated video. |
| `--sam3-annotated-video-label-style` | `id` | Overlay labels: `id` (`#0:3`), `name` (prompt), `track` (`person_T001`), or `none`. |
| `--sam3-annotated-video-mask-opacity` | `0` | Mask fill opacity for SAM3 annotated videos, from `0` to `100`. |
| `--sam3-write-masks` | off | Writes per-detection mask PNGs under `sidecars/sam3/masks/` for SoM-style inspection. |
| `--sam3-score-threshold-detection` | backend default | Overrides the SAM3 detection score threshold. |
| `--sam3-det-nms-thresh` | backend default | Overrides the SAM3 detection NMS threshold. |
| `--sam3-new-det-thresh` | backend default | Overrides the SAM3 new-object detection threshold. |
| `--sam3-fill-hole-area` | backend default | Overrides SAM3 mask hole filling area. |
| `--sam3-recondition-every-nth-frame` | backend default | Overrides how often SAM3 reconditions tracking state. |
| `--sam3-recondition-on-trk-masks` | backend default | Overrides whether SAM3 reconditions from tracked masks. Accepts optional boolean values such as `true` or `false`. |
| `--sam3-high-conf-thresh` | backend default | Overrides the SAM3 high-confidence threshold. |
| `--sam3-high-iou-thresh` | backend default | Overrides the SAM3 high-IoU threshold. |
| `--sam3-multiplex-count` | `16` | Objects per Object Multiplex bucket (native SAM 3.1 only). |
| `--sam3-max-num-objects` | `16` | Maximum tracked objects for native SAM 3.1. |
| `--sam3-compile` | off | Enable `torch.compile` for native SAM 3.1 when supported. |

### PAS crop extraction

Use `--extract-crops` when the Visual Attribute Search video flow
needs per-track person crops. The service writes a backend-independent
`sidecars/detection_and_tracking/tracks.json` seam and crops under
`sidecars/tracks/crops/` by default.

Use `--crop-classes`, `--crops-per-track`, `--crop-padding`, and
`--min-crop-size` to control sampling. `--min-detection-score` and
`--min-track-seconds` filter weak or short tracks before crops are written.
Use `--crop-format`, `--crop-subdir`, and `--tracks-sidecar` only when the
consumer is configured for matching custom paths.

## Outputs

The service writes `contextual/objects.json` and
`contextual/instances.json` under each `data_path`. Optional overlays and
diagnostic files are written under `sidecars/rfdetr/` or `sidecars/sam3/`.

## Containers

The image targets install the matching service extras, which forward to the
task extras: `detection-and-tracking[rfdetr]` or
`detection-and-tracking[sam3]`.

For `--sam3-version sam3.1`, the sam3 image installs Meta's native `sam3`
package by default (`INSTALL_NATIVE_SAM3=1`). Mount `sam3.1_multiplex.pt` under
`/models/sam3` or set `SAM3_MODEL_PATH` to the checkpoint file. SAM3 itself
always uses local weights; only VLM/LLM stages use endpoint URLs.

Use `--sam3-tracking-mode continuous` for unbroken tracks over the clip, and
`--sam3-annotated-video-label-style track` for `person_T001`-style overlays.

Both targets build the same restricted FFmpeg profile as superresolution with
GPL and nonfree components disabled. They rebuild PyAV against that FFmpeg and
install only source-built `opencv-python-headless`, with OpenCV's FFmpeg and
GStreamer backends disabled.

RF-DETR/BoostTrack and SAM3 require their matching runtime extras and model
assets; build or run the image target that matches the selected tracker.

The service images recognize `MODEL_CACHE_PATH`, `RFDETR_MODEL_PATH`, and
`SAM3_MODEL_PATH`. Prefer read-only checkpoint mounts and keep automatic model
downloads disabled in offline or controlled environments.
