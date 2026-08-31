# Detection and Tracking

This task runs object detection and multi-object tracking over image or video
scenes. It emits DAFT `contextual/objects.json` and
`contextual/instances.json`, joined by `track_id`.

Detection and tracking remain one task because two-stage backends associate
detections frame by frame, while SAM3 performs both operations in one model
pass. Splitting the stages would duplicate decoding and weaken the shared
artifact contract.

## Backends

| Kind | Family | Purpose |
| --- | --- | --- |
| `rfdetr-bytetrack` | two-stage | RF-DETR with ByteTrack |
| `rfdetr-boosttrack` | two-stage | RF-DETR with BoostTrack |
| `sam3` | unified | Text-prompted SAM3 tracking |
| `stub` | test | Deterministic model-free validation |

`rfdetr-deepocsort` is reserved but intentionally unregistered. Production
backends keep heavy imports lazy so config parsing and unit tests do not
require their model runtimes.

New backends implement `Tracker` under `detection_and_tracking/backends/` and
register a builder. The task and service contracts remain unchanged.

## Runtime Contract

RF-DETR resolves its checkpoint from `RFDETR_MODEL_PATH` or
`<model_cache_path>/rfdetr/rf-detr-base.pth`. SAM3 resolves weights from
`SAM3_MODEL_PATH` or `<model_cache_path>/sam3`; it requires CUDA and never
downloads weights implicitly.

Supported video codecs are H.264, VP9, and MPEG-4 Part 2. Decode and encode
policy is shared with super resolution through `core.media`.

When super resolution succeeds, detection consumes the enhanced active media;
otherwise it uses the original source. Detection records task-owned state at
`pipeline_state.task_artifacts["detection_and_tracking"]`.

## Outputs

- `contextual/objects.json` contains per-frame boxes and optional contours.
- `contextual/instances.json` contains track summaries.
- `sidecars/rfdetr/` contains requested RF-DETR overlays and sampled frames.
- `sidecars/sam3/` contains requested SAM3 annotated videos.
- Optional PAS crop extraction writes `sidecars/tracks/crops/`.

Detection does not write DAFT task annotations. See the
[Artifact Contract](../../../docs/developer/architecture/artifact-contract.md) for
scene ownership and reuse rules.

## Development

```bash
make run SCRIPT=detection-and-tracking-service:main ARGS='--help'
make build IMAGE=detection-and-tracking-service:rfdetr
make build IMAGE=detection-and-tracking-service:sam3
```

Use the service CLI `--help` output for backend-specific thresholds, overlay
controls, and SAM3 conditioning options.
