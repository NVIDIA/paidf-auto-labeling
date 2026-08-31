# Visual Attribute Search Cookbooks

Visual Attribute Search is the cookbook family for turning image or video
evidence into searchable visual attributes and retrieval queries. The internal
stage key is still `person_attribute_search`, and it still runs the PAS-only
`event-and-person-attribute-search-service`: that service consumes existing
attributes or upstream sidecars and generates search artifacts. It does not
caption media, run Visual QA, copy crops, or create tracking data.

Copy a config to a gitignored `*.local.yaml`, replace its placeholder paths and
model settings, and inspect the generated plan before executing:

```bash
CONFIG=cookbooks/visual_attribute_search/configs/<config>.local.yaml
ARGS="--cookbook-file ${CONFIG} --container-dry-run" \
  make run SCRIPT=workflow-runner:main
```

## Contract summary

| Config | Input | Node flow | Model endpoints |
|---|---|---|---|
| `pipeline_image_attributes_pas.yaml` | One attribute JSON; the same file is the single `media_path` | `person_attribute_search` | LLM only |
| `pipeline_image_multiview_pas.yaml` | One representative image plus one local directory containing all views of one identity | `captioning -> visual_qa -> person_attribute_search` | VLM for caption/VQA; LLM for PAS queries |
| `pipeline_video_pas_reasoning.yaml` | One video or a video directory | `detection_and_tracking -> captioning -> event_verification_visual_qa -> reasoning -> person_attribute_visual_qa -> person_attribute_search -> training_export` | VLM and LLM |
| `pipeline_video_epas.yaml` | One video or a video directory | `detection_and_tracking -> captioning -> anomaly_visual_qa -> person_attribute_visual_qa -> person_attribute_search` | VLM and LLM |

The two video configs intentionally execute `visual_qa` twice. Their node IDs,
question banks, output sidecars, and pipeline-state artifact keys are distinct,
so the second pass cannot overwrite the first.

## Image Attribute Search

`pipeline_image_attributes_pas.yaml` never opens media. Keep exactly one data
entry and point both `data[0].inputs.media_path` and `--attribute-json` at the
same JSON file. The runner mounts that file's parent and the synonymous-query
prompt. No VLM endpoint, image directory, detector, or model cache is required.

Expected outputs under the scene's `sidecars/person_attribute_search/` are
`bundle_attributes.json`, `bundle_queries.json`, and, when enabled,
`bundle_hitl.json`.

```bash
make run SCRIPT=workflow-runner:main \
  ARGS='--cookbook-file cookbooks/visual_attribute_search/configs/pipeline_image_attributes_pas.yaml --container-dry-run'
```

## Multi-View Image Attribute Search

`pipeline_image_multiview_pas.yaml` treats the image group as exactly one
identity. `media_path` must remain a representative file so directory expansion
does not create one scene per view. Both producer nodes receive
`--image-group-dir` and `--max-group-images 0`, which recursively selects every
supported image in deterministic order and sends all selected views in one
request per node.

The group directory and person-attribute question bank are read-only mounts.
Captioning writes `sidecars/captioning/image_caption.json`; Visual QA writes
`sidecars/visual_qa/items.json` and `windows.normalized.json`; PAS consumes those
artifacts and writes `attributes.json`, `queries.json`, and optional `hitl.json`.

Person images are **not shipped in git**. Download
[RSTPReid](https://github.com/NjtechCVLab/RSTPReid-Dataset)
([Google Drive archive](https://drive.google.com/file/d/1HTeDZUVrZr6nL56ZlkYBNqjSWh3IGV2X/view?usp=sharing))
and copy one identity into
`data/input_media/images/image_attribute_augmentations/<person_id>_RSTP/`
before a real run. The tracked config expects `00000_RSTP` (RSTPReid person
`0000`) with representative file `0000_c14_0032.jpg`. See
[image_attribute_augmentations/README.md](../../data/input_media/images/image_attribute_augmentations/README.md).

```bash
make run SCRIPT=workflow-runner:main \
  ARGS='--cookbook-file cookbooks/visual_attribute_search/configs/pipeline_image_multiview_pas.yaml --container-dry-run'
```

## Video Attribute Search With Reasoning

`pipeline_video_pas_reasoning.yaml` mounts the input/output paths, model cache,
SAM3 weights, event-verification bank, person-attribute bank, and the cookbook
itself as the reasoning/PAS config. The VLM serves captioning and both Visual QA
passes; the LLM serves event verification, reasoning, and PAS query generation.
The tracked `media_path` is a warehouse clip that is not in git; stage it from
NGC first
([warehouse README](../../data/input_media/videos/warehouse_safety/README.md)).

The event pass writes namespaced raw/normalized evidence plus DAFT `task/` QA
files with reasoning traces. Reasoning adds its configured DAFT artifacts. The
person pass writes `sidecars/visual_qa_per_track/{items.json,windows.normalized.json}`.
PAS consumes those files with `sidecars/detection_and_tracking/tracks.json` and
`sidecars/captioning/video_captions.json`, then writes `pas.json`,
`chunk_queries.json`, and the configured contextual mirror. Training export
writes the configured TAO-VL-Reason dataset outside the scene directory.

```bash
make run SCRIPT=workflow-runner:main \
  ARGS='--cookbook-file cookbooks/visual_attribute_search/configs/pipeline_video_pas_reasoning.yaml --container-dry-run'
```

## Video Attribute Search Without Export

`pipeline_video_epas.yaml` has the same media, output, cache, SAM3, question-bank,
VLM, and LLM mount/endpoint requirements as the reasoning flow, but omits the
reasoning and export nodes. Its anomaly pass writes
`sidecars/visual_qa_anomaly/items.json`; its person pass writes
`sidecars/visual_qa_per_track/{items.json,windows.normalized.json}`. PAS consumes
those artifacts together with tracks and video captions to emit `pas.json`,
`chunk_queries.json`, anomaly/caption query buckets, `pas_anomaly.json`, and
`contextual/person_attributes.json`.

```bash
make run SCRIPT=workflow-runner:main \
  ARGS='--cookbook-file cookbooks/visual_attribute_search/configs/pipeline_video_epas.yaml --container-dry-run'
```

For a real video run, replace the checked-in absolute SAM3 example mount with a
valid host path, keep it read-only at `/models/sam3`, and forward only the name
of any credential environment variable. Never commit credentials or local
machine paths to a cookbook.
