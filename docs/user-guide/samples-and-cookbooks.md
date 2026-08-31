# Samples and Cookbooks

Cookbooks are the main product examples in this repo. They define stage order,
mounts, endpoints, prompts, question banks, and output roots.

Traffic clips, warehouse clips, and intersection frames come from NGC, not
from this repository. Stage them before a real Path A/B/C/D run. Do not
commit secrets or machine-specific paths into tracked cookbook files.

Index: [Workflow Runner Cookbooks](../../cookbooks/README.md)

## Download VSS Sample Clips

The clips are the NGC resource
`nvidia/vss-developer/dev-profile-sample-data:3.2.0`. That archive mixes
traffic, warehouse, and unused videos in one folder. Stage the mapped files
into the Auto-Labeling sample directories below. Do not point a cookbook `media_path`
at the mixed extract.

Host `ffmpeg` and `ffprobe` are required for the chunk step. Traffic clips
are 10 seconds; warehouse clips are 30 seconds. Each loop caps at
`210 / clip_seconds` from the longest pack file, `warehouse_sample.mp4`.

### Install NGC CLI

NGC CLI version 4.10.0 or later is required to download this resource.
Authenticate with an NGC API key. External customers typically have no NGC
org or team; if the CLI prompts for those fields, leave them empty.

Download NGC CLI. ARM64 Linux:

```bash
curl -sLo "/tmp/ngccli.zip" \
  https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/4.10.0/files/ngccli_arm64.zip
```

AMD64 Linux:

```bash
curl -sLo "/tmp/ngccli.zip" \
  https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/4.10.0/files/ngccli_linux.zip
```

Install:

```bash
sudo mkdir -p /usr/local/bin
sudo unzip -qo /tmp/ngccli.zip -d /usr/local/lib
sudo chmod +x /usr/local/lib/ngc-cli/ngc
sudo ln -sfn /usr/local/lib/ngc-cli/ngc /usr/local/bin/ngc
ngc --version
```

Configure the CLI with your API key. Do not put the key on the command line.
For how to create a key, see NGC API keys in the NGC documentation.

```bash
ngc config set
```

NGC CLI downloads: https://ngc.nvidia.com/setup/installers/cli

NGC CLI documentation: https://docs.ngc.nvidia.com/cli/index.html

You can use NGC CLI as below, or download the same resource from the NGC UI.

```bash
ngc registry resource download-version \
  nvidia/vss-developer/dev-profile-sample-data:3.2.0

EXTRACT_DIR=./sample-data
mkdir -p "${EXTRACT_DIR}"
tar -xf dev-profile-sample-data_v3.2.0/dev-profile-sample-data.tar.gz \
  -C "${EXTRACT_DIR}"

rm -rf dev-profile-sample-data_v3.2.0
```

The extract layout is `sample-data/dev-profile-sample-data/*.mp4`.

### Chunk clips into the sample directories

Use copies, not symlinks. Keep the cookbook filenames below; recipes that
point at a directory read every media file in it.

```bash
SRC=./sample-data/dev-profile-sample-data
TRAFFIC_CLIP_SECS=10
WAREHOUSE_CLIP_SECS=30
TRAFFIC_MAX_CLIPS=$((210 / TRAFFIC_CLIP_SECS))
WAREHOUSE_MAX_CLIPS=$((210 / WAREHOUSE_CLIP_SECS))

mkdir -p data/input_media/videos/traffic_video_analytics \
  data/input_media/videos/warehouse_safety \
  data/input_media/images/traffic_intersection_frames

i=0
for src in "${SRC}/sample-sim-jaywalking.mp4" "${SRC}/sample-sim-traffic.mp4"
do
  start=0
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "${src}")
  dur=${dur%.*}
  while [ "${i}" -lt "${TRAFFIC_MAX_CLIPS}" ] && [ "${start}" -lt "${dur:-0}" ]; do
    dest=$(printf \
      'data/input_media/videos/traffic_video_analytics/traffic_sample_%03d.mp4' \
      "${i}")
    ffmpeg -y -ss "${start}" -t "${TRAFFIC_CLIP_SECS}" -i "${src}" -c copy \
      -avoid_negative_ts make_zero "${dest}"
    i=$((i + 1))
    start=$((start + TRAFFIC_CLIP_SECS))
  done
done

i=0
for src in \
  "${SRC}/warehouse_sample.mp4" \
  "${SRC}/sample-warehouse-ladder.mp4" \
  "${SRC}/warehouse_safety_0001.mp4" \
  "${SRC}/warehouse_safety_0002.mp4"
do
  start=0
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "${src}")
  dur=${dur%.*}
  while [ "${i}" -lt "${WAREHOUSE_MAX_CLIPS}" ] && [ "${start}" -lt "${dur:-0}" ]; do
    dest=$(printf \
      'data/input_media/videos/warehouse_safety/warehouse_safety_%04d.mp4' \
      "${i}")
    ffmpeg -y -ss "${start}" -t "${WAREHOUSE_CLIP_SECS}" -i "${src}" -c copy \
      -avoid_negative_ts make_zero "${dest}"
    i=$((i + 1))
    start=$((start + WAREHOUSE_CLIP_SECS))
  done
done

for mp4 in data/input_media/videos/traffic_video_analytics/traffic_sample_*.mp4
do
  base=$(basename "${mp4}" .mp4)
  ffmpeg -y -i "${mp4}" -frames:v 1 \
    "data/input_media/images/traffic_intersection_frames/${base}.jpg"
done
```

| Sample directory | NGC files to chunk |
| --- | --- |
| `data/input_media/videos/traffic_video_analytics/` | `sample-sim-jaywalking.mp4`, `sample-sim-traffic.mp4` |
| `data/input_media/videos/warehouse_safety/` | `warehouse_sample.mp4`, `sample-warehouse-ladder.mp4`, `warehouse_safety_0001.mp4`, `warehouse_safety_0002.mp4` |
| `data/input_media/images/traffic_intersection_frames/` | First frame of each staged traffic clip |

Leave these extract files unused: `sample-sim-box-conveyor.mp4`,
`sample-drone-bridge.mp4`.

Folder-level commands:
[traffic README](../../data/input_media/videos/traffic_video_analytics/README.md),
[warehouse README](../../data/input_media/videos/warehouse_safety/README.md),
[frames README](../../data/input_media/images/traffic_intersection_frames/README.md).

## Which Sample To Use

| Goal | Cookbook / sample | What you must change |
| --- | --- | --- |
| General video auto-labeling first run | `cookbooks/video_data_augmentation/configs/pipeline_video.yaml` | Stage NGC traffic clips first; copy to `pipeline_video.local.yaml` to set output path, model cache, SAM3 mount, endpoint URLs, or a different media path |
| Smart Spaces warehouse event reasoning | `cookbooks/smart_spaces/configs/pipeline_warehouse_event_reasoning.yaml` | Stage NGC warehouse clips first; copy to `*.local.yaml` to set output path, model cache, SAM3 mount, endpoint URLs, or a different warehouse clip |
| Image spatial grounding | `cookbooks/image_spatial_grounding/configs/pipeline_grounding.yaml` | Stage NGC traffic clips and extract frames first; copy to `*.local.yaml` to set output path, VLM endpoint, SAM3 mount, or different images |
| Full video visual attribute search + reasoning + training export | `cookbooks/visual_attribute_search/configs/pipeline_video_pas_reasoning.yaml` | Stage NGC warehouse clips first; copy to `*.local.yaml` to set output path, model cache, SAM3 mount, endpoint URLs, or a different media path |
| Video visual attribute search without export | `cookbooks/visual_attribute_search/configs/pipeline_video_epas.yaml` | Same as above |
| Attribute-only visual search | `cookbooks/visual_attribute_search/configs/pipeline_image_attributes_pas.yaml` | Replace the attribute JSON path in both `media_path` and `--attribute-json`; keep exactly one data entry |
| Multi-view visual attribute search | `cookbooks/visual_attribute_search/configs/pipeline_image_multiview_pas.yaml` | Download RSTPReid, copy one identity into `data/input_media/images/image_attribute_augmentations/`, then set both `--image-group-dir` paths and `media_path` together |
| Image spatial grounding, referring only | `cookbooks/image_spatial_grounding/configs/pipeline_referring.yaml` | Run `pipeline_grounding.yaml` first, then this config on the same grounding `out_dir` |

## Sample Media

| Sample | Path | Used by |
| --- | --- | --- |
| Traffic clips from NGC (not in git) | `data/input_media/videos/traffic_video_analytics/traffic_sample_000.mp4` | Video data augmentation |
| Warehouse clips from NGC (not in git) | `data/input_media/videos/warehouse_safety/warehouse_safety_0000.mp4` | Smart Spaces and video visual attribute search |
| Traffic-intersection frames (not in git) | `data/input_media/images/traffic_intersection_frames/` | Image spatial grounding |
| RSTPReid multi-view identity (not in git) | `data/input_media/images/image_attribute_augmentations/<id>_RSTP/` | Multi-view visual attribute search |

## Video Data Augmentation

This cookbook is the cleanest general video path:

`super_resolution -> detection_and_tracking -> captioning -> visual_qa -> reasoning`

Use it when you want to validate the standard media enhancement and annotation
flow before you add PAS or training export. The tracked config points at
`data/input_media/videos/traffic_video_analytics/` after you stage NGC clips
there.

## Smart Spaces

The warehouse event-reasoning cookbook is the clearest Smart Spaces path:

`detection_and_tracking -> captioning -> event_verification_visual_qa -> reasoning`

Use it when you want to verify warehouse safety or operational events and write
reasoning artifacts without also generating person-attribute search outputs or
training exports. The tracked config points at
`data/input_media/videos/warehouse_safety/` after you stage NGC clips there.

## Visual Attribute Search Cookbooks

The four checked-in configs are distinct contracts:

| Config | Ordered nodes | Endpoint needs |
| --- | --- | --- |
| `pipeline_image_attributes_pas.yaml` | PAS only over one attribute JSON | LLM |
| `pipeline_image_multiview_pas.yaml` | captioning -> visual_qa -> PAS | VLM + LLM |
| `pipeline_video_pas_reasoning.yaml` | tracking -> captioning -> event VQA -> reasoning -> person VQA -> PAS -> export | VLM + LLM |
| `pipeline_video_epas.yaml` | tracking -> captioning -> anomaly VQA -> person VQA -> PAS | VLM + LLM |

The Visual Attribute Search README is the authoritative scenario runbook:
[cookbooks/visual_attribute_search/README.md](../../cookbooks/visual_attribute_search/README.md).

## Multi-View Identity Images (RSTPReid)

`pipeline_image_multiview_pas.yaml` does not ship person images in git. Download
RSTPReid from
[NjtechCVLab/RSTPReid-Dataset](https://github.com/NjtechCVLab/RSTPReid-Dataset)
(archive:
[Google Drive](https://drive.google.com/file/d/1HTeDZUVrZr6nL56ZlkYBNqjSWh3IGV2X/view?usp=sharing)),
copy one identity's camera views into
`data/input_media/images/image_attribute_augmentations/<person_id>_RSTP/`,
and point both `--image-group-dir` entries plus `media_path` at that group.
Keep `media_path` as a single file. Placement details:
[image_attribute_augmentations/README.md](../../data/input_media/images/image_attribute_augmentations/README.md).

## Image Spatial Grounding

Two configs cover the full image use case:

`captioning -> grounding_2d -> referring_expressions`

- `pipeline_grounding.yaml` runs `captioning -> grounding_2d` over
  `data/input_media/images/traffic_intersection_frames/` (one still per
  staged traffic clip). Stage the NGC traffic clips and extract frames
  first. Scenes land under
  `output/auto_labeling/image_spatial_grounding/grounding/<filename>/`.
- `pipeline_referring.yaml` runs `referring_expressions` only on those
  grounding scenes. Use the same frames directory and the same `out_dir`.
  Run grounding first so each scene already has `contextual/objects.json`.

## Safe Sample Practice

- Copy tracked configs to `*.local.yaml`
- Keep `container.env` entries as variable names, not secret values
- Keep checkpoint mounts read-only
- Always run `--container-dry-run` before a real workflow
