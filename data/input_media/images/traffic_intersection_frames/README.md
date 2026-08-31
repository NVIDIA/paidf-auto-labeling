# Traffic intersection frames

Image spatial grounding reads every JPEG in this folder. Each file is the
first frame of a traffic clip. You prepare the frames locally after you
stage those clips.

## Get the frames

1. Download the NGC VSS sample pack and cut the traffic videos into this
   repo. Follow
   [Traffic video samples](../../videos/traffic_video_analytics/README.md).
2. Extract the first frame of each clip:

```bash
SRC=data/input_media/videos/traffic_video_analytics
DEST=data/input_media/images/traffic_intersection_frames
mkdir -p "${DEST}"

for mp4 in "${SRC}"/traffic_sample_*.mp4; do
  base=$(basename "${mp4}" .mp4)
  ffmpeg -y -i "${mp4}" -frames:v 1 "${DEST}/${base}.jpg"
done
```

NGC CLI install and extract steps:
[Samples and Cookbooks](../../../../docs/user-guide/samples-and-cookbooks.md#download-vss-sample-clips).

## What the cookbooks expect

```text
data/input_media/images/traffic_intersection_frames/
  traffic_sample_000.jpg
  ...
  traffic_sample_020.jpg
```

- `pipeline_grounding.yaml` reads every image in this directory and writes
  scenes under `output/auto_labeling/image_spatial_grounding/grounding/`.
- `pipeline_referring.yaml` uses the same frames and that same grounding
  `out_dir`. Run grounding first so each scene has `contextual/objects.json`.
