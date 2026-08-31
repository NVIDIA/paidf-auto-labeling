# Warehouse video samples

Smart Spaces and video visual attribute search read clips from this folder.
Download the NGC VSS sample pack, then cut the warehouse videos into
30-second clips here.

## Get the clips

1. Download and extract `nvidia/vss-developer/dev-profile-sample-data:3.2.0`.
   NGC CLI install and extract steps:
   [Samples and Cookbooks](../../../../docs/user-guide/samples-and-cookbooks.md#download-vss-sample-clips).
2. Cut these files into 30-second clips: `warehouse_sample.mp4`,
   `sample-warehouse-ladder.mp4`, `warehouse_safety_0001.mp4`, and
   `warehouse_safety_0002.mp4`. Host `ffmpeg` and `ffprobe` must be on
   `PATH`. The clip cap is `210 / 30` from the longest pack file,
   `warehouse_sample.mp4`. Start with that file so the cap is not used up by
   the shorter clips.

```bash
SRC=./sample-data/dev-profile-sample-data
DEST=data/input_media/videos/warehouse_safety
CLIP_SECS=30
MAX_CLIPS=$((210 / CLIP_SECS))
mkdir -p "${DEST}"

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
  while [ "${i}" -lt "${MAX_CLIPS}" ] && [ "${start}" -lt "${dur:-0}" ]; do
    dest=$(printf '%s/warehouse_safety_%04d.mp4' "${DEST}" "${i}")
    ffmpeg -y -ss "${start}" -t "${CLIP_SECS}" -i "${src}" -c copy \
      -avoid_negative_ts make_zero "${dest}"
    i=$((i + 1))
    start=$((start + CLIP_SECS))
  done
done
```

Copy the files. Do not symlink them into this folder if a later bind-mount
would hide the target.

## What the cookbooks expect

```text
data/input_media/videos/warehouse_safety/
  warehouse_safety_0000.mp4
  ...
  warehouse_safety_0006.mp4
```

- Smart Spaces reads every MP4 in this directory.
- Visual attribute search video configs use `warehouse_safety_0000.mp4` as
  the first-run clip.
