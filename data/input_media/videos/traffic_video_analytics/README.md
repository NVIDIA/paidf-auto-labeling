# Traffic video samples

Video data augmentation reads every MP4 in this folder. Download the NGC
VSS sample pack, then cut the two traffic videos into 10-second clips here.

## Get the clips

1. Download and extract `nvidia/vss-developer/dev-profile-sample-data:3.2.0`.
   NGC CLI install and extract steps:
   [Samples and Cookbooks](../../../../docs/user-guide/samples-and-cookbooks.md#download-vss-sample-clips).
2. Cut `sample-sim-jaywalking.mp4` and `sample-sim-traffic.mp4` into
   10-second clips. Host `ffmpeg` and `ffprobe` must be on `PATH`. The clip
   cap is `210 / 10` from the longest pack file, `warehouse_sample.mp4`.

```bash
SRC=./sample-data/dev-profile-sample-data
DEST=data/input_media/videos/traffic_video_analytics
CLIP_SECS=10
MAX_CLIPS=$((210 / CLIP_SECS))
mkdir -p "${DEST}"

i=0
for src in "${SRC}/sample-sim-jaywalking.mp4" "${SRC}/sample-sim-traffic.mp4"
do
  start=0
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "${src}")
  dur=${dur%.*}
  while [ "${i}" -lt "${MAX_CLIPS}" ] && [ "${start}" -lt "${dur:-0}" ]; do
    dest=$(printf '%s/traffic_sample_%03d.mp4' "${DEST}" "${i}")
    ffmpeg -y -ss "${start}" -t "${CLIP_SECS}" -i "${src}" -c copy \
      -avoid_negative_ts make_zero "${dest}"
    i=$((i + 1))
    start=$((start + CLIP_SECS))
  done
done
```

Copy the files. Do not symlink them into this folder if a later bind-mount
would hide the target. Leave `sample-sim-box-conveyor.mp4` and
`sample-drone-bridge.mp4` unused.

## What the cookbook expects

```text
data/input_media/videos/traffic_video_analytics/
  traffic_sample_000.mp4
  ...
  traffic_sample_020.mp4
```

After the clips are here, extract one still per clip for image spatial
grounding:
[Traffic intersection frames](../../images/traffic_intersection_frames/README.md).
