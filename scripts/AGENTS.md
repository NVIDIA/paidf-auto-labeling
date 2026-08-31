# Project Scripts

## Controlled media toolchain

`media_toolchain.py` is the public build interface for the repository's compressed-media
dependencies. Its source versions, URLs, checksums, and Python build requirements are pinned in
`media-toolchain.toml`.

Container builders install the toolchain in cacheable steps. Use the `input-only` profile for
services that only probe or decode video, and `vp9-output` for services that generate VP9 video
outputs. Services that use PyAV or OpenCV must exclude the resolved media wheels from `uv sync`
and then source-build the needed Python media packages against the controlled FFmpeg:

```bash
python3 scripts/media_toolchain.py ffmpeg-install --profile vp9-output

/app/.venv/bin/python scripts/media_toolchain.py python-install \
  --python /app/.venv/bin/python \
  --component opencv-headless

/app/.venv/bin/python scripts/media_toolchain.py python-install \
  --python /app/.venv/bin/python \
  --component pyav

/app/.venv/bin/python scripts/media_toolchain.py verify \
  --profile vp9-output \
  --python /app/.venv/bin/python
```

The default artifact root is `/opt/media-runtime`. Multi-stage images copy its `bin`, `lib`, and
`oss-sources` directories into the runtime stage and run `verify` again after the final Python
environment and native libraries are installed.

The `verify` command fails when FFmpeg is missing. It checks the approved profile-specific
encoder, decoder, and bitstream-filter surface; allows the native MPEG-4 Part 2 input decoder
and FFmpeg's internal H.263 decoder dependency; rejects common GPL, nonfree, H.264 software, and
H.265 markers; and rejects private FFmpeg payloads under the selected Python package roots.

`ffmpeg_codec_policy.py` is an internal helper for configure flags and marker checks. Do not call
it directly from service Dockerfiles; use `media_toolchain.py ffmpeg-install` and
`media_toolchain.py verify`.

This is a codec-policy check, not a general-purpose license scanner. Dependency-license
inventory, SPDX classification, attribution generation, and SBOM validation should remain a
separate compliance step.
