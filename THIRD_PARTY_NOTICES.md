<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Third-Party Notices

This project is distributed under the Apache License 2.0 (see `LICENSE`).

This file summarizes third-party open source components and other third-party materials referenced/used by this repository.

## Vendored source code included in this repository

### BoostTrack (MIT)
- **Upstream**: `https://github.com/vukasin-stanojevic/BoostTrack`
- **Where used**: `modules/detection_and_tracking/boosttrack/`
- **License**: MIT
- **License text**: `third_party_licenses/MIT-BoostTrack.txt`
- **Upstream info**: `modules/detection_and_tracking/boosttrack/UPSTREAM_LICENSE.md`

### Deep-OC-SORT (MIT)
- **Where used**: `modules/detection_and_tracking/deepocsort/`
- **License**: MIT
- **License text**: `third_party_licenses/MIT-DeepOCSort.txt`

### SeedVR / SeedVR2 (Apache-2.0)
- **Upstream**: `https://github.com/bytedance-seed/SeedVR`
- **Where used**: `modules/sr_runner/inference_seedvr2_window.py`
- **License**: Apache-2.0 (as stated by upstream project)
- **License text**: `third_party_licenses/Apache-2.0-SeedVR2.txt`
- **Notes**: Apache-2.0 derivative of upstream `projects/inference_seedvr2_window.py` with NVIDIA modifications layered on top; dual SPDX-FileCopyrightText (ByteDance 2025 + NVIDIA 2026) is preserved in the file header.

## Third-party-derived source code in this repository

### OpenAI CLIP (MIT) – architecture/naming reference
- **Upstream**: `https://github.com/openai/CLIP`
- **Where used**: `modules/detection_and_tracking/reid/vehicle_clip_vit.py`
- **License**: MIT
- **License text**: `third_party_licenses/MIT-OpenAI-CLIP.txt`

### CLIP-ReID (MIT) – VehicleID ReID integration reference
- **Upstream**: `https://github.com/Syliz517/CLIP-ReID`
- **Where used**:
  - `modules/detection_and_tracking/reid/vehicle_clip_vit.py`
  - runtime-downloaded weight described below
- **License**: MIT
- **License text**: `third_party_licenses/MIT-CLIP-ReID.txt`

## Runtime-downloaded / runtime-cloned artifacts (not distributed with this repository)

### `clip_vehicleid.pt` (VehicleID CLIP-ReID weights)
- **Filename**: `ckpts/reid/clip_vehicleid.pt` (inside the container: `/workspace/ckpts/reid/clip_vehicleid.pt`)
- **How it is obtained**: downloaded at runtime by `docker/entrypoint.sh` (Google Drive file id `168BLegHHxNqatW5wx1YyL2REaThWoof5`)
- **Source project**: CLIP-ReID (`https://github.com/Syliz517/CLIP-ReID`)
- **Note**: this repository does **not** distribute the model weights; users download them at runtime (or provide their own).

### SeedVR2 repository (cloned into the Docker image at build time)
- **Upstream**: `https://github.com/bytedance-seed/SeedVR` (pinned commit, see `docker/Dockerfile`)
- **What is cloned**: the upstream `SeedVR` repo and its bundled artifacts `pos_emb.pt`, `neg_emb.pt`, `configs_3b/`, `configs_7b/`, `common/`, `projects/`
- **Where it lives at runtime**: `/opt/seedvr/` inside the container only
- **License**: Apache-2.0 (upstream)
- **Note**: this repository does **not** vendor the SeedVR source tree or its weights. They are pulled into the container image by `docker/Dockerfile` at image build time.

## Python dependencies

Python dependencies are declared in `pyproject.toml` (and pinned in `uv.lock`). Dependencies are installed by the user/build process (not included as source in this repo).

For a per-package license audit of the Python packages actually installed inside the published Docker image, see [`third_party_licenses/python-deps-licenses.md`](third_party_licenses/python-deps-licenses.md) (generated with [`licensecheck`](https://pypi.org/project/licensecheck/) from inside the container).
