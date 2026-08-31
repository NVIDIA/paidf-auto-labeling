# syntax=docker/dockerfile:1.7

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build from the repository root:
#   docker build -t paidf-super-resolution-service \
#     --build-arg PACKAGE_NAME=super-resolution-service \
#     -f services/super_resolution_service/docker/Dockerfile .
#
# Runtime expectation:
# - this is the standalone SR service container boundary;
# - SeedVR code is baked into the image at SEEDVR_ROOT, default /opt/seedvr;
# - checkpoints are available at MODEL_CACHE_PATH, default /models;
# - the NGC PyTorch base supplies CUDA, PyTorch, TorchVision, and torchrun;
# - FlashAttention is compiled into the service venv.

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c

FROM ${BASE_IMAGE} AS builder

ARG PACKAGE_NAME
ARG FLASH_ATTN_VERSION=2.8.3
ARG SEEDVR_REPO_URL=https://github.com/ByteDance-Seed/SeedVR.git
ARG SEEDVR_GIT_SHA=e4de8c24441a67e1b7df56abea10645059bb1185

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        nasm \
        pkg-config \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        libvpx-dev \
        ninja-build \
        xz-utils \
        yasm \
    && rm -rf /var/lib/apt/lists/*

# Build FFmpeg from the shared manifest and LGPL-friendly component allow-list.
COPY scripts/media-toolchain.toml scripts/media_toolchain.py \
    scripts/ffmpeg_codec_policy.py scripts/build_restricted_ffmpeg.sh /tmp/media-build/
RUN python3 /tmp/media-build/media_toolchain.py ffmpeg-install --profile vp9-output \
    --prefix /opt/media-toolchain

COPY --from=ghcr.io/astral-sh/uv:0.11.24@sha256:99ea34acedc870ba4ad11a1f540a1c04267c9f30aadc465a94406f52dfda2c36 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=0 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_NO_DEV=1 \
    UV_PYTHON=/usr/bin/python3.12

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY packages ./packages
COPY services ./services
COPY scripts ./scripts

# Reuse the base image's CUDA-enabled PyTorch and TorchVision, which are
# available for both supported architectures. Do not install PyPI CUDA wheels:
# they would shadow these matching libraries.
RUN /usr/bin/python3.12 -m venv --system-site-packages /app/.venv

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable --package "${PACKAGE_NAME}" \
        --no-install-package av \
        --no-install-package torch \
        --no-install-package torchvision \
        --no-install-package nvidia-cublas-cu12 \
        --no-install-package nvidia-cuda-cupti-cu12 \
        --no-install-package nvidia-cuda-nvrtc-cu12 \
        --no-install-package nvidia-cuda-runtime-cu12 \
        --no-install-package nvidia-cudnn-cu12 \
        --no-install-package nvidia-cufft-cu12 \
        --no-install-package nvidia-cufile-cu12 \
        --no-install-package nvidia-curand-cu12 \
        --no-install-package nvidia-cusolver-cu12 \
        --no-install-package nvidia-cusparse-cu12 \
        --no-install-package nvidia-cusparselt-cu12 \
        --no-install-package nvidia-nccl-cu12 \
        --no-install-package nvidia-nvjitlink-cu12 \
        --no-install-package nvidia-nvshmem-cu12 \
        --no-install-package nvidia-nvtx-cu12

# Rebuild PyAV against the restricted /usr/local FFmpeg libraries.
RUN /app/.venv/bin/python scripts/media_toolchain.py python-install \
    --python /app/.venv/bin/python --component pyav --prefix /opt/media-toolchain

# Build FlashAttention against the PyTorch 26.07 CUDA 13.3 stack. ninja is
# installed above and PyTorch honors MAX_JOBS to parallelize native compilation.
# Keep the job count bounded to avoid exhausting memory on CI runners.
RUN --mount=type=cache,target=/root/.cache/uv \
    CUDA_HOME=/usr/local/cuda \
    MAX_JOBS=4 \
    uv pip install --verbose --python /app/.venv/bin/python --no-build-isolation \
        "flash-attn==${FLASH_ATTN_VERSION}"

RUN LD_LIBRARY_PATH=/opt/media-toolchain/lib \
    /app/.venv/bin/python scripts/media_toolchain.py verify \
        --profile vp9-output --python /app/.venv/bin/python \
        --ffmpeg /opt/media-toolchain/bin/ffmpeg

RUN set -eux; \
    git init /opt/seedvr; \
    cd /opt/seedvr; \
    git remote add origin "${SEEDVR_REPO_URL}"; \
    git fetch --depth 1 origin "${SEEDVR_GIT_SHA}"; \
    git checkout --detach FETCH_HEAD; \
    mkdir -p /opt/seedvr/projects/video_diffusion_sr; \
    cp /app/packages/tasks/super_resolution/src/super_resolution/video_diffusion_sr/color_fix.py \
        /opt/seedvr/projects/video_diffusion_sr/color_fix.py; \
    sed -i 's/fusedrms/rms/g; s/fusedln/layer/g' \
        /opt/seedvr/configs_3b/main.yaml \
        /opt/seedvr/configs_7b/main.yaml; \
    ! grep -E "fusedrms|fusedln" /opt/seedvr/configs_3b/main.yaml /opt/seedvr/configs_7b/main.yaml; \
    test -d /opt/seedvr/common; \
    test -d /opt/seedvr/projects; \
    test -f /opt/seedvr/pos_emb.pt; \
    test -f /opt/seedvr/neg_emb.pt; \
    rm -rf /opt/seedvr/.git

FROM ${BASE_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        python3.12 \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=from=builder,source=/opt/media-runtime,target=/tmp/media-runtime,readonly \
    mkdir -p /opt/media-runtime \
    && cp -a /tmp/media-runtime/. /opt/media-runtime/ \
    && mkdir -p /usr/share/oss-sources \
    && cp -a /tmp/media-runtime/oss-sources/. /usr/share/oss-sources/

COPY --from=builder /app/.venv /app/.venv
COPY THIRD-PARTY-NOTICES /app/THIRD-PARTY-NOTICES
COPY LICENSE /app/LICENSE
COPY --from=builder --chown=ubuntu:ubuntu /opt/seedvr /opt/seedvr
COPY --from=builder /app/scripts/media-toolchain.toml /app/scripts/media_toolchain.py \
    /app/scripts/ffmpeg_codec_policy.py /app/scripts/

COPY --from=builder /app/scripts/prune_runtime_python.sh /tmp/prune-runtime-python
RUN bash /tmp/prune-runtime-python /app/.venv && rm /tmp/prune-runtime-python

ENV PATH="/opt/media-runtime/bin:/app/.venv/bin:$PATH" \
    LD_LIBRARY_PATH="/opt/media-runtime/lib" \
    PYTHONUNBUFFERED=1 \
    MODEL_CACHE_PATH=/models \
    SEEDVR_ROOT=/opt/seedvr \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

RUN /app/.venv/bin/python /app/scripts/media_toolchain.py verify \
    --profile vp9-output --python /app/.venv/bin/python \
    --ffmpeg /opt/media-runtime/bin/ffmpeg

RUN mkdir -p /models/seedvr2 /workspace \
    && chown -R ubuntu:ubuntu /models /workspace /opt/seedvr

USER ubuntu

ENTRYPOINT ["main"]
