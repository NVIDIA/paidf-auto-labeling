# syntax=docker/dockerfile:1.7

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG BUILDER_BASE_IMAGE=upa-media-base:ubuntu-input-only
ARG BASE_IMAGE=nvcr.io/nvidia/base/ubuntu:noble-20260610@sha256:a6015cf73597a174b8624ef0a9f78937f79a0862c078df4922dc35f26cec8fad

# Builder starts from the shared media base, which already carries the restricted
# FFmpeg build in /usr/local and /opt/media-runtime. CI passes the published base
# ref via --build-arg BUILDER_BASE_IMAGE=...; a local `make build` uses the default
# below, so build `upa-media-base:ubuntu-input-only` first.
FROM ${BUILDER_BASE_IMAGE} AS builder

ARG PACKAGE_NAME=reasoning-service

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

COPY --from=ghcr.io/astral-sh/uv:0.11.24@sha256:99ea34acedc870ba4ad11a1f540a1c04267c9f30aadc465a94406f52dfda2c36 /uv /uvx /bin/

# Keep bytecode compilation disabled during image build. The install tree includes
# packages with large metadata directories, and uv's compile walk can exhaust
# Docker's default open-file limit on common developer machines.
ENV PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=0 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PYTHON=/usr/bin/python3.12

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY packages ./packages
COPY services ./services
COPY scripts ./scripts

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable --package "${PACKAGE_NAME}"

RUN /app/.venv/bin/python scripts/media_toolchain.py verify \
    --profile input-only --python /app/.venv/bin/python

FROM ${BASE_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        python3.12 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system appuser \
    && useradd --system --gid appuser --home-dir /home/appuser --create-home appuser

WORKDIR /app

RUN --mount=from=builder,source=/opt/media-runtime,target=/tmp/media-runtime,readonly \
    cp -a /tmp/media-runtime/bin/. /usr/local/bin/ \
    && cp -a /tmp/media-runtime/lib/. /usr/local/lib/ \
    && mkdir -p /usr/share/oss-sources \
    && cp -a /tmp/media-runtime/oss-sources/. /usr/share/oss-sources/ \
    && ldconfig

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/scripts/prune_runtime_python.sh /tmp/prune-runtime-python
RUN bash /tmp/prune-runtime-python /app/.venv && rm /tmp/prune-runtime-python
COPY THIRD-PARTY-NOTICES /app/THIRD-PARTY-NOTICES
COPY LICENSE /app/LICENSE
COPY --from=builder /app/scripts/media-toolchain.toml /app/scripts/media_toolchain.py \
    /app/scripts/ffmpeg_codec_policy.py /app/scripts/

ENV HOME=/home/appuser \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

RUN /app/.venv/bin/python /app/scripts/media_toolchain.py verify \
    --profile input-only --python /app/.venv/bin/python

RUN chown -R appuser:appuser /app /home/appuser

USER appuser

ENTRYPOINT ["main"]
