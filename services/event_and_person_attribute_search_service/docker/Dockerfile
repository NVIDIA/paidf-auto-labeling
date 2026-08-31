# syntax=docker/dockerfile:1.7

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE=nvcr.io/nvidia/base/ubuntu:noble-20260610@sha256:a6015cf73597a174b8624ef0a9f78937f79a0862c078df4922dc35f26cec8fad

FROM ${BASE_IMAGE} AS builder

ARG PACKAGE_NAME=event-and-person-attribute-search-service

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates python3.12 python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.24@sha256:99ea34acedc870ba4ad11a1f540a1c04267c9f30aadc465a94406f52dfda2c36 /uv /uvx /bin/

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

FROM ${BASE_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates python3.12 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system appuser \
    && useradd --system --gid appuser --home-dir /home/appuser --create-home appuser

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/scripts/prune_runtime_python.sh /tmp/prune-runtime-python
RUN bash /tmp/prune-runtime-python /app/.venv && rm /tmp/prune-runtime-python
COPY THIRD-PARTY-NOTICES /app/THIRD-PARTY-NOTICES
COPY LICENSE /app/LICENSE

ENV HOME=/home/appuser \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

RUN chown -R appuser:appuser /app /home/appuser

USER appuser

ENTRYPOINT ["main"]
