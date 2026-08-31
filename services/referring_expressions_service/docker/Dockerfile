# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM nvcr.io/nvidia/base/ubuntu:noble-20260610@sha256:a6015cf73597a174b8624ef0a9f78937f79a0862c078df4922dc35f26cec8fad AS builder

ARG PACKAGE_NAME

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        python3.12 \
        python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.14@sha256:f0786ad49e2e684c18d38697facb229f538a6f5e374c56f54125aabe7d14b3f7 /uv /uvx /bin/

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

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable --package "${PACKAGE_NAME}"

FROM nvcr.io/nvidia/base/ubuntu:noble-20260610@sha256:a6015cf73597a174b8624ef0a9f78937f79a0862c078df4922dc35f26cec8fad AS runtime

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        python3.12 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/scripts/prune_runtime_python.sh /tmp/prune-runtime-python
RUN bash /tmp/prune-runtime-python /app/.venv && rm /tmp/prune-runtime-python
COPY THIRD-PARTY-NOTICES /app/THIRD-PARTY-NOTICES
COPY LICENSE /app/LICENSE

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER ubuntu

ENTRYPOINT ["main"]
