# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest fixtures for the end-to-end (E2E) service test suite.

These tests exercise real service entrypoints in *source mode*: each service is
launched exactly the way ``scripts/run.py`` launches it (``uv run --package
<project> python -c <entrypoint-runner>``), so the ``main`` console-script name
collision across packages is avoided and the tested path matches ``make run``.

The suite runs as an MR gating job on a GPU runner while staying runnable
locally. Individual tests self-skip when their prerequisites are missing so that
always-on smoke tests gate every service with no GPU, ffmpeg, or network access,
while endpoint-backed tests activate only when the external LLM/VLM endpoints are
configured. See ``tests/e2e/README.md`` for the full contract.

Shared logic and types live in ``e2e_harness`` (importable because ``tests/e2e``
is on the pytest ``pythonpath``); the fixtures below are thin wrappers over it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from e2e_harness import (
    Endpoints,
    RunService,
    build_completed_scene,
    build_service_runner,
    ffmpeg_on_path,
    generate_image,
    require_ffmpeg,
    resolve_endpoints,
    resolve_sample_video,
)


@pytest.fixture(scope="session")
def run_service() -> RunService:
    """Launch a workspace service by project name and capture its result."""
    return build_service_runner()


@pytest.fixture(scope="session")
def endpoints() -> Endpoints:
    """External LLM/VLM endpoint configuration resolved from the environment."""
    return resolve_endpoints()


@pytest.fixture(scope="session")
def ffmpeg_available() -> bool:
    """Whether ffmpeg/ffprobe are available on PATH for source-mode decode."""
    return ffmpeg_on_path()


@pytest.fixture(scope="session")
def require_ffmpeg_fixture() -> Callable[[], None]:
    """Return a callable that skips the test when ffmpeg/ffprobe are missing."""
    return require_ffmpeg


@pytest.fixture(scope="session")
def make_image() -> Callable[..., Path]:
    """Return a callable that writes a small JPEG image fixture."""
    return generate_image


@pytest.fixture(scope="session")
def sample_video() -> Callable[..., Path]:
    """Return a callable resolving the input video for decode-path tests.

    Uses ``UPA_E2E_SAMPLE_VIDEO`` when set (e.g. a traffic H.264 sample under
    ``data/input_media/videos/traffic_video_analytics``), otherwise generates a
    small synthetic VP9 clip.
    """
    return resolve_sample_video


@pytest.fixture
def completed_scene() -> Callable[..., Path]:
    """Return a callable that builds a completed DAFT scene for training-export."""
    return build_completed_scene
