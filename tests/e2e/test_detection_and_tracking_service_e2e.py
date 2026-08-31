# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for the detection-and-tracking service.

Real detection/tracking backends (RF-DETR, SAM3) require GPU model weights, which
are out of scope for the MR-gating suite. These tests cover the service wiring:

* always-on: ``--disabled`` exits cleanly without touching media.
* stub backend: run the weight-free ``stub`` tracker over a real video fixture
  (requires ffmpeg for source-mode decode; no GPU or weights).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from e2e_harness import RunService, entries_payload, require_ffmpeg

pytestmark = pytest.mark.e2e


def test_detection_and_tracking_disabled_exits_cleanly(
    run_service: RunService,
    tmp_path: Path,
) -> None:
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"video")
    scene_dir = tmp_path / "scene"
    payload = entries_payload([{"media_path": str(media_path), "data_path": str(scene_dir)}])

    result = run_service("detection-and-tracking-service", ["--disabled", "--input", payload])

    result.assert_ok()


def test_detection_and_tracking_stub_backend_processes_video(
    run_service: RunService,
    sample_video: Callable[..., Path],
    tmp_path: Path,
) -> None:
    require_ffmpeg()
    media_path = sample_video(tmp_path / "clip.webm")
    scene_dir = tmp_path / "scene"
    payload = entries_payload([{"media_path": str(media_path), "data_path": str(scene_dir)}])

    result = run_service(
        "detection-and-tracking-service",
        ["--tracker", "stub", "--input", payload],
    )

    result.assert_ok()
    assert (scene_dir / "sidecars").is_dir()
