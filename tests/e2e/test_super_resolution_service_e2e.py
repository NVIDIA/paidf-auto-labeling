# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for the super-resolution service.

Real SeedVR2 upscaling requires GPU model weights, which are out of scope for the
MR-gating suite. These tests cover the service wiring without heavy inference:

* always-on: ``--disabled`` exits cleanly without touching media.
* endpoint/ffmpeg-free auto path: skip SR when the input already meets the target
  resolution (requires ffmpeg to probe a real video fixture).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from e2e_harness import RunService, entries_payload, require_ffmpeg

pytestmark = pytest.mark.e2e


def test_super_resolution_disabled_exits_cleanly(
    run_service: RunService,
    tmp_path: Path,
) -> None:
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"video")
    scene_dir = tmp_path / "scene"
    payload = entries_payload([{"media_path": str(media_path), "data_path": str(scene_dir)}])

    result = run_service("super-resolution-service", ["--disabled", "--input", payload])

    result.assert_ok()


def test_super_resolution_auto_policy_skips_high_res_input(
    run_service: RunService,
    sample_video: Callable[..., Path],
    tmp_path: Path,
) -> None:
    require_ffmpeg()
    media_path = sample_video(tmp_path / "hires.webm", width=1280, height=720)
    scene_dir = tmp_path / "scene"
    payload = entries_payload([{"media_path": str(media_path), "data_path": str(scene_dir)}])

    result = run_service(
        "super-resolution-service",
        [
            "--input",
            payload,
            "--resolution-policy",
            "auto",
            "--min-input-short-side",
            "720",
            "--min-input-long-side",
            "1280",
            "--empty-output-policy",
            "warn",
        ],
    )

    result.assert_ok()
