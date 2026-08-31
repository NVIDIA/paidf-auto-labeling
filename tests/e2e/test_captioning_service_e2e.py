# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for the captioning service against an external VLM endpoint.

These tests require ``VLM_ENDPOINT_URL`` (and optionally ``VLM_MODEL`` /
``NVIDIA_API_KEY``); they self-skip otherwise. The video test additionally needs
ffmpeg on PATH for source-mode decode.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from e2e_harness import Endpoints, RunService, entries_payload, require_ffmpeg, vlm_cli_args

pytestmark = pytest.mark.e2e


def test_captioning_image_caption_against_vlm(
    run_service: RunService,
    endpoints: Endpoints,
    make_image: Callable[..., Path],
    tmp_path: Path,
) -> None:
    vlm = endpoints.require_vlm()
    media_path = make_image(tmp_path / "frame.jpg")
    scene_dir = tmp_path / "scene"
    payload = entries_payload([{"media_path": str(media_path), "data_path": str(scene_dir)}])

    result = run_service(
        "captioning-service",
        ["--input", payload, "--max-tokens", "256", *vlm_cli_args(vlm)],
    )

    result.assert_ok()
    assert (scene_dir / "sidecars" / "captioning" / "image_caption.json").exists(), (
        f"No image caption sidecar written. Output:\n{result.output}"
    )


def test_captioning_video_windows_against_vlm(
    run_service: RunService,
    endpoints: Endpoints,
    sample_video: Callable[..., Path],
    tmp_path: Path,
) -> None:
    vlm = endpoints.require_vlm()
    require_ffmpeg()
    media_path = sample_video(tmp_path / "clip.webm", seconds=2.0, fps=8.0)
    scene_dir = tmp_path / "scene"
    payload = entries_payload([{"media_path": str(media_path), "data_path": str(scene_dir)}])

    result = run_service(
        "captioning-service",
        [
            "--input",
            payload,
            "--single-window",
            "--media-mode",
            "frames",
            "--sampling-fps",
            "1",
            "--max-frames",
            "4",
            "--max-tokens",
            "256",
            *vlm_cli_args(vlm),
        ],
    )

    result.assert_ok()
    assert (scene_dir / "sidecars" / "captioning" / "metadata_chunk.json").exists(), (
        f"No dense caption sidecar written. Output:\n{result.output}"
    )
