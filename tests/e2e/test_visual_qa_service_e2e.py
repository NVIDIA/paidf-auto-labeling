# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E test for the visual-QA service against an external VLM endpoint.

Requires ``VLM_ENDPOINT_URL`` (and optionally ``VLM_MODEL`` /
``NVIDIA_API_KEY``) plus ffmpeg on PATH; self-skips otherwise.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from e2e_harness import Endpoints, RunService, entries_payload, require_ffmpeg, vlm_cli_args

pytestmark = pytest.mark.e2e


def test_visual_qa_window_direct_vlm(
    run_service: RunService,
    endpoints: Endpoints,
    sample_video: Callable[..., Path],
    tmp_path: Path,
) -> None:
    vlm = endpoints.require_vlm()
    require_ffmpeg()
    media_path = sample_video(tmp_path / "clip.webm", seconds=2.0, fps=8.0)
    scene_dir = tmp_path / "scene"
    question_bank = tmp_path / "questions.json"
    question_bank.write_text(
        json.dumps(
            {
                "questions": [
                    {"id": "q1", "question": "What is the main moving object in the scene?"},
                    {"id": "q2", "question": "Describe the background of the scene."},
                ]
            }
        ),
        encoding="utf-8",
    )
    payload = entries_payload([{"media_path": str(media_path), "data_path": str(scene_dir)}])

    result = run_service(
        "visual-qa-service",
        [
            "--input",
            payload,
            "--generation-mode",
            "window-direct-vlm",
            "--single-window",
            "--media-mode",
            "frames",
            "--sampling-fps",
            "1",
            "--max-frames",
            "4",
            "--question-bank-file",
            str(question_bank),
            *vlm_cli_args(vlm),
        ],
    )

    result.assert_ok()
    assert (scene_dir / "sidecars" / "visual_qa" / "items.json").exists(), (
        f"No visual QA items sidecar written. Output:\n{result.output}"
    )
