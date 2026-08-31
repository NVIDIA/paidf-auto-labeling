# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for the reasoning service.

* always-on: with no ``--config-file`` the LLM stages are skipped and the service
  runs its DAFT validation bookends over a scene, exiting cleanly. No GPU, ffmpeg,
  or endpoints required.
* endpoint-backed: when an LLM endpoint and a reasoning config file are provided
  (``LLM_ENDPOINT_URL`` and ``UPA_E2E_REASONING_CONFIG``; ``LLM_MODEL`` optional),
  run an opt-in LLM stage over a captioning sidecar. The config file is supplied by
  the environment because enabled stages depend on the deployed model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from e2e_harness import Endpoints, RunService, entries_payload, llm_cli_args

pytestmark = pytest.mark.e2e


def test_reasoning_without_config_runs_validation_only(
    run_service: RunService,
    tmp_path: Path,
) -> None:
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"video")
    scene_dir = tmp_path / "scene"
    payload = entries_payload([{"media_path": str(media_path), "data_path": str(scene_dir)}])

    result = run_service("reasoning-service", ["--input", payload])

    result.assert_ok()


def test_reasoning_llm_stage_from_captioning_sidecar(
    run_service: RunService,
    endpoints: Endpoints,
    tmp_path: Path,
) -> None:
    llm = endpoints.require_llm()
    config_file = os.environ.get("UPA_E2E_REASONING_CONFIG")
    if not config_file:
        pytest.skip(
            "Set UPA_E2E_REASONING_CONFIG to a reasoning config file to run the LLM-stage test."
        )
    if not Path(config_file).is_file():
        pytest.skip(f"UPA_E2E_REASONING_CONFIG={config_file!r} does not point to an existing file.")

    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"video")
    scene_dir = tmp_path / "scene"
    sidecar = scene_dir / "sidecars" / "captioning" / "metadata_chunk.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        json.dumps(
            {
                "windows": [
                    {
                        "index": 0,
                        "start_s": 0.0,
                        "end_s": 2.0,
                        "description": "A red box moves from left to right across a dark scene.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    payload = entries_payload([{"media_path": str(media_path), "data_path": str(scene_dir)}])

    result = run_service(
        "reasoning-service",
        ["--input", payload, "--config-file", config_file, *llm_cli_args(llm)],
    )

    result.assert_ok()
