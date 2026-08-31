# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E test for the workflow runner (always-on, dry-run).

The workflow runner orchestrates the other services as Docker containers. A real
multi-container run needs a GPU-capable container runtime, which the DinD CI
service does not provide, so this MR-gating test validates the container plan in
``--container-dry-run`` mode (no Docker or GPU required).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from e2e_harness import RunService, entries_payload

pytestmark = pytest.mark.e2e


def test_workflow_runner_dry_run_plans_video_pipeline(
    run_service: RunService,
    tmp_path: Path,
) -> None:
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"video")
    scene_dir = tmp_path / "scene"
    payload = entries_payload([{"media_path": str(media_path), "data_path": str(scene_dir)}])

    result = run_service(
        "workflow-runner",
        [
            "--input",
            payload,
            "--pipeline",
            "video",
            "--no-expand-input-dirs",
            "--container-dry-run",
        ],
    )

    result.assert_ok()
