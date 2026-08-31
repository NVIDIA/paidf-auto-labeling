# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E test for the training-export service (always-on).

Builds a completed DAFT scene and exports it to a training dataset format. No GPU,
ffmpeg, or model endpoints are required.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from e2e_harness import RunService, entries_payload

pytestmark = pytest.mark.e2e


def test_training_export_writes_cosmos_dataset(
    run_service: RunService,
    completed_scene: Callable[..., Path],
    tmp_path: Path,
) -> None:
    scene_dir = completed_scene(tmp_path / "scene")
    export_dir = tmp_path / "exports"
    payload = entries_payload(
        [{"media_path": str(scene_dir / "raw" / "clip.mp4"), "data_path": str(scene_dir)}]
    )

    result = run_service(
        "training-export-service",
        [
            "--input",
            payload,
            "--training-export-format",
            "cosmos-reason-v1.0",
            "--training-export-dir",
            str(export_dir),
        ],
    )

    result.assert_ok()
    meta_files = list(export_dir.rglob("meta.json"))
    assert meta_files, f"No meta.json written under {export_dir}. Output:\n{result.output}"
    meta = json.loads(meta_files[0].read_text(encoding="utf-8"))
    assert meta.get("samples"), "Exported cosmos-reason dataset has no samples."
