# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E smoke test for the example service (always-on baseline)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from e2e_harness import RunService

pytestmark = pytest.mark.e2e


def test_example_service_seeds_scene(run_service: RunService, tmp_path: Path) -> None:
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"media")
    scene_dir = tmp_path / "scene"
    payload = json.dumps(
        [{"id": "entry-1", "media_path": str(media_path), "data_path": str(scene_dir)}]
    )

    result = run_service("example-service", ["--input", payload])

    result.assert_ok()
    assert (scene_dir / "sidecars" / "active.mp4").exists()
    assert (scene_dir / "sidecars" / "raw.mp4").exists()
