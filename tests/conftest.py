# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _reset_scene_media_id():
    """Reset the module-level scene media id between tests."""
    import daft_export.common as _common

    original = _common._scene_media_id
    yield
    _common._scene_media_id = original


# Automatically add SEEDVR_ROOT to sys.path so tests can import 'common' from SeedVR.
# This removes the need to manually set PYTHONPATH=/opt/seedvr when running pytest.


def pytest_configure(config):
    # 1. Ensure project modules are importable (handled by pyproject.toml usually, but safe to add)
    # This assumes conftest.py is in auto-labeling/tests/
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # 2. Find and add SeedVR
    seedvr_root = os.getenv("SEEDVR_ROOT")

    # If env var not set, try default docker location
    if not seedvr_root and os.path.isdir("/opt/seedvr"):
        seedvr_root = "/opt/seedvr"

    # If still not set, try local vendor location
    if not seedvr_root:
        local_vendor = repo_root / "modules" / "super_resolution"
        if local_vendor.is_dir() and (local_vendor / "common").is_dir():
            seedvr_root = str(local_vendor)

    if seedvr_root and os.path.isdir(seedvr_root):
        # Verify it looks like SeedVR (has common/)
        if (Path(seedvr_root) / "common").is_dir():
            if seedvr_root not in sys.path:
                sys.path.append(seedvr_root)
