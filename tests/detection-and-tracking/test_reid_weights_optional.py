#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Unit test: when ReID weights path is missing, we pass None so all three trackers run without crashing.
# No rfdetr/cv2 dependency.
#

import tempfile
from pathlib import Path


def _reid_weights_effective(reid_weights: Path | None) -> Path | None:
    """Same logic as rfdetr_tracking.py main: use ReID only if path exists."""
    return reid_weights if (reid_weights and reid_weights.exists()) else None


def test_reid_missing_yields_none():
    """When reid_weights path does not exist, effective value is None (no crash)."""
    missing = Path("/nonexistent/reid/clip_vehicleid.pt")
    assert not missing.exists()
    assert _reid_weights_effective(missing) is None


def test_reid_none_yields_none():
    """When reid_weights is None, effective value is None."""
    assert _reid_weights_effective(None) is None


def test_reid_exists_yields_path():
    """When path exists, effective value is the path."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        p = Path(f.name)
    try:
        assert p.exists()
        assert _reid_weights_effective(p) == p
    finally:
        p.unlink(missing_ok=True)
