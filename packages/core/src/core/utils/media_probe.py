# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Media frame-count probing."""

from __future__ import annotations

from pathlib import Path

from core.scene import is_image_path


def probe_frame_count(media_path: str | Path) -> int | None:
    """Return the frame count of ``media_path`` (1 for images), or ``None`` if unknown."""
    path = Path(media_path)
    if not path.is_file():
        return None
    if is_image_path(path):
        return 1

    # Video frame counting is disabled until it can be implemented without making
    # OpenCV (and its bundled FFmpeg libraries) a dependency of the core package.
    return None


__all__ = ["probe_frame_count"]
