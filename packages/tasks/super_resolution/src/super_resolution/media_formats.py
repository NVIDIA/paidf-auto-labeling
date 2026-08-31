# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Media-format policy for the SeedVR2 super-resolution runtime."""

from __future__ import annotations

from pathlib import Path

SUPPORTED_SR_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }
)
"""Image suffixes accepted by the SR image path."""

SUPPORTED_SR_VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".mp4",
        ".mov",
        ".webm",
    }
)
"""Video suffixes accepted by the SR video path."""


def is_supported_sr_image_path(path: Path | str) -> bool:
    """Return true when ``path`` is an SR-supported image path."""
    return Path(path).suffix.lower() in SUPPORTED_SR_IMAGE_EXTENSIONS


def is_supported_sr_video_path(path: Path | str) -> bool:
    """Return true when ``path`` is an SR-supported video path."""
    return Path(path).suffix.lower() in SUPPORTED_SR_VIDEO_EXTENSIONS


def is_supported_sr_media_path(path: Path | str) -> bool:
    """Return true when ``path`` has an SR-supported image or video suffix."""
    return is_supported_sr_image_path(path) or is_supported_sr_video_path(path)


def sr_supported_extensions_message() -> str:
    """Return a stable human-readable summary of SR-supported suffixes."""
    image_exts = ", ".join(sorted(SUPPORTED_SR_IMAGE_EXTENSIONS))
    video_exts = ", ".join(sorted(SUPPORTED_SR_VIDEO_EXTENSIONS))
    return f"supported image extensions: {image_exts}; supported video extensions: {video_exts}"


def sr_output_name_for_input(path: Path | str, *, stem: str = "sr_output") -> str:
    """Return the SR sidecar output filename for a supported input path."""
    suffix = Path(path).suffix.lower()
    if suffix in SUPPORTED_SR_IMAGE_EXTENSIONS:
        return f"{stem}{suffix}"
    if suffix in SUPPORTED_SR_VIDEO_EXTENSIONS:
        return f"{stem}.mp4"
    rendered_suffix = suffix or "<none>"
    raise ValueError(
        f"Unsupported super-resolution media extension {rendered_suffix!r}; "
        f"{sr_supported_extensions_message()}"
    )


__all__ = [
    "SUPPORTED_SR_IMAGE_EXTENSIONS",
    "SUPPORTED_SR_VIDEO_EXTENSIONS",
    "is_supported_sr_image_path",
    "is_supported_sr_media_path",
    "is_supported_sr_video_path",
    "sr_output_name_for_input",
    "sr_supported_extensions_message",
]
