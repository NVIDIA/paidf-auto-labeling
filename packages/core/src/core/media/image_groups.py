# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared image-group discovery and selection helpers."""

from __future__ import annotations

from pathlib import Path

from core.scene import is_image_path


def discover_image_group(directory: Path | str, *, max_images: int = 0) -> tuple[Path, ...]:
    """Recursively discover a deterministic, optionally even-sampled image group."""
    group_dir = Path(directory).expanduser()
    if not group_dir.exists():
        raise ValueError(f"Image group directory does not exist: {group_dir}")
    if not group_dir.is_dir():
        raise ValueError(f"Image group path is not a directory: {group_dir}")
    if max_images < 0:
        raise ValueError("max_images must be greater than or equal to 0")

    images = sorted(
        (path for path in group_dir.rglob("*") if path.is_file() and is_image_path(path)),
        key=lambda path: path.relative_to(group_dir).as_posix(),
    )
    if not images:
        raise ValueError(f"Image group directory contains no supported images: {group_dir}")
    if max_images == 0 or len(images) <= max_images:
        return tuple(images)
    if max_images == 1:
        return (images[0],)

    last_index = len(images) - 1
    indices = [round(index * last_index / (max_images - 1)) for index in range(max_images)]
    return tuple(images[index] for index in indices)


__all__ = ["discover_image_group"]
