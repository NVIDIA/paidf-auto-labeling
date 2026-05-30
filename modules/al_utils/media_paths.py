# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
# Limited to mp4-family containers: the shipped FFmpeg build is LGPL-only and
# only ships the mov demuxer (which transparently handles .mp4 / .m4v / .mov /
# 3gp variants). Matroska (.mkv / .webm) and AVI demuxers are intentionally
# not compiled in — see docker/Dockerfile FFmpeg configure block.
VIDEO_EXTS = {".mp4", ".mov", ".m4v"}


def is_image_path(path: Path) -> bool:
    try:
        return Path(path).suffix.lower() in IMAGE_EXTS
    except Exception:
        return False


def is_video_path(path: Path) -> bool:
    try:
        return Path(path).suffix.lower() in VIDEO_EXTS
    except Exception:
        return False


__all__ = [
    "IMAGE_EXTS",
    "VIDEO_EXTS",
    "is_image_path",
    "is_video_path",
]
