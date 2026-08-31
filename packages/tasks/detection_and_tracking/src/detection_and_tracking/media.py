# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Approved video decoding and VP9 output helpers for detection backends."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.media import (
    VideoStreamInfo,
    Vp9VideoWriter,
    iter_decoded_rgb24_frames,
    prepare_video_decode,
)


@dataclass(frozen=True)
class DecodedBgrVideo:
    """Approved video metadata and a streaming BGR frame iterator."""

    stream: VideoStreamInfo
    frames: Iterator[Any]


def decode_video_bgr(media_path: Path, *, cv2: Any, np: Any) -> DecodedBgrVideo:
    """Validate and stream an H.264, VP9, or MPEG-4 Part 2 video as BGR frames."""
    plan = prepare_video_decode(media_path)

    def iter_frames() -> Iterator[Any]:
        shape = (plan.stream.height, plan.stream.width, 3)
        for payload in iter_decoded_rgb24_frames(media_path, plan):
            rgb = np.frombuffer(payload, dtype=np.uint8).reshape(shape)
            yield cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    return DecodedBgrVideo(stream=plan.stream, frames=iter_frames())


def iter_selected_video_frames_bgr(
    media_path: Path,
    frame_numbers: list[int],
    *,
    cv2: Any,
    np: Any,
) -> Iterator[tuple[int, Any]]:
    """Yield ``(frame_index, bgr_ndarray)`` for requested 0-based frame indices.

    Uses the approved core FFmpeg decode path. OpenCV is only used for RGB→BGR
    conversion; ``cv2.VideoCapture`` is intentionally unavailable in service
    images.
    """
    if not frame_numbers:
        return
    if any(frame_number < 0 for frame_number in frame_numbers):
        raise ValueError("frame_numbers must be non-negative 0-based indices")

    wanted = set(frame_numbers)
    last_wanted = max(wanted)
    plan = prepare_video_decode(media_path)
    shape = (plan.stream.height, plan.stream.width, 3)
    for idx, payload in enumerate(iter_decoded_rgb24_frames(media_path, plan)):
        if idx > last_wanted:
            break
        if idx not in wanted:
            continue
        rgb = np.frombuffer(payload, dtype=np.uint8).reshape(shape)
        yield idx, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


__all__ = [
    "DecodedBgrVideo",
    "Vp9VideoWriter",
    "decode_video_bgr",
    "iter_selected_video_frames_bgr",
]
