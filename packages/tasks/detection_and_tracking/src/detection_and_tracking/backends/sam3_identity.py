# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared SAM3 identity helpers: chunking policy and human track labels."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from detection_and_tracking.config import DetectionAndTrackingConfig


def resolve_sam3_chunk_size(config: DetectionAndTrackingConfig) -> int:
    """Return frames per SAM3 video session.

    ``chunked`` (default) resets on ``sam3_session_reset_s``.
    ``continuous`` keeps one session for the whole allowed clip
    (bounded by ``sam3_max_duration_s`` at the configured target FPS).
    """
    if config.sam3_tracking_mode == "continuous":
        return max(1, int(config.sam3_max_duration_s * config.sam3_target_fps) + 1)
    return max(1, int(config.sam3_session_reset_s * config.sam3_target_fps))


def resolve_sam3_frame_step(source_fps: float, target_fps: float) -> int:
    """Return source-frame stride so sampled rate does not exceed ``target_fps``.

    Uses ``ceil`` so slightly-above-target source rates (e.g. 7→5) never
    underestimate the stride as 1 and fill continuous sessions too early.
    """
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    if source_fps <= 0:
        raise ValueError("source_fps must be positive")
    return max(1, math.ceil(source_fps / target_fps))


def slugify_prompt(prompt: str) -> str:
    clean = prompt.strip().lower().replace(" ", "_")
    return clean or "object"


@dataclass
class TrackLabelAllocator:
    """Assign stable ``{label}_T{nnn}`` ids like sam3_reasoning video prop."""

    _next_by_label: dict[str, int] = field(default_factory=dict)

    def next_label(self, prompt: str) -> str:
        label = slugify_prompt(prompt)
        number = self._next_by_label.get(label, 1)
        self._next_by_label[label] = number + 1
        return f"{label}_T{number:03d}"


def format_sam3_overlay_label(
    *,
    label_style: str,
    display_id: str,
    prompt: str,
    track_label: str | None,
) -> str | None:
    """Return overlay text for the configured label style, or None to skip."""
    if label_style == "none":
        return None
    if label_style == "name":
        return str(prompt)
    if label_style == "track":
        return str(track_label or display_id)
    # default: chunk-local id style "#0:3"
    return f"#{display_id}"


__all__ = [
    "TrackLabelAllocator",
    "format_sam3_overlay_label",
    "resolve_sam3_chunk_size",
    "resolve_sam3_frame_step",
    "slugify_prompt",
]
