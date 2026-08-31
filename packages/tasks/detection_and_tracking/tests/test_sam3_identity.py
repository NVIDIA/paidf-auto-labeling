# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SAM3 identity helpers."""

from __future__ import annotations

import pytest
from detection_and_tracking.backends.sam3_identity import (
    TrackLabelAllocator,
    format_sam3_overlay_label,
    resolve_sam3_chunk_size,
    resolve_sam3_frame_step,
)
from detection_and_tracking.config import DetectionAndTrackingConfig


def test_chunked_mode_uses_session_reset() -> None:
    config = DetectionAndTrackingConfig(
        sam3_tracking_mode="chunked",
        sam3_session_reset_s=10.0,
        sam3_target_fps=5.0,
        sam3_max_duration_s=60.0,
    )
    assert resolve_sam3_chunk_size(config) == 50


def test_continuous_mode_uses_max_duration_window() -> None:
    config = DetectionAndTrackingConfig(
        sam3_tracking_mode="continuous",
        sam3_session_reset_s=2.0,
        sam3_target_fps=10.0,
        sam3_max_duration_s=6.0,
    )
    assert resolve_sam3_chunk_size(config) == 61


def test_continuous_mode_non_divisible_fps_stays_single_session() -> None:
    """source 7 / target 5 must ceil the stride so continuous stays one window."""
    config = DetectionAndTrackingConfig(
        sam3_tracking_mode="continuous",
        sam3_session_reset_s=2.0,
        sam3_target_fps=5.0,
        sam3_max_duration_s=6.0,
    )
    # round(7/5)=1 would undersample-stride and fill the window too early;
    # ceil keeps stride 2 and one continuous session for the allowed clip.
    assert resolve_sam3_frame_step(7.0, config.sam3_target_fps) == 2
    assert resolve_sam3_frame_step(5.0, config.sam3_target_fps) == 1
    assert resolve_sam3_frame_step(10.0, config.sam3_target_fps) == 2
    assert resolve_sam3_chunk_size(config) == 31


def test_frame_step_rejects_non_positive_rates() -> None:
    with pytest.raises(ValueError, match="target_fps"):
        resolve_sam3_frame_step(30.0, 0.0)
    with pytest.raises(ValueError, match="source_fps"):
        resolve_sam3_frame_step(0.0, 5.0)


def test_track_label_allocator_matches_reasoning_style() -> None:
    allocator = TrackLabelAllocator()
    assert allocator.next_label("person") == "person_T001"
    assert allocator.next_label("person") == "person_T002"
    assert allocator.next_label("forklift") == "forklift_T001"


def test_overlay_label_styles() -> None:
    assert (
        format_sam3_overlay_label(
            label_style="id",
            display_id="0:3",
            prompt="person",
            track_label="person_T001",
        )
        == "#0:3"
    )
    assert (
        format_sam3_overlay_label(
            label_style="track",
            display_id="0:3",
            prompt="person",
            track_label="person_T001",
        )
        == "person_T001"
    )
    assert (
        format_sam3_overlay_label(
            label_style="name",
            display_id="0:3",
            prompt="person",
            track_label="person_T001",
        )
        == "person"
    )
    assert (
        format_sam3_overlay_label(
            label_style="none",
            display_id="0:3",
            prompt="person",
            track_label="person_T001",
        )
        is None
    )
