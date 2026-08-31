# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-owned pipeline-state payloads for detection and tracking."""

from __future__ import annotations

from pydantic import BaseModel

TRACKING_ARTIFACTS_KEY = "detection_and_tracking"
"""Key used under ``ScenePipelineState.task_artifacts`` for tracking outputs."""


class TrackingArtifactsState(BaseModel):
    """Artifact references written by ``DetectionAndTrackingTask``."""

    success: bool = False
    objects_json: str | None = None
    instances_json: str | None = None
    detection_overlay_path: str | None = None
    tracking_overlay_path: str | None = None
    annotated_video_path: str | None = None
    red_id_overlay_path: str | None = None
    tracks_json: str | None = None
    crops_root: str | None = None


__all__ = ["TRACKING_ARTIFACTS_KEY", "TrackingArtifactsState"]
