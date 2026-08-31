# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Captioning-owned pipeline artifact models."""

from __future__ import annotations

from pydantic import BaseModel

CAPTION_ARTIFACTS_KEY = "captioning"


class CaptionArtifactsState(BaseModel):
    """State slice for caption artifacts consumed by downstream tasks."""

    success: bool = False
    metadata_chunk_json: str | None = None
    video_json: str | None = None
    events_json: str | None = None
    image_json: str | None = None


__all__ = ["CAPTION_ARTIFACTS_KEY", "CaptionArtifactsState"]
