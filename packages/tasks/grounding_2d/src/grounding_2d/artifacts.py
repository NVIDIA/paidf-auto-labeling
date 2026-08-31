# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-owned pipeline-state payloads for 2D grounding."""

from __future__ import annotations

from pydantic import BaseModel

GROUNDING_2D_ARTIFACTS_KEY = "grounding_2d"
"""Key used under ``ScenePipelineState.task_artifacts`` for 2D grounding outputs."""


class Grounding2DArtifactsState(BaseModel):
    """Artifact references written by ``Grounding2DTask``."""

    success: bool = False
    artifact_json: str | None = None
    final_json: str | None = None
    expression_json: str | None = None
    grounding_json: str | None = None
    segmentation_json: str | None = None
    expression_count: int = 0
    instance_count: int = 0
    has_masks: bool = False


__all__ = ["GROUNDING_2D_ARTIFACTS_KEY", "Grounding2DArtifactsState"]
