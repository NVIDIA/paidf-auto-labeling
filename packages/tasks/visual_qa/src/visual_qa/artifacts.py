# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visual-QA-owned pipeline artifact models."""

from __future__ import annotations

from pydantic import BaseModel

VISUAL_QA_ARTIFACTS_KEY = "visual_qa"


class VisualQaArtifactsState(BaseModel):
    """State slice for visual QA sidecars consumed by DAFT export."""

    success: bool = False
    items_json: str | None = None
    windows_json: str | None = None
    source_sidecar: str | None = None


__all__ = ["VISUAL_QA_ARTIFACTS_KEY", "VisualQaArtifactsState"]
