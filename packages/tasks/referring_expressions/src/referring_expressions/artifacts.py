# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-owned pipeline-state payloads for referring expressions."""

from __future__ import annotations

from pydantic import BaseModel

REFERRING_EXPRESSIONS_ARTIFACTS_KEY = "referring_expressions"


class ReferringExpressionsArtifactsState(BaseModel):
    """Artifact references written by ``ReferringExpressionsTask``."""

    success: bool = False
    artifact_json: str | None = None
    region_json: str | None = None
    region_count: int = 0


__all__ = [
    "REFERRING_EXPRESSIONS_ARTIFACTS_KEY",
    "ReferringExpressionsArtifactsState",
]
