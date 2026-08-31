# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipeline-state bookkeeping for the Person Attribute Search task."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

PERSON_ATTRIBUTE_SEARCH_ARTIFACTS_KEY = "person_attribute_search"


class PersonAttributeSearchArtifactsState(BaseModel):
    """Recorded under ``ScenePipelineState.task_artifacts`` for this task."""

    model_config = ConfigDict(extra="forbid")

    success: bool
    attributes_json: str | None = None
    bundle_attributes_json: str | None = None
    queries_json: str | None = None
    bundle_queries_json: str | None = None
    hitl_json: str | None = None
    bundle_hitl_json: str | None = None
    # Per-chunk (per-track video) outputs.
    pas_json: str | None = None
    chunk_queries_json: str | None = None
    n_people: int | None = None
    warnings: list[str] = Field(default_factory=list)
    optional_failures: list[str] = Field(default_factory=list)


__all__ = [
    "PERSON_ATTRIBUTE_SEARCH_ARTIFACTS_KEY",
    "PersonAttributeSearchArtifactsState",
]
