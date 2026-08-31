# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Person Attribute Search (PAS) assembly task.

A model-free orchestration task that turns the outputs of existing UPA services
(``detection_and_tracking``, ``super_resolution``, ``captioning``, ``visual_qa``,
``reasoning``) into PAS-parity artifacts: structured attributes, difficulty-tiered
retrieval queries, and data-factory HITL preannotations.
"""

from __future__ import annotations

from person_attribute_search.config import PersonAttributeSearchConfig
from person_attribute_search.prompts import (
    ATTRIBUTES_PROMPT,
    IMAGE_QUERIES_PROMPT,
    QUERIES_PROMPT,
    load_prompt,
    prompt_path,
    render_image_query_prompt,
    render_query_prompt,
)
from person_attribute_search.queries import (
    QuerySet,
    assemble_query_set,
    flatten_queries,
)
from person_attribute_search.schema import (
    Accessory,
    PersonAttributes,
    validate_attributes,
)
from person_attribute_search.task import PersonAttributeSearchTask
from person_attribute_search.track_inputs import assemble_track_inputs
from person_attribute_search.tracks import TrackRecord, build_people

__all__ = [
    "ATTRIBUTES_PROMPT",
    "IMAGE_QUERIES_PROMPT",
    "QUERIES_PROMPT",
    "Accessory",
    "PersonAttributeSearchConfig",
    "PersonAttributeSearchTask",
    "PersonAttributes",
    "QuerySet",
    "TrackRecord",
    "assemble_query_set",
    "assemble_track_inputs",
    "build_people",
    "flatten_queries",
    "load_prompt",
    "prompt_path",
    "render_image_query_prompt",
    "render_query_prompt",
    "validate_attributes",
]
