# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DAFT type constants and canonical file routing."""

from __future__ import annotations

from typing import Literal

ContextualType = Literal[
    "calibration",
    "chunks",
    "events",
    "image",
    "instances",
    "msted",
    "objects",
    "pas_queries",
    "person_attributes",
    "tracking",
    "video",
]
TaskType = Literal[
    "bcq",
    "bcq_openended",
    "causal_linkage",
    "mcq",
    "mcq_openended",
    "open_qa",
    "scene_description",
    "temporal_description",
    "temporal_localization",
    "video_summarization",
]
DaftType = ContextualType | TaskType
DaftKind = Literal["contextual", "task"]

CONTEXTUAL_TYPES: tuple[ContextualType, ...] = (
    "calibration",
    "chunks",
    "events",
    "image",
    "instances",
    "msted",
    "objects",
    "pas_queries",
    "person_attributes",
    "tracking",
    "video",
)
TASK_TYPES: tuple[TaskType, ...] = (
    "bcq",
    "bcq_openended",
    "causal_linkage",
    "mcq",
    "mcq_openended",
    "open_qa",
    "scene_description",
    "temporal_description",
    "temporal_localization",
    "video_summarization",
)

CONTEXTUAL_DEFAULT_FILENAMES: dict[ContextualType, str] = {
    type_name: f"{type_name}.json" for type_name in CONTEXTUAL_TYPES
}
TASK_DEFAULT_FILENAMES: dict[TaskType, str] = {
    type_name: f"{type_name}.json" for type_name in TASK_TYPES
}


def daft_kind(type_name: str) -> DaftKind:
    """Return the canonical DAFT directory kind for ``type_name``.

    Args:
        type_name: DAFT ``metadata.type`` value.
    Returns:
        ``"contextual"`` for contextual annotation types or ``"task"`` for task types.
    Raises:
        ValueError: If ``type_name`` is not a known DAFT v3 annotation type.
    """
    if type_name in CONTEXTUAL_TYPES:
        return "contextual"
    if type_name in TASK_TYPES:
        return "task"
    raise ValueError(f"Unknown DAFT metadata.type {type_name!r}")


__all__ = [
    "CONTEXTUAL_DEFAULT_FILENAMES",
    "CONTEXTUAL_TYPES",
    "TASK_DEFAULT_FILENAMES",
    "TASK_TYPES",
    "ContextualType",
    "DaftKind",
    "DaftType",
    "TaskType",
    "daft_kind",
]
