# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Parse upstream Visual QA / caption artifacts into ``PersonAttributes``.

The model-backed extraction is performed by the existing ``visual_qa`` service
(driven by the PAS attribute prompt/question bank). This module is the pure
adapter that turns that service's normalized ``items`` payload into the typed
PAS domain model, splitting ``"primary (fine)"`` colors and parsing the
accessory tuples.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from person_attribute_search.schema import Accessory, PersonAttributes, split_color

# Visual-QA question ids (normalized) that map onto a categorical attribute slot.
_SCALAR_FIELDS: frozenset[str] = frozenset(
    {
        "age",
        "gender",
        "skin_tone",
        "body_shape",
        "hair_length",
        "hair_style",
        "hair_color",
        "top_outer_length",
        "top_outer_type",
        "top_inner_length",
        "top_inner_type",
        "bottom_length",
        "bottom_type",
        "shoe_type",
        "headwear",
        "headwear_type",
        "mask",
        "glasses",
        "carrying",
        "motion_or_action",
        "potential_anomaly",
        "anomaly_type",
        "multiview_consistency",
        "image_quality",
        "viewpoint",
    }
)

# Color slots that carry a ``"primary (fine)"`` grade.
_COLOR_FIELDS: frozenset[str] = frozenset(
    {
        "top_outer_color",
        "top_inner_color",
        "bottom_color",
        "shoe_color",
        "headwear_color",
    }
)


def _normalize_key(raw: str) -> str:
    """Normalize a question id / answer key to a schema field name."""
    text = raw.strip().lower()
    for token in (" ", "-"):
        text = text.replace(token, "_")
    if text in {"natural_language_caption", "caption", "natural_caption"}:
        return "natural_caption"
    if text in {"action", "pose"}:
        return "motion_or_action"
    return text


def _coerce_accessories(value: Any) -> list[Accessory]:
    """Parse an accessories answer into ``Accessory`` objects."""
    if not isinstance(value, list):
        return []
    out: list[Accessory] = []
    for entry in value:
        if isinstance(entry, (list, tuple)) and len(entry) == 3:
            relationship, color, item = (str(part).strip() for part in entry)
        elif isinstance(entry, Mapping):
            relationship = str(entry.get("relationship") or "").strip()
            color = str(entry.get("color") or "").strip()
            item = str(entry.get("item") or "").strip()
        elif isinstance(entry, str):
            relationship = ""
            color = ""
            item = entry.strip()
        else:
            continue
        if item:
            out.append(Accessory(relationship=relationship, color=color, item=item))
    return out


def attributes_from_answers(answers: Mapping[str, Any]) -> PersonAttributes:
    """
    Build ``PersonAttributes`` from a flat answer mapping.

    Keys are question ids or attribute names (any of ``"top outer color"``,
    ``"top-outer-color"``, ``"top_outer_color"`` resolve to the same field).
    Unknown keys are ignored (taxonomy tolerance is handled downstream by
    ``schema.validate_attributes``).

    Args:
        answers: Mapping of question id/attribute name to answer value.

    Returns:
        A populated ``PersonAttributes`` instance.
    """
    fields: dict[str, Any] = {}
    for raw_key, raw_value in answers.items():
        key = _normalize_key(str(raw_key))
        if key == "accessories":
            fields["accessories"] = _coerce_accessories(raw_value)
            continue
        if raw_value is None:
            continue
        value = str(raw_value).strip()
        if not value:
            continue
        if key in _COLOR_FIELDS:
            primary, fine = split_color(value)
            fields[key] = primary
            if fine is not None:
                fields[f"{key}_fine"] = fine
        elif key in _SCALAR_FIELDS or key == "natural_caption":
            fields[key] = value

    return PersonAttributes(**fields)


def attributes_from_visual_qa_items(items: Iterable[Mapping[str, Any]]) -> PersonAttributes:
    """
    Build ``PersonAttributes`` from a normalized Visual QA ``items`` list.

    Args:
        items: Normalized visual-QA items, each with ``id`` and ``answer``.

    Returns:
        A populated ``PersonAttributes`` instance.
    """
    answers: dict[str, Any] = {}
    for item in items:
        qid = str(item.get("id") or "").strip()
        if not qid:
            continue
        answers[qid] = item.get("answer")
    return attributes_from_answers(answers)


__all__ = ["attributes_from_answers", "attributes_from_visual_qa_items"]
