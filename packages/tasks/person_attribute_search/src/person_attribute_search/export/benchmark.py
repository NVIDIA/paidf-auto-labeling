# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Legacy PAS benchmark JSON builders.

Produces the ``generated_attributes_v3`` / ``generated_queries_v3`` envelope
shapes so the UPA output is shape-compatible with the artifacts consumers
already ingest (values are model-dependent and the envelope carries a
``generated_at`` timestamp, so it is not byte-identical). Pure dict
construction; the task layer owns writing the files.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime

from person_attribute_search.queries import QuerySet
from person_attribute_search.schema import Accessory, PersonAttributes


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _accessory_tuple(accessory: Accessory) -> list[str]:
    return [accessory.relationship, accessory.color, accessory.item]


def attributes_to_legacy_dict(attributes: PersonAttributes) -> dict[str, object]:
    """Render ``PersonAttributes`` into the legacy flat ``attributes`` dict."""
    out: dict[str, object] = {}
    simple_fields = (
        "age",
        "gender",
        "skin_tone",
        "body_shape",
        "hair_length",
        "hair_style",
        "hair_color",
        "top_outer_length",
        "top_outer_type",
        "top_outer_color",
        "top_inner_length",
        "top_inner_type",
        "top_inner_color",
        "bottom_length",
        "bottom_type",
        "bottom_color",
        "shoe_type",
        "shoe_color",
        "headwear",
        "headwear_type",
        "headwear_color",
        "mask",
        "glasses",
        "carrying",
        "motion_or_action",
        "potential_anomaly",
        "anomaly_type",
        "multiview_consistency",
        "image_quality",
        "viewpoint",
    )
    for field_name in simple_fields:
        value = getattr(attributes, field_name)
        if value is not None:
            out[field_name.replace("_", " ")] = value

    for fine_field in (
        "top_outer_color_fine",
        "top_inner_color_fine",
        "bottom_color_fine",
        "shoe_color_fine",
        "headwear_color_fine",
    ):
        value = getattr(attributes, fine_field)
        if value is not None:
            label = fine_field.replace("_fine", "").replace("_", " ") + " (fine)"
            out[label] = value

    if attributes.accessories:
        out["accessories"] = [_accessory_tuple(a) for a in attributes.accessories]
    return out


def build_metadata(*, source_file: str, model: str, prompt: str = "") -> dict[str, object]:
    """Build the envelope ``metadata`` block."""
    return {
        "source_file": source_file,
        "model": {"name": model},
        "prompt": prompt,
        "generated_at": _now_iso(),
    }


def build_attribute_entry(
    *,
    person_key: str,
    person_id: str,
    dataset: str,
    images: dict[str, list[str]],
    attributes: PersonAttributes,
) -> dict[str, object]:
    """Build one ``generated_attributes_v3`` entry."""
    entry: dict[str, object] = {
        "person_key": person_key,
        "person_id": person_id,
        "dataset": dataset,
        "images": images,
        "attributes": attributes_to_legacy_dict(attributes),
    }
    if attributes.natural_caption:
        entry["natural_caption"] = attributes.natural_caption
    return entry


def build_query_entry(
    *,
    person_key: str,
    person_id: str,
    dataset: str,
    attributes: PersonAttributes,
    query_set: QuerySet,
    flat_query_tiers: bool = False,
) -> dict[str, object]:
    """Build one tiered-query entry.

    Legacy query generation represents easy and medium entries as
    ``[query, attributes_used]`` pairs. Bundle generation has no
    ``attributes_used`` value and returns strings in every tier, so callers can
    request ``flat_query_tiers`` rather than emitting placeholder empty strings.
    """
    entry: dict[str, object] = {
        "person_key": person_key,
        "person_id": person_id,
        "dataset": dataset,
        "attributes": attributes_to_legacy_dict(attributes),
        "queries": (query_set.to_flat_dict() if flat_query_tiers else query_set.to_legacy_dict()),
    }
    if attributes.natural_caption:
        entry["natural_caption"] = attributes.natural_caption
    return entry


def build_result_json(
    *,
    metadata: dict[str, object],
    entries: Iterable[dict[str, object]],
) -> dict[str, object]:
    """Wrap entries in the standard ``{metadata, entries}`` envelope."""
    return {"metadata": metadata, "entries": list(entries)}


__all__ = [
    "attributes_to_legacy_dict",
    "build_attribute_entry",
    "build_metadata",
    "build_query_entry",
    "build_result_json",
]
