# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tiered query-bundle export (PAS image-augmentation flow).

Turns the per-person :class:`QuerySet` objects assembled by ``tracks.build_people``
into the image-augmentation deliverables: one aggregated document keyed by
``<person_key>`` and one per-person document each, all in the
``{"queries": {"easy": [...], "medium": [...], "hard": [...]}}`` shape.

This is pure dict shaping: no model calls and no filesystem access.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import PurePath
from typing import Any

from person_attribute_search.bundle_queries import query_set_to_bundle
from person_attribute_search.export.benchmark import attributes_to_legacy_dict
from person_attribute_search.export.hitl import build_preannotations
from person_attribute_search.queries import QuerySet
from person_attribute_search.sources import AttributeImageSource


def _person_key(entry: dict[str, Any]) -> str:
    """Resolve the stable per-person key, falling back to the track id."""
    object_id = entry.get("object_id")
    if isinstance(object_id, str) and object_id.strip():
        return object_id.strip()
    return f"track_{int(entry.get('track_id', 0)):04d}"


def build_bundle_documents(
    *,
    chunk_id: str,
    people: Sequence[dict[str, Any]],
    query_sets: Sequence[QuerySet],
) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any]]]]:
    """Build the aggregated and per-person query-bundle documents.

    Args:
        chunk_id: Identifier of the scene/chunk.
        people: Per-person ``people[]`` entries from ``tracks.build_people``.
        query_sets: Per-person query sets aligned by index with ``people``.

    Returns:
        ``(aggregated, per_person)`` where ``aggregated`` maps each person key to
        its tiered queries and ``per_person`` is a list of
        ``(filename, document)`` pairs (one JSON document per person).
    """
    if len(people) != len(query_sets):
        raise ValueError(
            "people and query_sets must be the same length; "
            f"got len(people)={len(people)}, len(query_sets)={len(query_sets)}"
        )
    aggregated_people: dict[str, Any] = {}
    per_person: list[tuple[str, dict[str, Any]]] = []
    for entry, query_set in zip(people, query_sets, strict=True):
        key = _person_key(entry)
        bundle = query_set_to_bundle(query_set)
        aggregated_people[key] = {
            "track_id": entry.get("track_id"),
            "queries": bundle,
        }
        per_person.append(
            (
                f"{key}.json",
                {
                    "person_key": key,
                    "track_id": entry.get("track_id"),
                    "queries": bundle,
                },
            )
        )
    aggregated = {
        "chunk_id": chunk_id,
        "n_people": len(aggregated_people),
        "people": aggregated_people,
    }
    return aggregated, per_person


def build_image_bundle_document(
    *,
    chunk_id: str,
    sources: Sequence[AttributeImageSource],
    query_sets: Sequence[QuerySet],
) -> dict[str, Any]:
    """Build one bundle entry per explicit attribute image.

    A unique ``person_key`` remains the map key. When several images share that
    key, their ``image_id`` values become the keys so none are overwritten.
    The enclosing map key identifies the person/image and is shared with the
    attribute and HITL bundles, so each value contains only ``image_filename``
    and ``queries``.
    """
    if len(sources) != len(query_sets):
        raise ValueError(
            "sources and query_sets must be the same length; "
            f"got len(sources)={len(sources)}, len(query_sets)={len(query_sets)}"
        )
    people: dict[str, Any] = {}
    assignments = _image_bundle_assignments(sources)
    for (key, _track_id, source), query_set in zip(assignments, query_sets, strict=True):
        people[key] = {
            "image_filename": PurePath(source.image_id).name,
            "queries": query_set_to_bundle(query_set),
        }
    return {"chunk_id": chunk_id, "n_people": len(people), "people": people}


def build_image_attribute_bundle_document(
    *,
    chunk_id: str,
    sources: Sequence[AttributeImageSource],
) -> dict[str, Any]:
    """Build a minimal per-image attribute bundle aligned with query bundles."""
    people = {
        key: {
            "track_id": track_id,
            "attributes": _image_bundle_attributes(source),
        }
        for key, track_id, source in _image_bundle_assignments(sources)
    }
    return {"chunk_id": chunk_id, "n_people": len(people), "people": people}


def build_image_hitl_bundle_document(
    *,
    chunk_id: str,
    sources: Sequence[AttributeImageSource],
    query_sets: Sequence[QuerySet],
    image_url_base: str = "",
) -> dict[str, Any]:
    """Build one HITL preannotation entry per explicit attribute image."""
    if len(sources) != len(query_sets):
        raise ValueError(
            "sources and query_sets must be the same length; "
            f"got len(sources)={len(sources)}, len(query_sets)={len(query_sets)}"
        )
    people = {
        key: {
            "track_id": track_id,
            "image_url": f"{image_url_base}{source.image_id}",
            "preannotations": build_preannotations(
                query_set,
                source.attributes.natural_caption,
            ),
        }
        for (key, track_id, source), query_set in zip(
            _image_bundle_assignments(sources), query_sets, strict=True
        )
    }
    return {"chunk_id": chunk_id, "n_people": len(people), "people": people}


def _image_bundle_attributes(source: AttributeImageSource) -> dict[str, object]:
    """Serialize attributes while preserving the source accessories list."""
    attributes = attributes_to_legacy_dict(source.attributes)
    raw_attributes = source.source_entry.get("attributes")
    if isinstance(raw_attributes, dict) and "accessories" in raw_attributes:
        raw_accessories = raw_attributes["accessories"]
        attributes["accessories"] = (
            list(raw_accessories) if isinstance(raw_accessories, list) else []
        )
    return attributes


def _image_bundle_assignments(
    sources: Sequence[AttributeImageSource],
) -> list[tuple[str, Any, AttributeImageSource]]:
    """Assign identical map keys and track ids across image bundle documents."""
    key_counts: dict[str, int] = {}
    for source in sources:
        key_counts[source.person_key] = key_counts.get(source.person_key, 0) + 1

    assignments: list[tuple[str, Any, AttributeImageSource]] = []
    used_keys: set[str] = set()
    for fallback_track_id, source in enumerate(sources):
        preferred_key = source.person_key if key_counts[source.person_key] == 1 else source.image_id
        key = preferred_key
        suffix = 2
        while key in used_keys:
            key = f"{preferred_key}#{suffix}"
            suffix += 1
        used_keys.add(key)
        assignments.append((key, source.source_entry.get("track_id", fallback_track_id), source))
    return assignments


__all__ = [
    "build_bundle_documents",
    "build_image_attribute_bundle_document",
    "build_image_bundle_document",
    "build_image_hitl_bundle_document",
]
