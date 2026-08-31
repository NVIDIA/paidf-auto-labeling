# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Difficulty-tiered query assembly.

Combines the deterministic template engine (easy/medium) with LLM/VLM-produced
"hard" queries (supplied by upstream UPA services) into a single ``QuerySet``.
When no hard queries are available upstream, a deterministic caption-derived
fallback keeps the contract populated so downstream export never sees an empty
tier.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from person_attribute_search.schema import PersonAttributes
from person_attribute_search.templates import (
    QueryPair,
    build_easy_queries,
    build_medium_queries,
)

_HARD_MAX_WORDS = 15


class QuerySet(BaseModel):
    """Easy/medium/hard retrieval queries for one person or image."""

    model_config = ConfigDict(extra="forbid")

    easy: list[QueryPair] = Field(default_factory=list)
    medium: list[QueryPair] = Field(default_factory=list)
    hard: list[str] = Field(default_factory=list)

    def to_legacy_dict(self) -> dict[str, object]:
        """Serialize to the legacy PAS ``queries`` JSON shape."""
        return {
            "easy": [[query, used] for query, used in self.easy],
            "medium": [[query, used] for query, used in self.medium],
            "hard": list(self.hard),
        }

    def to_flat_dict(self) -> dict[str, list[str]]:
        """Serialize every query tier as a list of query strings."""
        return {
            "easy": [query for query, _used in self.easy],
            "medium": [query for query, _used in self.medium],
            "hard": list(self.hard),
        }


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words])


def collect_hard_queries(
    items: Iterable[Mapping[str, Any]],
    hard_query_ids: Iterable[str],
) -> list[str]:
    """Pull non-empty upstream hard-query answers from normalized QA items."""
    wanted = {str(query_id).strip() for query_id in hard_query_ids if str(query_id).strip()}
    out: list[str] = []
    for item in items:
        if str(item.get("id") or "").strip() in wanted:
            answer = str(item.get("answer") or "").strip()
            if answer:
                out.append(answer)
    return out


def _fallback_hard_queries(attributes: PersonAttributes) -> list[str]:
    """Derive deterministic hard queries when none are supplied upstream."""
    queries: list[str] = []
    caption = (attributes.natural_caption or "").strip()
    if caption:
        queries.append(_truncate_words(caption, _HARD_MAX_WORDS))

    descriptors: list[str] = []
    for color_field, type_field in (
        ("top_outer_color_fine", "top_outer_type"),
        ("bottom_color_fine", "bottom_type"),
        ("shoe_color_fine", "shoe_type"),
    ):
        color = getattr(attributes, color_field) or ""
        garment = getattr(attributes, type_field) or ""
        descriptor = f"{color} {garment}".strip()
        if descriptor:
            descriptors.append(descriptor)
    for accessory in attributes.accessories:
        descriptor = f"{accessory.color} {accessory.item}".strip()
        if descriptor:
            descriptors.append(descriptor)
    if descriptors:
        query = _truncate_words(" ".join(descriptors), _HARD_MAX_WORDS)
        if query:
            queries.append(query)

    if not queries:
        # Never return an empty hard tier; downstream export relies on a
        # populated contract even when no caption or descriptors are available.
        queries.append("person")

    return queries


def normalize_anomaly_label(label: str) -> str:
    """Turn an anomaly label token into a readable query phrase.

    ``"warehouse_safety_violation"`` -> ``"warehouse safety violation"``.
    """
    return label.strip().replace("_", " ").strip()


def flatten_queries(
    query_sets: Iterable[QuerySet],
    *,
    anomaly_labels: Iterable[str] | None = None,
    extra_queries: Iterable[str] | None = None,
) -> list[str]:
    """
    Flatten per-person tiered queries into one de-duplicated chunk-level list.

    Combines, in order, every easy/medium/hard query text across ``query_sets``,
    then anomaly-derived queries, then any extra queries (e.g. caption-derived).
    Order is preserved and duplicates are dropped.

    Args:
        query_sets: Per-person assembled query sets.
        anomaly_labels: Anomaly labels to add as queries (normalized to phrases).
        extra_queries: Additional query strings (e.g. a scene caption sentence).

    Returns:
        A de-duplicated, ordered list of query strings.
    """
    ordered: list[str] = []
    seen: set[str] = set()

    def _add(value: str) -> None:
        text = value.strip()
        if text and text not in seen:
            seen.add(text)
            ordered.append(text)

    for query_set in query_sets:
        for query, _used in (*query_set.easy, *query_set.medium):
            _add(query)
        for query in query_set.hard:
            _add(query)
    for label in anomaly_labels or []:
        _add(normalize_anomaly_label(label))
    for query in extra_queries or []:
        _add(query)

    return ordered


def assemble_query_set(
    attributes: PersonAttributes,
    *,
    hard_queries: Iterable[str] | None = None,
    easy_count: int = 2,
    medium_count: int = 2,
    stop_words: Iterable[str] | None = None,
) -> QuerySet:
    """
    Build a full ``QuerySet`` for an identity or image.

    Args:
        attributes: Structured person attributes.
        hard_queries: Hard queries produced by an upstream VLM/LLM service. When
            ``None`` or empty, a deterministic caption-derived fallback is used.
        easy_count: Maximum easy queries.
        medium_count: Maximum medium queries.
        stop_words: Optional words to strip from template phrases.

    Returns:
        A populated ``QuerySet``.
    """
    easy = build_easy_queries(attributes, count=easy_count, stop_words=stop_words)
    medium = build_medium_queries(attributes, count=medium_count, stop_words=stop_words)

    hard = [query.strip() for query in (hard_queries or []) if query and query.strip()]
    if not hard:
        hard = _fallback_hard_queries(attributes)

    return QuerySet(easy=easy, medium=medium, hard=hard)


__all__ = [
    "QuerySet",
    "assemble_query_set",
    "collect_hard_queries",
    "flatten_queries",
    "normalize_anomaly_label",
]
