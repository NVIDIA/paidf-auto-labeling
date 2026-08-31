# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Deterministic template engine for easy/medium retrieval queries.

Easy and medium PAS queries are produced *without* an LLM by filling attribute
slots into short phrases. Determinism (fixed slot ordering, no RNG by default)
keeps the engine unit-testable and reproducible; callers that want the legacy
probabilistic behavior can pass their own ``random.Random``.

Each query is returned as a ``(query_text, attributes_used)`` pair, matching the
legacy PAS JSON contract where ``attributes_used`` is a comma-joined provenance
label.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import pairwise

from person_attribute_search.schema import PersonAttributes

# (display_label, color_field, type_field) for the garment slots usable in
# template queries, in deterministic priority order.
_GARMENT_SLOTS: tuple[tuple[str, str, str], ...] = (
    ("top outer", "top_outer_color", "top_outer_type"),
    ("bottom", "bottom_color", "bottom_type"),
    ("shoe", "shoe_color", "shoe_type"),
)

QueryPair = tuple[str, str]


def _apply_stop_words(text: str, stop_words: frozenset[str]) -> str:
    """Drop stop-word tokens from a phrase, preserving order and spacing."""
    if not stop_words:
        return text
    kept = [token for token in text.split() if token.lower() not in stop_words]
    return " ".join(kept)


def _garment_phrase(attributes: PersonAttributes, color_field: str, type_field: str) -> str | None:
    """Build a ``"<color> <type>"`` phrase for a garment slot, if populated."""
    color = getattr(attributes, color_field)
    garment = getattr(attributes, type_field)
    parts = [part.strip() for part in (color, garment) if part and part.strip()]
    return " ".join(parts) if parts else None


def _normalize_stop_words(stop_words: Iterable[str] | None) -> frozenset[str]:
    if not stop_words:
        return frozenset()
    return frozenset(word.strip().lower() for word in stop_words if word.strip())


def build_easy_queries(
    attributes: PersonAttributes,
    *,
    count: int = 2,
    stop_words: Iterable[str] | None = None,
) -> list[QueryPair]:
    """
    Generate single-garment "easy" queries (one color+type slot each).

    Args:
        attributes: Structured person attributes.
        count: Maximum number of easy queries to emit.
        stop_words: Optional words to strip from generated phrases.

    Returns:
        Up to ``count`` ``(query, attributes_used)`` pairs, in slot priority
        order, skipping slots with no usable color/type.
    """
    blocked = _normalize_stop_words(stop_words)
    out: list[QueryPair] = []
    for label, color_field, type_field in _GARMENT_SLOTS:
        if len(out) >= count:
            break
        phrase = _garment_phrase(attributes, color_field, type_field)
        if phrase is None:
            continue
        cleaned = _apply_stop_words(phrase, blocked)
        if not cleaned:
            continue
        used = f"{label} color, {label} type"
        out.append((cleaned, used))
    return out


def build_medium_queries(
    attributes: PersonAttributes,
    *,
    count: int = 2,
    stop_words: Iterable[str] | None = None,
) -> list[QueryPair]:
    """
    Generate 2-4 attribute "medium" queries (two garments, or garment+accessory).

    Args:
        attributes: Structured person attributes.
        count: Maximum number of medium queries to emit.
        stop_words: Optional words to strip from generated phrases.

    Returns:
        Up to ``count`` ``(query, attributes_used)`` pairs.
    """
    blocked = _normalize_stop_words(stop_words)
    garments: list[tuple[str, str]] = []
    for label, color_field, type_field in _GARMENT_SLOTS:
        phrase = _garment_phrase(attributes, color_field, type_field)
        if phrase is not None:
            garments.append((label, phrase))

    out: list[QueryPair] = []

    # Pair consecutive garments: (top outer + bottom), (bottom + shoe), ...
    for first, second in pairwise(garments):
        if len(out) >= count:
            break
        phrase = _apply_stop_words(f"{first[1]} and {second[1]}", blocked)
        if not phrase:
            continue
        used = f"{first[0]} color, {first[0]} type, {second[0]} color, {second[0]} type"
        out.append((phrase, used))

    # Fall back to garment + first usable accessory when more queries are needed.
    # Skip empty accessory descriptors so a blank leading entry never masks a
    # later, usable one.
    if len(out) < count and garments:
        for accessory in attributes.accessories:
            descriptor = f"{accessory.color} {accessory.item}".strip()
            if not descriptor:
                continue
            phrase = _apply_stop_words(f"{garments[0][1]} with {descriptor}", blocked)
            if phrase:
                label = garments[0][0]
                used = f"{label} color, {label} type, accessories color, accessories object"
                out.append((phrase, used))
            break

    return out[:count]


__all__ = ["QueryPair", "build_easy_queries", "build_medium_queries"]
