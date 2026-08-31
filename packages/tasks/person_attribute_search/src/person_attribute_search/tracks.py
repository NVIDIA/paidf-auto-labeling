# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Per-track aggregation (identity == SAM3 track).

In the video flow, one *person* is a tracking ``track_id`` followed across a
chunk. The upstream producers — detection/tracking crop extraction (#1) and the
per-track Visual QA fan-out (#2) — assemble a normalized ``track_inputs.json``
seam that this module consumes. Each ``TrackRecord`` carries the track's
bookkeeping plus its per-track Visual QA ``items``; this module turns each into
one ``people[]`` entry (attributes + natural caption), reusing the pure parsing
and query modules.

This is pure assembly: no model calls, no filesystem, no video decoding.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from person_attribute_search.attributes import attributes_from_visual_qa_items
from person_attribute_search.export.benchmark import attributes_to_legacy_dict
from person_attribute_search.queries import QuerySet, assemble_query_set, collect_hard_queries
from person_attribute_search.schema import PersonAttributes

# A pluggable per-person query builder: ``(attributes, upstream_hard) ->
# QuerySet``. Used to swap the deterministic assembler for the LLM-backed
# generator (legacy ``generate_queries.py`` parity) without changing this seam.
QuerySetBuilder = Callable[[PersonAttributes, list[str]], QuerySet]


class TrackRecord(BaseModel):
    """One track's bookkeeping plus its per-track Visual QA answers."""

    model_config = ConfigDict(extra="forbid")

    track_id: int
    # Stable per-identity label from the upstream producer (``object_id`` in
    # ``tracks.json``). In the image-augmentation flow this is the ``<person_key>``
    # directory name; downstream aggregation keys per-person outputs by it,
    # falling back to ``track_id`` when absent.
    object_id: str | None = None
    detection_score: float | None = None
    duration_sec: float | None = None
    first_frame: int | None = None
    last_frame: int | None = None
    n_crops: int | None = None
    n_crops_in_chunk: int | None = None
    first_frame_in_chunk: int | None = None
    last_frame_in_chunk: int | None = None
    crop_dir: str | None = None
    caption: str | None = None
    items: list[dict[str, Any]] = Field(default_factory=list)
    # Per-track retrieval queries produced by the per-track VLM/LLM fan-out. When
    # present they flow into the chunk queries verbatim (preserving the exact
    # legacy generative output); when empty the deterministic template engine
    # supplies easy/medium/hard queries instead.
    queries: list[str] = Field(default_factory=list)


def track_record_from_mapping(record: Mapping[str, Any]) -> TrackRecord:
    """Build a ``TrackRecord`` from a plain mapping, ignoring unknown keys."""
    known = set(TrackRecord.model_fields)
    fields = {key: value for key, value in record.items() if key in known}
    return TrackRecord.model_validate(fields)


def build_person_entry(
    track: TrackRecord,
    *,
    easy_count: int = 2,
    medium_count: int = 2,
    stop_words: Iterable[str] | None = None,
    hard_query_ids: Iterable[str] = (),
    query_builder: QuerySetBuilder | None = None,
) -> tuple[dict[str, Any], QuerySet]:
    """
    Build one per-chunk ``people[]`` entry plus its assembled ``QuerySet``.

    Query precedence (see the ``build_person_entry`` body):

    * If the track carries pre-generated ``track.queries`` (legacy explicit
      seam), only ``hard`` is populated from them; ``easy``/``medium`` from the
      deterministic templates are skipped.
    * Else if a ``query_builder`` is supplied (LLM tiered-query generation), it
      produces the full query set from the track attributes plus the upstream
      hard queries.
    * Otherwise the deterministic assembler supplies ``easy``/``medium`` and
      ``hard`` is sourced from the Visual QA ``items`` referenced by
      ``hard_query_ids``.

    Args:
        track: The track record with per-track Visual QA items (and optional
            pre-generated ``queries``).
        easy_count: Maximum easy template queries.
        medium_count: Maximum medium template queries.
        stop_words: Optional words to strip from template phrases.
        hard_query_ids: Visual-QA item ids whose answers are hard queries.
        query_builder: Optional ``(attributes, upstream_hard) -> QuerySet``
            override (e.g. the LLM-backed generator).

    Returns:
        A ``(person_entry, query_set)`` tuple. ``person_entry`` mirrors the
        per-chunk ``pas.people[]`` shape; ``query_set`` is returned so the caller
        can flatten chunk-level queries.
    """
    attributes = attributes_from_visual_qa_items(track.items)
    natural_caption = attributes.natural_caption or (track.caption or "")
    if natural_caption and not attributes.natural_caption:
        attributes = attributes.model_copy(update={"natural_caption": natural_caption})

    supplied = [query.strip() for query in track.queries if query and query.strip()]
    upstream_hard = collect_hard_queries(track.items, hard_query_ids)
    if supplied:
        query_set = QuerySet(hard=supplied)
    elif query_builder is not None:
        query_set = query_builder(attributes, upstream_hard)
    else:
        query_set = assemble_query_set(
            attributes,
            hard_queries=upstream_hard,
            easy_count=easy_count,
            medium_count=medium_count,
            stop_words=stop_words,
        )

    legacy_attributes = attributes_to_legacy_dict(attributes)
    if natural_caption:
        legacy_attributes["natural language caption"] = natural_caption

    entry: dict[str, Any] = {"track_id": track.track_id}
    if track.object_id:
        entry["object_id"] = track.object_id
    for key in ("detection_score", "duration_sec", "first_frame", "last_frame", "n_crops"):
        value = getattr(track, key)
        if value is not None:
            entry[key] = value
    entry["attributes"] = legacy_attributes
    entry["natural_language_caption"] = natural_caption
    for key in ("n_crops_in_chunk", "first_frame_in_chunk", "last_frame_in_chunk", "crop_dir"):
        value = getattr(track, key)
        if value is not None:
            entry[key] = value
    return entry, query_set


def build_people(
    tracks: Iterable[TrackRecord],
    *,
    easy_count: int = 2,
    medium_count: int = 2,
    stop_words: Iterable[str] | None = None,
    hard_query_ids: Iterable[str] = (),
    query_builder: QuerySetBuilder | None = None,
) -> tuple[list[dict[str, Any]], list[QuerySet]]:
    """
    Build all ``people[]`` entries and their query sets, preserving track order.

    Args:
        tracks: Track records to assemble.
        easy_count: Maximum easy template queries per person.
        medium_count: Maximum medium template queries per person.
        stop_words: Optional words to strip from template phrases.
        hard_query_ids: Visual-QA item ids whose answers are hard queries.
        query_builder: Optional LLM-backed ``(attributes, upstream_hard) ->
            QuerySet`` override applied to every track.

    Returns:
        A ``(people, query_sets)`` tuple aligned by index.
    """
    hard_ids = tuple(hard_query_ids)
    people: list[dict[str, Any]] = []
    query_sets: list[QuerySet] = []
    for track in tracks:
        entry, query_set = build_person_entry(
            track,
            easy_count=easy_count,
            medium_count=medium_count,
            stop_words=stop_words,
            hard_query_ids=hard_ids,
            query_builder=query_builder,
        )
        people.append(entry)
        query_sets.append(query_set)
    return people, query_sets


__all__ = [
    "QuerySetBuilder",
    "TrackRecord",
    "build_people",
    "build_person_entry",
    "track_record_from_mapping",
]
