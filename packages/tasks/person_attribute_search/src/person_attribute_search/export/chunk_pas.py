# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Per-chunk video PAS export.

Produces the two per-chunk documents the video flow emits:

* ``pas`` document: ``{chunk_id, source_annotation, pas: {n_people, people}, crop_root}``
* ``queries`` document: ``{chunk_id, source_annotation, queries: [{query}, ...]}``

These are PAS-native sidecars, not DAFT contextual/task annotations: the PAS
attribute and query vocabularies are open-ended, whereas DAFT uses a closed-enum
type system, so they are stored as sidecars (mirroring ``anomaly.json``).

This is pure assembly: it only shapes dictionaries already built by ``tracks``
and ``queries``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def build_pas_document(
    *,
    chunk_id: str,
    people: Sequence[dict[str, Any]],
    crop_root: str,
    source_annotation: str | None = None,
) -> dict[str, Any]:
    """
    Build the per-chunk PAS document.

    Args:
        chunk_id: Identifier of the video chunk (e.g. ``"chunk_000"``).
        people: Per-track ``people[]`` entries from ``tracks.build_people``.
        crop_root: Root directory the crops live under, relative to the document.
        source_annotation: Optional pointer to the upstream VLM annotation file.

    Returns:
        A dict matching the per-chunk ``pas.json`` shape.
    """
    document: dict[str, Any] = {"chunk_id": chunk_id}
    if source_annotation is not None:
        document["source_annotation"] = source_annotation
    document["pas"] = {"n_people": len(people), "people": list(people)}
    document["crop_root"] = crop_root
    return document


def build_queries_document(
    *,
    chunk_id: str,
    queries: Iterable[str],
    source_annotation: str | None = None,
    query_buckets: Mapping[str, Sequence[Sequence[str]]] | None = None,
) -> dict[str, Any]:
    """
    Build the per-chunk queries document.

    Args:
        chunk_id: Identifier of the video chunk.
        queries: Flat, de-duplicated query strings from ``queries.flatten_queries``.
        source_annotation: Optional pointer to the upstream VLM annotation file.
        query_buckets: Optional legacy 3-bucket queries
            (``{PAS, Anomaly, Caption}`` of ``[query, evidence]`` pairs). When
            provided it is written under ``query_buckets`` so the annotation
            adapter can pass the buckets through; the flat ``queries`` list is
            always written for backward compatibility.

    Returns:
        A dict matching the per-chunk ``queries.json`` shape.
    """
    document: dict[str, Any] = {"chunk_id": chunk_id}
    if source_annotation is not None:
        document["source_annotation"] = source_annotation
    document["queries"] = [{"query": query} for query in queries]
    if query_buckets is not None:
        document["query_buckets"] = {
            bucket: [list(pair) for pair in pairs] for bucket, pairs in query_buckets.items()
        }
    return document


__all__ = ["build_pas_document", "build_queries_document"]
