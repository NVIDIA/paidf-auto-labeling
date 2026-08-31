# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the per-chunk video exporter."""

from __future__ import annotations

from person_attribute_search.export.chunk_pas import (
    build_pas_document,
    build_queries_document,
)


def test_build_pas_document_shape() -> None:
    people = [{"track_id": 2}, {"track_id": 3}]
    doc = build_pas_document(
        chunk_id="chunk_000",
        people=people,
        crop_root="../tracks/crops",
        source_annotation="../vlm_annotations/chunk_000.json",
    )
    assert doc["chunk_id"] == "chunk_000"
    assert doc["source_annotation"] == "../vlm_annotations/chunk_000.json"
    assert doc["pas"]["n_people"] == 2
    assert doc["pas"]["people"] == people
    assert doc["crop_root"] == "../tracks/crops"


def test_build_pas_document_omits_source_when_absent() -> None:
    doc = build_pas_document(chunk_id="chunk_000", people=[], crop_root="")
    assert "source_annotation" not in doc
    assert doc["pas"]["n_people"] == 0


def test_build_queries_document_shape() -> None:
    doc = build_queries_document(
        chunk_id="chunk_000",
        queries=["person in yellow vest", "suspicious running"],
    )
    assert doc["chunk_id"] == "chunk_000"
    assert doc["queries"] == [
        {"query": "person in yellow vest"},
        {"query": "suspicious running"},
    ]
    assert "query_buckets" not in doc


def test_build_queries_document_includes_buckets_when_provided() -> None:
    doc = build_queries_document(
        chunk_id="chunk_000",
        queries=["person in yellow vest"],
        query_buckets={
            "PAS": [["person in yellow vest", ""]],
            "Anomaly": [["person falling", "fall category"]],
            "Caption": [["warehouse aisle", "scene"]],
        },
    )
    assert doc["queries"] == [{"query": "person in yellow vest"}]
    assert doc["query_buckets"] == {
        "PAS": [["person in yellow vest", ""]],
        "Anomaly": [["person falling", "fall category"]],
        "Caption": [["warehouse aisle", "scene"]],
    }
