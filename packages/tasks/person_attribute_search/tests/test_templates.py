# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the deterministic template query engine."""

from __future__ import annotations

from person_attribute_search.schema import Accessory, PersonAttributes
from person_attribute_search.templates import build_easy_queries, build_medium_queries


def _sample_attributes() -> PersonAttributes:
    return PersonAttributes(
        top_outer_color="red",
        top_outer_type="t-shirt",
        bottom_color="blue",
        bottom_type="jeans",
        shoe_color="white",
        shoe_type="sneakers",
        accessories=[Accessory(relationship="carrying", color="black", item="backpack")],
    )


def test_build_easy_queries_single_garment() -> None:
    queries = build_easy_queries(_sample_attributes(), count=2)
    assert queries[0] == ("red t-shirt", "top outer color, top outer type")
    assert len(queries) == 2


def test_build_easy_queries_respects_count() -> None:
    assert build_easy_queries(_sample_attributes(), count=1) == [
        ("red t-shirt", "top outer color, top outer type")
    ]


def test_build_easy_queries_skips_empty_slots() -> None:
    attributes = PersonAttributes(bottom_color="blue", bottom_type="jeans")
    queries = build_easy_queries(attributes, count=3)
    assert queries == [("blue jeans", "bottom color, bottom type")]


def test_build_easy_queries_applies_stop_words() -> None:
    queries = build_easy_queries(_sample_attributes(), count=1, stop_words=["red"])
    assert queries == [("t-shirt", "top outer color, top outer type")]


def test_build_medium_queries_pairs_garments() -> None:
    queries = build_medium_queries(_sample_attributes(), count=1)
    assert queries[0][0] == "red t-shirt and blue jeans"


def test_build_medium_queries_falls_back_to_accessory() -> None:
    attributes = PersonAttributes(
        top_outer_color="red",
        top_outer_type="t-shirt",
        accessories=[Accessory(relationship="carrying", color="black", item="backpack")],
    )
    queries = build_medium_queries(attributes, count=2)
    assert any("backpack" in query for query, _used in queries)


def test_build_medium_queries_skips_empty_accessory_descriptor() -> None:
    attributes = PersonAttributes(
        top_outer_color="red",
        top_outer_type="t-shirt",
        accessories=[Accessory(relationship="holding", color="", item="")],
    )
    queries = build_medium_queries(attributes, count=2)
    assert all(" with" not in query for query, _used in queries)
