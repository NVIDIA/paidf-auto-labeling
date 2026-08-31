# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for difficulty-tiered query assembly."""

from __future__ import annotations

from person_attribute_search.queries import (
    assemble_query_set,
    collect_hard_queries,
    flatten_queries,
    normalize_anomaly_label,
)
from person_attribute_search.schema import Accessory, PersonAttributes


def _attributes() -> PersonAttributes:
    return PersonAttributes(
        top_outer_color="red",
        top_outer_color_fine="crimson",
        top_outer_type="t-shirt",
        bottom_color="blue",
        bottom_type="jeans",
        natural_caption="A woman in a crimson t-shirt and blue jeans.",
    )


def test_assemble_uses_supplied_hard_queries() -> None:
    query_set = assemble_query_set(_attributes(), hard_queries=["custom hard query"])
    assert query_set.hard == ["custom hard query"]
    assert query_set.easy
    assert query_set.medium


def test_collect_hard_queries_filters_ids_and_empty_answers() -> None:
    items = [
        {"id": "gender", "answer": "female"},
        {"id": "hard", "answer": "custom hard query"},
        {"id": "hard_1", "answer": "  "},
        {"id": "hard_2", "answer": "second hard query"},
    ]
    assert collect_hard_queries(items, ("hard", "hard_1", "hard_2")) == [
        "custom hard query",
        "second hard query",
    ]


def test_assemble_falls_back_to_caption_hard_query() -> None:
    query_set = assemble_query_set(_attributes(), hard_queries=None)
    assert query_set.hard
    assert query_set.hard[0].startswith("A woman in a crimson")


def test_assemble_hard_never_empty_for_sparse_attributes() -> None:
    # No caption and no garment/accessory descriptors: hard must still be set.
    query_set = assemble_query_set(PersonAttributes(), hard_queries=None)
    assert query_set.hard == ["person"]


def test_assemble_ignores_empty_accessory_descriptors() -> None:
    query_set = assemble_query_set(
        PersonAttributes(accessories=[Accessory(relationship="holding", color="", item="")]),
        hard_queries=None,
    )
    assert query_set.hard == ["person"]


def test_assemble_fallback_truncates_to_15_words() -> None:
    long_caption = " ".join(f"word{i}" for i in range(40))
    attributes = PersonAttributes(natural_caption=long_caption)
    query_set = assemble_query_set(attributes, hard_queries=[])
    assert len(query_set.hard[0].split()) == 15


def test_to_legacy_dict_shape() -> None:
    query_set = assemble_query_set(_attributes(), hard_queries=["h"])
    legacy = query_set.to_legacy_dict()
    assert set(legacy) == {"easy", "medium", "hard"}
    easy = legacy["easy"]
    assert isinstance(easy, list)
    assert easy[0] == [query_set.easy[0][0], query_set.easy[0][1]]
    assert legacy["hard"] == ["h"]


def test_to_flat_dict_omits_attributes_used() -> None:
    query_set = assemble_query_set(_attributes(), hard_queries=["h"])
    flat = query_set.to_flat_dict()
    assert flat["easy"] == [query for query, _used in query_set.easy]
    assert flat["medium"] == [query for query, _used in query_set.medium]
    assert flat["hard"] == ["h"]


def test_normalize_anomaly_label() -> None:
    assert normalize_anomaly_label("warehouse_safety_violation") == ("warehouse safety violation")


def test_flatten_queries_dedupes_and_orders() -> None:
    first = assemble_query_set(_attributes(), hard_queries=["shared hard"])
    second = assemble_query_set(_attributes(), hard_queries=["shared hard"])
    flat = flatten_queries(
        [first, second],
        anomaly_labels=["suspicious_running", "suspicious_running"],
        extra_queries=["a scene caption", "shared hard"],
    )
    # No duplicates.
    assert len(flat) == len(set(flat))
    # Easy/medium texts come first, anomaly phrase included, extra appended.
    assert "shared hard" in flat
    assert "suspicious running" in flat
    assert "a scene caption" in flat
    # Anomaly phrases precede the trailing extra query.
    assert flat.index("suspicious running") < flat.index("a scene caption")


def test_flatten_queries_handles_empty_sources() -> None:
    assert flatten_queries([]) == []
