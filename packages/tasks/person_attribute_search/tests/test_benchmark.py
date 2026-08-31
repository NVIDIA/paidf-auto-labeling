# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for legacy PAS benchmark JSON builders."""

from __future__ import annotations

from person_attribute_search.export import benchmark
from person_attribute_search.queries import assemble_query_set
from person_attribute_search.schema import Accessory, PersonAttributes


def _attributes() -> PersonAttributes:
    return PersonAttributes(
        gender="female",
        top_outer_color="red",
        top_outer_color_fine="crimson",
        top_outer_type="t-shirt",
        accessories=[Accessory(relationship="carrying", color="black", item="backpack")],
        natural_caption="A woman in red.",
    )


def test_attributes_to_legacy_dict_uses_spaced_keys() -> None:
    legacy = benchmark.attributes_to_legacy_dict(_attributes())
    assert legacy["top outer color"] == "red"
    assert legacy["top outer color (fine)"] == "crimson"
    assert legacy["accessories"] == [["carrying", "black", "backpack"]]


def test_build_attribute_entry() -> None:
    entry = benchmark.build_attribute_entry(
        person_key="1_rstp",
        person_id="1",
        dataset="rstp",
        images={"all": ["1"]},
        attributes=_attributes(),
    )
    assert entry["person_key"] == "1_rstp"
    assert entry["natural_caption"] == "A woman in red."


def test_build_query_entry_includes_queries() -> None:
    entry = benchmark.build_query_entry(
        person_key="1_rstp",
        person_id="1",
        dataset="rstp",
        attributes=_attributes(),
        query_set=assemble_query_set(_attributes(), hard_queries=["h"]),
    )
    assert "queries" in entry
    queries = entry["queries"]
    assert isinstance(queries, dict)
    assert queries["hard"] == ["h"]


def test_build_query_entry_can_emit_flat_query_tiers() -> None:
    query_set = assemble_query_set(_attributes(), hard_queries=["h"])
    entry = benchmark.build_query_entry(
        person_key="1_rstp",
        person_id="1",
        dataset="rstp",
        attributes=_attributes(),
        query_set=query_set,
        flat_query_tiers=True,
    )
    queries = entry["queries"]
    assert isinstance(queries, dict)
    assert queries["easy"] == [query for query, _used in query_set.easy]
    assert queries["medium"] == [query for query, _used in query_set.medium]
    assert queries["hard"] == ["h"]


def test_build_result_json_envelope() -> None:
    metadata = benchmark.build_metadata(source_file="src.json", model="m")
    result = benchmark.build_result_json(metadata=metadata, entries=[{"a": 1}])
    assert result["metadata"] == metadata
    assert result["entries"] == [{"a": 1}]
