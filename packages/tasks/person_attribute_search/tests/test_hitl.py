# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the HITL preannotation export."""

from __future__ import annotations

from person_attribute_search.export.hitl import build_preannotation_file, build_preannotations
from person_attribute_search.queries import QuerySet


def _query_set() -> QuerySet:
    return QuerySet(
        easy=[("red t-shirt", "top outer color, top outer type"), ("blue jeans", "bottom")],
        medium=[("red t-shirt and blue jeans", "combo"), ("red top black backpack", "combo")],
        hard=["a detailed caption", "another"],
    )


def test_build_preannotations_default_mix() -> None:
    preannotations = build_preannotations(_query_set(), "A natural caption.")
    types = [item["caption_type"] for item in preannotations]
    assert types == ["Easy", "Easy", "Medium", "Medium", "Hard", "Natural"]


def test_build_preannotations_omits_empty_natural() -> None:
    preannotations = build_preannotations(_query_set(), "   ")
    assert all(item["caption_type"] != "Natural" for item in preannotations)


def test_build_preannotation_file_shape() -> None:
    document = build_preannotation_file("https://s3/x.jpg", _query_set(), "cap")
    assert document["image_url"] == "https://s3/x.jpg"
    assert isinstance(document["preannotations"], list)
