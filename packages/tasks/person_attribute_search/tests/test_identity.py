# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for identity aggregation."""

from __future__ import annotations

import pytest
from person_attribute_search.identity import group_by_identity, make_person_key


def test_make_person_key() -> None:
    assert make_person_key("rstp", "00005") == ("rstp", "00005")


def test_group_by_identity_groups_and_dedupes() -> None:
    records = [
        {"person_id": "1", "dataset": "rstp", "image_path": "a.jpg"},
        {"person_id": "1", "dataset": "rstp", "image_path": "b.jpg"},
        {"person_id": "1", "dataset": "rstp", "image_path": "a.jpg"},
        {"person_id": "2", "dataset": "rstp", "image_path": "c.jpg"},
    ]
    groups = group_by_identity(records)
    assert len(groups) == 2
    assert groups[0].person_key == ("rstp", "1")
    assert groups[0].images == ["a.jpg", "b.jpg"]
    assert groups[1].images == ["c.jpg"]


def test_group_by_identity_tuple_keys_avoid_underscore_collisions() -> None:
    records = [
        {"person_id": "a_b", "dataset": "c", "image_path": "first.jpg"},
        {"person_id": "a", "dataset": "b_c", "image_path": "second.jpg"},
    ]
    groups = group_by_identity(records)
    assert [group.person_key for group in groups] == [("c", "a_b"), ("b_c", "a")]


def test_group_by_identity_preserves_first_seen_order() -> None:
    records = [
        {"person_id": "9", "dataset": "d", "image_path": "x.jpg"},
        {"person_id": "3", "dataset": "d", "image_path": "y.jpg"},
    ]
    groups = group_by_identity(records)
    assert [group.person_id for group in groups] == ["9", "3"]


def test_group_by_identity_rejects_missing_fields() -> None:
    with pytest.raises(ValueError, match="missing"):
        group_by_identity([{"dataset": "d", "image_path": "x.jpg"}])
