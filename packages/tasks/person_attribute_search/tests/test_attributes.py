# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for parsing upstream answers into ``PersonAttributes``."""

from __future__ import annotations

from person_attribute_search.attributes import (
    attributes_from_answers,
    attributes_from_visual_qa_items,
)


def test_attributes_from_answers_splits_color() -> None:
    attributes = attributes_from_answers({"top outer color": "blue (navy blue)"})
    assert attributes.top_outer_color == "blue"
    assert attributes.top_outer_color_fine == "navy blue"


def test_attributes_from_answers_normalizes_keys() -> None:
    attributes = attributes_from_answers(
        {"top-outer-type": "t-shirt", "natural language caption": "A person."}
    )
    assert attributes.top_outer_type == "t-shirt"
    assert attributes.natural_caption == "A person."


def test_attributes_from_answers_parses_accessory_tuples() -> None:
    attributes = attributes_from_answers({"accessories": [["carrying", "black", "backpack"]]})
    assert len(attributes.accessories) == 1
    assert attributes.accessories[0].item == "backpack"


def test_attributes_from_answers_ignores_blank_values() -> None:
    attributes = attributes_from_answers({"gender": "  ", "age": None})
    assert attributes.gender is None
    assert attributes.age is None


def test_attributes_from_visual_qa_items() -> None:
    items = [
        {"id": "gender", "answer": "female"},
        {"id": "bottom color", "answer": "blue (navy blue)"},
        {"id": "", "answer": "ignored"},
    ]
    attributes = attributes_from_visual_qa_items(items)
    assert attributes.gender == "female"
    assert attributes.bottom_color == "blue"
    assert attributes.bottom_color_fine == "navy blue"


def test_attributes_parses_video_superset_fields() -> None:
    attributes = attributes_from_answers(
        {
            "headwear": "yes",
            "headwear type": "helmet",
            "headwear color": "yellow",
            "mask": "no",
            "glasses": "yes",
            "carrying": "none",
            "motion or action": "walking",
            "potential anomaly": "yes",
            "anomaly type": "warehouse_safety_violation",
            "top inner length": "long sleeve",
            "bottom length": "long",
        }
    )
    assert attributes.headwear == "yes"
    assert attributes.headwear_type == "helmet"
    assert attributes.headwear_color == "yellow"
    assert attributes.mask == "no"
    assert attributes.glasses == "yes"
    assert attributes.carrying == "none"
    assert attributes.motion_or_action == "walking"
    assert attributes.potential_anomaly == "yes"
    assert attributes.anomaly_type == "warehouse_safety_violation"
    assert attributes.top_inner_length == "long sleeve"
    assert attributes.bottom_length == "long"
