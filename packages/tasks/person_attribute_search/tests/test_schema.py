# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the PAS attribute domain model."""

from __future__ import annotations

from person_attribute_search.schema import (
    STRICT_V3_FIELDS,
    VIDEO_EXTRA_FIELDS,
    Accessory,
    PersonAttributes,
    split_color,
    validate_attributes,
)


def test_split_color_with_fine_grade() -> None:
    assert split_color("blue (navy blue)") == ("blue", "navy blue")


def test_split_color_without_fine_grade() -> None:
    assert split_color("red") == ("red", None)


def test_split_color_strips_whitespace() -> None:
    assert split_color("  grey ( silver ) ") == ("grey", "silver")


def test_accessory_phrase() -> None:
    accessory = Accessory(relationship="carrying", color="red", item="handbag")
    assert accessory.phrase() == "carrying red handbag"


def test_validate_attributes_clean() -> None:
    attributes = PersonAttributes(age="adult", gender="female", top_outer_color="red")
    assert validate_attributes(attributes) == []


def test_validate_attributes_flags_unknown_category() -> None:
    attributes = PersonAttributes(age="middle-aged")
    warnings = validate_attributes(attributes)
    assert any("age" in warning for warning in warnings)


def test_validate_attributes_flags_non_primary_color() -> None:
    attributes = PersonAttributes(top_outer_color="turquoise")
    warnings = validate_attributes(attributes)
    assert any("top_outer_color" in warning for warning in warnings)


def test_image_quality_is_strict_v3_and_validated() -> None:
    assert "image_quality" in STRICT_V3_FIELDS
    assert validate_attributes(PersonAttributes(image_quality="high")) == []
    assert any(
        "image_quality" in warning
        for warning in validate_attributes(PersonAttributes(image_quality="pristine"))
    )


def test_strict_v3_and_video_fields_partition_the_schema() -> None:
    model_fields = set(PersonAttributes.model_fields)
    # The two views are disjoint and together cover every attribute field.
    assert STRICT_V3_FIELDS.isdisjoint(VIDEO_EXTRA_FIELDS)
    assert STRICT_V3_FIELDS | VIDEO_EXTRA_FIELDS == model_fields
