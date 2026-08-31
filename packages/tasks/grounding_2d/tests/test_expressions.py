# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for groundable-expression filtering."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from grounding_2d.expressions import (
    ExpressionFilterPolicy,
    filter_groundable_expressions,
    fits_sam3_text_encoder,
    is_groundable_expression,
)


def test_keeps_concrete_object_phrases_by_default() -> None:
    assert is_groundable_expression({"text": "cars", "noun_chunk": "cars"})
    assert is_groundable_expression({"text": "white truck", "noun_chunk": "truck"})
    assert is_groundable_expression(
        {"text": "black SUV in intersection", "noun_chunk": "SUV", "groundable": True}
    )


def test_vlm_groundable_false_is_skipped() -> None:
    assert not is_groundable_expression(
        {
            "text": "busy traffic scene",
            "noun_chunk": "scene",
            "groundable": False,
        }
    )


def test_filter_splits_on_vlm_flag() -> None:
    expressions = [
        {
            "expression_id": "expr_00000",
            "text": "cars",
            "noun_chunk": "cars",
            "groundable": True,
        },
        {
            "expression_id": "expr_00001",
            "text": "the scene",
            "noun_chunk": "scene",
            "groundable": False,
        },
        {
            "expression_id": "expr_00002",
            "text": "trucks",
            "noun_chunk": "trucks",
            "groundable": True,
        },
    ]
    groundable, skipped = filter_groundable_expressions(expressions)
    assert [e["text"] for e in groundable] == ["cars", "trucks"]
    assert [e["text"] for e in skipped] == ["the scene"]


def test_fits_sam3_text_encoder_accepts_short_phrases() -> None:
    assert fits_sam3_text_encoder("dark gray sedan")
    assert fits_sam3_text_encoder("motorcycle on its side")


def test_fits_sam3_text_encoder_rejects_overlong_and_cjk_heavy() -> None:
    assert not fits_sam3_text_encoder("green for the direction of the main flow of traffic today")
    assert not fits_sam3_text_encoder('business named "大水電材行" (Dàshuǐ Diàn Cái Háng)')
    assert not fits_sam3_text_encoder("a" * 80)


def test_optional_json_policy_deny_lists(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "deny_phrases": ["the scene"],
                "deny_nouns": ["background"],
            }
        ),
        encoding="utf-8",
    )
    policy = ExpressionFilterPolicy.from_json_file(policy_path)
    assert not is_groundable_expression({"text": "the scene", "noun_chunk": "scene"}, policy=policy)
    assert not is_groundable_expression(
        {"text": "trees in background", "noun_chunk": "trees"}, policy=policy
    )
    assert is_groundable_expression(
        {"text": "red box on shelf", "noun_chunk": "box"}, policy=policy
    )


def test_empty_policy_json_makes_no_keyword_assumptions(tmp_path: Path) -> None:
    policy_path = tmp_path / "empty.json"
    policy_path.write_text(json.dumps({"deny_phrases": [], "deny_nouns": []}), encoding="utf-8")
    policy = ExpressionFilterPolicy.from_json_file(policy_path)
    # Traffic-ish and retail-ish phrases alike are allowed without VLM false flag.
    assert is_groundable_expression(
        {"text": "collision debris pile", "noun_chunk": "pile"}, policy=policy
    )
    assert is_groundable_expression(
        {"text": "cardboard box on pallet", "noun_chunk": "box"}, policy=policy
    )


def test_invalid_policy_shape_raises(tmp_path: Path) -> None:
    policy_path = tmp_path / "bad.json"
    policy_path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        ExpressionFilterPolicy.from_json_file(policy_path)
