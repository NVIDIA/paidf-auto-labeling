# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for PAS prompt assets and rendering."""

from __future__ import annotations

import json

import pytest
from person_attribute_search.prompts import (
    ATTRIBUTES_PROMPT,
    IMAGE_QUERIES_PROMPT,
    QUERIES_PROMPT,
    load_prompt,
    prompt_path,
    render_image_query_prompt,
    render_query_prompt,
)


def test_all_named_prompts_exist() -> None:
    for name in (ATTRIBUTES_PROMPT, QUERIES_PROMPT, IMAGE_QUERIES_PROMPT):
        assert prompt_path(name).is_file()
        assert load_prompt(name).strip()


def test_attributes_prompt_carries_v3_schema() -> None:
    text = load_prompt(ATTRIBUTES_PROMPT)
    assert "natural language caption" in text
    assert "image quality" in text
    assert "Primary Color" in text


def test_attributes_prompt_schema_block_is_strict_json() -> None:
    text = load_prompt(ATTRIBUTES_PROMPT)
    schema = json.loads(text[text.index("{") :])
    assert schema["accessories"] == [["relationship", "primary_color", "item"]]


def test_render_query_prompt_substitutes_placeholders() -> None:
    rendered = render_query_prompt(attributes='{"gender": "male"}', caption="A man in a vest.")
    assert "{attributes}" not in rendered
    assert "{caption}" not in rendered
    assert '{"gender": "male"}' in rendered
    assert "A man in a vest." in rendered
    # Literal JSON braces in the template survive substitution.
    assert '"medium"' in rendered
    assert '"hard"' in rendered


def test_render_image_query_prompt_substitutes_attributes() -> None:
    rendered = render_image_query_prompt(attributes='{"top outer color": "yellow"}')
    assert "{attributes}" not in rendered
    assert '{"top outer color": "yellow"}' in rendered
    assert "visible_attribute_keys" in rendered


def test_load_prompt_unknown_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt("does_not_exist")


def test_prompt_path_rejects_traversal() -> None:
    with pytest.raises(ValueError):
        prompt_path("../pyproject")
