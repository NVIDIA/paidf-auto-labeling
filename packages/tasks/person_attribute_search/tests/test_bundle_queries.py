# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the single-call tiered query-bundle generator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from core.model_clients import ChatRequest
from person_attribute_search.attributes import attributes_from_visual_qa_items
from person_attribute_search.bundle_queries import (
    BundleQueryParams,
    BundleQuerySetBuilder,
    parse_bundle_response,
    query_set_to_bundle,
    render_bundle_prompt,
    resolve_bundle_prompt,
)
from person_attribute_search.schema import PersonAttributes


class _FakeClient:
    """Records the last request and returns a canned response."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.last_request: ChatRequest | None = None
        self.calls = 0

    def generate(self, request: ChatRequest) -> str:
        self.last_request = request
        self.calls += 1
        return self.response


class _SequenceClient:
    """Return canned responses in order, repeating the final response."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    def generate(self, request: ChatRequest) -> str:
        del request
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


def _attributes() -> PersonAttributes:
    return attributes_from_visual_qa_items(
        [
            {"id": "top outer color", "answer": "yellow (mustard)"},
            {"id": "top outer type", "answer": "vest"},
            {"id": "bottom color", "answer": "black"},
            {"id": "bottom type", "answer": "pants"},
            {"id": "natural language caption", "answer": "A worker in a yellow vest."},
        ]
    )


def _bundle_response(easy: list[str], medium: list[str], hard: list[str]) -> str:
    return json.dumps({"queries": {"easy": easy, "medium": medium, "hard": hard}})


def test_parse_bundle_response_nested_shape() -> None:
    query_set = parse_bundle_response(
        _bundle_response(["yellow vest"], ["yellow vest and black pants"], ["a worker ..."])
    )
    assert query_set.easy == [("yellow vest", "")]
    assert query_set.medium == [("yellow vest and black pants", "")]
    assert query_set.hard == ["a worker ..."]


def test_parse_bundle_response_top_level_without_wrapper() -> None:
    query_set = parse_bundle_response('{"easy": ["a"], "medium": ["b"], "hard": ["c"]}')
    assert query_set.easy == [("a", "")]
    assert query_set.medium == [("b", "")]
    assert query_set.hard == ["c"]


def test_parse_bundle_response_object_entries_and_dedup() -> None:
    text = json.dumps(
        {"queries": {"easy": [{"query": "yellow vest"}, "yellow vest", "black pants"]}}
    )
    query_set = parse_bundle_response(text)
    # Object entries are unwrapped and exact duplicates dropped, order preserved.
    assert query_set.easy == [("yellow vest", ""), ("black pants", "")]


def test_parse_bundle_response_markdown_fence() -> None:
    text = "```json\n" + _bundle_response(["a"], [], []) + "\n```"
    assert parse_bundle_response(text).easy == [("a", "")]


def test_parse_bundle_response_applies_caps() -> None:
    text = _bundle_response(["e1", "e2", "e3"], ["m1", "m2"], ["h1", "h2", "h3"])
    query_set = parse_bundle_response(text, easy_cap=2, medium_cap=1, hard_cap=2)
    assert [q for q, _ in query_set.easy] == ["e1", "e2"]
    assert [q for q, _ in query_set.medium] == ["m1"]
    assert query_set.hard == ["h1", "h2"]


def test_parse_bundle_response_strict_count_requires_distinct_queries() -> None:
    text = _bundle_response(["e1", "e1", "e2"], ["m1", "m2", "m3"], ["h1", "h2", "h3"])
    with pytest.raises(ValueError, match="distinct easy queries.*expected at least 3"):
        parse_bundle_response(text, easy_cap=3, medium_cap=3, hard_cap=3, strict_count=3)


def test_parse_bundle_response_strict_count_accepts_overproduction() -> None:
    text = _bundle_response(
        ["e1", "e2", "e3", "e4"],
        ["m1", "m2", "m3"],
        ["h1", "h2", "h3"],
    )
    query_set = parse_bundle_response(text, strict_count=3)

    assert [query for query, _ in query_set.easy] == ["e1", "e2", "e3", "e4"]


def test_parse_bundle_response_malformed_returns_empty() -> None:
    for text in ("no json here", "{not valid json}", ""):
        query_set = parse_bundle_response(text)
        assert not query_set.easy
        assert not query_set.medium
        assert not query_set.hard


def test_resolve_bundle_prompt_prefers_inline_text() -> None:
    assert resolve_bundle_prompt(prompt_text="  hello  ", prompt_file=None) == "hello"


def test_resolve_bundle_prompt_requires_a_source() -> None:
    with pytest.raises(ValueError, match="query_prompt_text or query_prompt_file"):
        resolve_bundle_prompt(prompt_text=None, prompt_file=None)


def test_resolve_bundle_prompt_reads_txt_file(tmp_path: Path) -> None:
    path = tmp_path / "prompt.txt"
    path.write_text("raw prompt body", encoding="utf-8")
    assert resolve_bundle_prompt(prompt_text=None, prompt_file=str(path)) == "raw prompt body"


def test_resolve_bundle_prompt_extracts_question_bank_json(tmp_path: Path) -> None:
    path = tmp_path / "prompt.json"
    path.write_text(
        json.dumps(
            {
                "name": "person_attributes",
                "questions": [
                    {"id": "query_bundle", "question": "Generate queries.", "options": []}
                ],
            }
        ),
        encoding="utf-8",
    )
    assert resolve_bundle_prompt(prompt_text=None, prompt_file=str(path)) == "Generate queries."


def test_resolve_bundle_prompt_reads_epas_synonymous_prompt() -> None:
    prompt = resolve_bundle_prompt(
        prompt_text=None,
        prompt_file="cookbooks/visual_attribute_search/pas_synonymous_query_prompt.json",
    )
    assert "exactly 3 queries" in prompt
    assert "red={maroon, bright red, crimson, burgundy}" in prompt


def test_render_bundle_prompt_substitutes_placeholder() -> None:
    rendered = render_bundle_prompt("desc: {description}", description_json="{DATA}")
    assert rendered == "desc: {DATA}"


def test_render_bundle_prompt_appends_when_no_placeholder() -> None:
    rendered = render_bundle_prompt("Generate queries.", description_json="{DATA}")
    assert rendered.startswith("Generate queries.")
    assert "Structured visual description:" in rendered
    assert "{DATA}" in rendered


def test_builder_single_call_returns_parsed_bundle() -> None:
    client = _FakeClient(_bundle_response(["yellow vest"], ["yellow vest black pants"], ["worker"]))
    builder = BundleQuerySetBuilder(client, BundleQueryParams(prompt="Generate queries."))
    query_set = builder(_attributes(), [])

    assert client.calls == 1
    assert query_set.easy == [("yellow vest", "")]
    assert query_set.medium == [("yellow vest black pants", "")]
    assert query_set.hard == ["worker"]


def test_builder_retries_invalid_bundle_then_succeeds() -> None:
    valid = _bundle_response(
        ["e1", "e2", "e3"],
        ["m1", "m2", "m3"],
        ["h1", "h2", "h3"],
    )
    client = _SequenceClient(["not json", valid])
    builder = BundleQuerySetBuilder(
        client,
        BundleQueryParams(prompt="Generate queries.", strict_count=3),
        response_retries=2,
        response_retry_backoff_s=0.0,
    )

    query_set = builder(_attributes(), [])

    assert client.calls == 2
    assert query_set.hard == ["h1", "h2", "h3"]


def test_builder_strict_bundle_raises_after_response_retries() -> None:
    client = _SequenceClient([_bundle_response(["e1"], ["m1"], ["h1"])])
    builder = BundleQuerySetBuilder(
        client,
        BundleQueryParams(prompt="Generate queries.", strict_count=3),
        response_retries=2,
        response_retry_backoff_s=0.0,
    )

    with pytest.raises(ValueError, match="expected at least 3"):
        builder(_attributes(), [])

    assert client.calls == 3


def test_builder_embeds_description_in_prompt() -> None:
    client = _FakeClient(_bundle_response([], [], ["x"]))
    BundleQuerySetBuilder(client, BundleQueryParams(prompt="Generate queries."))(_attributes(), [])

    assert client.last_request is not None
    # The structured attribute description is injected into the prompt payload.
    assert "vest" in client.last_request.prompt


def test_query_set_to_bundle_flattens_pairs() -> None:
    client = _FakeClient(_bundle_response(["a", "b"], ["c"], ["d"]))
    query_set = BundleQuerySetBuilder(client, BundleQueryParams(prompt="p"))(_attributes(), [])
    assert query_set_to_bundle(query_set) == {"easy": ["a", "b"], "medium": ["c"], "hard": ["d"]}
