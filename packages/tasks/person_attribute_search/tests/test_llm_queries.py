# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the LLM tiered-query generation op (generate_queries.py parity)."""

from __future__ import annotations

from core.model_clients import ChatRequest
from person_attribute_search.attributes import attributes_from_visual_qa_items
from person_attribute_search.llm_queries import (
    LlmQueryParams,
    LlmQuerySetBuilder,
    parse_query_response,
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
            {"id": "gender", "answer": "female"},
            {"id": "top outer color", "answer": "red (crimson)"},
            {"id": "top outer type", "answer": "t-shirt"},
            {"id": "bottom color", "answer": "blue (navy)"},
            {"id": "bottom type", "answer": "jeans"},
            {"id": "natural language caption", "answer": "A woman in a red t-shirt."},
        ]
    )


def test_parse_query_response_medium_and_hard() -> None:
    text = (
        '{"medium": [["red t-shirt and blue jeans", "top outer color, bottom color"]],'
        ' "hard": ["A woman with a crimson t-shirt and navy jeans", "Crimson tee, navy denim"]}'
    )
    medium, hard = parse_query_response(text)
    assert medium == [("red t-shirt and blue jeans", "top outer color, bottom color")]
    assert hard == [
        "A woman with a crimson t-shirt and navy jeans",
        "Crimson tee, navy denim",
    ]


def test_parse_query_response_markdown_fence() -> None:
    text = '```json\n{"hard": ["one", "two"]}\n```'
    medium, hard = parse_query_response(text)
    assert medium == []
    assert hard == ["one", "two"]


def test_parse_query_response_handles_bare_string_medium() -> None:
    medium, hard = parse_query_response('{"medium": ["just a string"], "hard": []}')
    assert medium == [("just a string", "")]
    assert hard == []


def test_parse_query_response_handles_object_medium_items() -> None:
    medium, hard = parse_query_response(
        '{"medium": [{"query": "red shirt and blue jeans", '
        '"attributes_used": ["top outer color", "bottom type"]}], "hard": []}'
    )
    assert medium == [("red shirt and blue jeans", "top outer color, bottom type")]
    assert hard == []


def test_parse_query_response_malformed_returns_empty() -> None:
    assert parse_query_response("no json here") == ([], [])
    assert parse_query_response("{not valid json}") == ([], [])
    assert parse_query_response("") == ([], [])


def test_builder_uses_llm_for_medium_and_hard() -> None:
    client = _FakeClient(
        '{"medium": [["red t-shirt and blue jeans", "top outer color"]],'
        ' "hard": ["crimson tee navy denim", "woman in red shirt"]}'
    )
    builder = LlmQuerySetBuilder(client, LlmQueryParams())
    query_set = builder(_attributes(), [])

    assert client.calls == 1
    assert query_set.medium == [("red t-shirt and blue jeans", "top outer color")]
    assert query_set.hard == ["crimson tee navy denim", "woman in red shirt"]
    # Easy queries always come from templates (never empty for clothed attributes).
    assert query_set.easy


def test_builder_caps_medium_to_medium_count() -> None:
    client = _FakeClient('{"medium": [["q1", "a"], ["q2", "b"], ["q3", "c"]], "hard": ["h1"]}')
    builder = LlmQuerySetBuilder(client, LlmQueryParams(medium_count=2))
    query_set = builder(_attributes(), [])

    # The LLM returned three medium pairs; the configured cap keeps the first two.
    assert query_set.medium == [("q1", "a"), ("q2", "b")]


def test_builder_hard_only_uses_template_medium() -> None:
    client = _FakeClient('{"hard": ["crimson tee navy denim", "woman red shirt"]}')
    params = LlmQueryParams(use_template_for_medium=True)
    builder = LlmQuerySetBuilder(client, params)
    query_set = builder(_attributes(), [])

    assert client.calls == 1
    # Hard-only prompt asks for no medium; medium comes from the template engine.
    assert query_set.hard == ["crimson tee navy denim", "woman red shirt"]
    assert query_set.medium  # template-generated, not from the LLM


def test_builder_falls_back_to_upstream_hard_when_llm_empty() -> None:
    client = _FakeClient("garbage no json")
    builder = LlmQuerySetBuilder(client, LlmQueryParams())
    query_set = builder(_attributes(), ["upstream hard query"])

    assert query_set.hard == ["upstream hard query"]


def test_builder_retries_invalid_tiered_response_then_succeeds() -> None:
    client = _SequenceClient(
        [
            '{"medium": [], "hard": []}',
            '{"medium": [["medium query", "shirt"]], "hard": ["hard query"]}',
        ]
    )
    builder = LlmQuerySetBuilder(
        client,
        LlmQueryParams(),
        response_retries=2,
        response_retry_backoff_s=0.0,
    )

    query_set = builder(_attributes(), [])

    assert client.calls == 2
    assert query_set.medium == [("medium query", "shirt")]
    assert query_set.hard == ["hard query"]


def test_builder_falls_back_after_response_retries_exhausted() -> None:
    client = _SequenceClient(["not json"])
    builder = LlmQuerySetBuilder(
        client,
        LlmQueryParams(),
        response_retries=2,
        response_retry_backoff_s=0.0,
    )

    query_set = builder(_attributes(), ["upstream hard query"])

    assert client.calls == 3
    assert query_set.hard == ["upstream hard query"]


def test_builder_passes_attributes_and_caption_in_prompt() -> None:
    client = _FakeClient('{"medium": [], "hard": ["x"]}')
    LlmQuerySetBuilder(client, LlmQueryParams())(_attributes(), [])

    assert client.last_request is not None
    prompt = client.last_request.prompt
    assert "A woman in a red t-shirt." in prompt
    assert "crimson" in prompt  # fine-grained color carried into the attributes block
