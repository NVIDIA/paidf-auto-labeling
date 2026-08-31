# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for PAS invalid-response retries."""

from __future__ import annotations

import logging

import pytest
from core.model_clients import ChatRequest, EndpointError
from person_attribute_search.response_retry import generate_with_response_retries


class _SequenceClient:
    """Return or raise canned outcomes in order."""

    def __init__(self, outcomes: list[str | Exception]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    def generate(self, request: ChatRequest) -> str:
        del request
        outcome = self.outcomes[min(self.calls, len(self.outcomes) - 1)]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _request() -> ChatRequest:
    return ChatRequest(prompt="prompt")


def test_retries_invalid_text_with_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = _SequenceClient(["bad", "bad", "valid"])
    sleeps: list[float] = []
    monkeypatch.setattr("person_attribute_search.response_retry.time.sleep", sleeps.append)

    with caplog.at_level(logging.WARNING):
        result = generate_with_response_retries(
            client,
            _request(),
            validate_response=lambda raw: None if raw == "valid" else "bad output",
            stage="test",
            retries=2,
            retry_backoff_s=2.0,
            logger=logging.getLogger("test-response-retry"),
        )

    assert result == "valid"
    assert client.calls == 3
    assert sleeps == [2.0, 4.0]
    assert "Invalid LLM response" in caplog.text


def test_retries_endpoint_response_shape_error() -> None:
    client = _SequenceClient([EndpointError("missing content"), "valid"])

    result = generate_with_response_retries(
        client,
        _request(),
        validate_response=lambda raw: None if raw == "valid" else "bad output",
        stage="test",
        retries=1,
        retry_backoff_s=0.0,
    )

    assert result == "valid"
    assert client.calls == 2


def test_does_not_retry_non_response_exception() -> None:
    client = _SequenceClient([ValueError("invalid request")])

    with pytest.raises(ValueError, match="invalid request"):
        generate_with_response_retries(
            client,
            _request(),
            validate_response=lambda _: None,
            stage="test",
            retries=2,
            retry_backoff_s=0.0,
        )

    assert client.calls == 1
