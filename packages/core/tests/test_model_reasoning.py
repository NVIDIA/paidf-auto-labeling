# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from core.llm.clients import call_chat_raw
from core.model_clients.gemini import GeminiClient
from core.model_clients.openai_compatible import OpenAICompatibleClient
from core.model_clients.reasoning import (
    extra_body_for_parser,
    merge_extra_body,
    strip_think_blocks,
)
from core.model_clients.types import ChatRequest


def test_extra_body_for_parser_disables_thinking_in_instruct_mode() -> None:
    assert extra_body_for_parser("instruct") == {"chat_template_kwargs": {"enable_thinking": False}}


def test_extra_body_for_parser_leaves_reasoning_mode_untouched() -> None:
    assert extra_body_for_parser("reasoning") is None


def test_merge_extra_body_handles_missing_inputs() -> None:
    assert merge_extra_body(None, None) is None
    assert merge_extra_body(None, {"a": 1}) == {"a": 1}
    assert merge_extra_body({"a": 1}, None) == {"a": 1}


def test_merge_extra_body_deep_merges_and_prefers_override() -> None:
    base = {"chat_template_kwargs": {"enable_thinking": False}, "keep": 1}
    override = {"chat_template_kwargs": {"foo": "bar"}, "keep": 2}

    merged = merge_extra_body(base, override)

    assert merged == {
        "chat_template_kwargs": {"enable_thinking": False, "foo": "bar"},
        "keep": 2,
    }
    # Inputs are not mutated.
    assert base == {"chat_template_kwargs": {"enable_thinking": False}, "keep": 1}


def test_merge_extra_body_override_scalar_replaces_dict() -> None:
    assert merge_extra_body({"x": {"a": 1}}, {"x": 5}) == {"x": 5}


def test_strip_think_blocks_removes_reasoning_traces() -> None:
    assert strip_think_blocks("<think>secret</think>answer") == "answer"
    assert strip_think_blocks("a<think>one</think>b<think>two</think>c") == "abc"
    assert strip_think_blocks("<think>line1\nline2</think>\nresult") == "result"
    assert strip_think_blocks("<THINK>x</THINK>kept") == "kept"


def test_strip_think_blocks_passthrough_when_no_block() -> None:
    assert strip_think_blocks("plain answer") == "plain answer"
    assert strip_think_blocks("") == ""


def _fake_openai_client(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeOpenAI:
        def __init__(self, **_: Any) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **__: None))

    monkeypatch.setattr("core.model_clients.openai_compatible.OpenAI", _FakeOpenAI)


def test_openai_client_instruct_injects_extra_body_and_strips_think(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_completion(_: Any, **kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content="<think>reason</think>done"),
                )
            ],
            usage=None,
        )

    _fake_openai_client(monkeypatch)
    monkeypatch.setattr(
        "core.model_clients.openai_compatible.create_openai_chat_completion",
        fake_completion,
    )

    client = OpenAICompatibleClient(endpoint_url="http://m/v1", model="m", retries=0)

    assert client.generate(ChatRequest(prompt="hi")) == "done"
    assert captured["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_openai_client_reasoning_mode_sends_no_extra_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_completion(_: Any, **kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="ok"))],
            usage=None,
        )

    _fake_openai_client(monkeypatch)
    monkeypatch.setattr(
        "core.model_clients.openai_compatible.create_openai_chat_completion",
        fake_completion,
    )

    client = OpenAICompatibleClient(
        endpoint_url="http://m/v1", model="m", retries=0, parser="reasoning"
    )

    assert client.generate(ChatRequest(prompt="hi")) == "ok"
    assert captured["extra_body"] is None


def test_call_chat_raw_strips_think_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_DRY_RUN", raising=False)
    monkeypatch.setattr("core.llm.clients._get_openai_client", lambda **_: object())

    def fake_completion(_: Any, **__: Any) -> SimpleNamespace:
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content='<think>plan</think>{"ok": true}'),
                )
            ],
            usage=None,
        )

    monkeypatch.setattr("core.llm.clients.create_openai_chat_completion", fake_completion)

    text = call_chat_raw(
        base_url="http://m/v1",
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        timeout=10,
        max_tokens=16,
        temperature=0.0,
        top_p=1.0,
        retries=0,
    )

    assert text == '{"ok": true}'


def test_call_chat_raw_sanitizes_reasoning_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_DRY_RUN", raising=False)
    monkeypatch.setattr("core.llm.clients._get_openai_client", lambda **_: object())

    def fake_completion(_: Any, **__: Any) -> SimpleNamespace:
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content="", reasoning='<think>plan</think>{"ok": true}'
                    ),
                )
            ],
            usage=None,
        )

    monkeypatch.setattr("core.llm.clients.create_openai_chat_completion", fake_completion)

    text = call_chat_raw(
        base_url="http://m/v1",
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        timeout=10,
        max_tokens=16,
        temperature=0.0,
        top_p=1.0,
        retries=0,
    )

    assert text == '{"ok": true}'


def test_gemini_client_rejects_non_default_parser() -> None:
    with pytest.raises(NotImplementedError, match="does not support parser"):
        GeminiClient(endpoint_url="http://g", model="g", parser="reasoning")
