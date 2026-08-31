# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared LLM helper utilities."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import pytest
from core.llm import (
    build_structured_request_options,
    call_chat_raw,
    call_required_json_object_with_fallback,
    extract_json_object_from_llm_text,
    get_api_key,
    get_llm_api_key,
    get_vlm_api_key,
    require_items_list,
    reset_openai_client_cache,
)


class HelperError(RuntimeError):
    pass


def test_structured_options_include_nim_guided_json() -> None:
    schema = {"type": "object", "properties": {"items": {"type": "array"}}}

    mode, extra_body, response_format = build_structured_request_options(
        structured_output="nim",
        base_url="http://nim/v1",
        model="model",
        guided_json_schema=schema,
    )

    assert mode == "nim"
    assert extra_body == {"nvext": {"guided_json": schema}, "guided_json": schema}
    assert response_format is None


def test_required_json_object_helper_preserves_call_options() -> None:
    captured: dict[str, Any] = {}

    def caller(**kwargs: Any) -> tuple[dict[str, object], str]:
        captured.update(kwargs)
        return {"items": [{"id": "1"}]}, '{"items": [{"id": "1"}]}'

    result = call_required_json_object_with_fallback(
        base_url="http://llm/v1",
        model="model",
        messages=[{"role": "user", "content": "hi"}],
        timeout=30,
        max_tokens=128,
        temperature=0.1,
        top_p=0.9,
        logger=None,
        retries=2,
        retry_backoff_s=0.5,
        structured_output="openai",
        seed=7,
        guided_json_schema={"type": "object"},
        retry_stage="stage",
        api_key="key",
        error_type=HelperError,
        caller=caller,
    )

    assert result == {"items": [{"id": "1"}]}
    assert captured["base_url"] == "http://llm/v1"
    assert captured["model"] == "model"
    assert captured["retry_stage"] == "stage"
    assert captured["guided_json_schema"] == {"type": "object"}


def test_extract_json_object_repairs_common_model_output() -> None:
    assert extract_json_object_from_llm_text('prefix ```json\n{"items": [1,],}\n```') == {
        "items": [1]
    }


def test_require_items_list_validates_shape() -> None:
    assert require_items_list(
        {"items": [{"id": "1"}]},
        error_type=HelperError,
        missing_message="missing {type_name}",
        empty_message="empty",
    ) == [{"id": "1"}]

    with pytest.raises(HelperError, match="missing str"):
        require_items_list(
            {"items": "bad"},
            error_type=HelperError,
            missing_message="missing {type_name}",
            empty_message="empty",
        )


def test_get_api_key_reads_nvidia_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-key")

    assert get_api_key() == "nvapi-key"
    assert os.environ["NVIDIA_API_KEY"] == "nvapi-key"


def test_get_api_key_returns_empty_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-key")
    monkeypatch.setenv("VLM_API_KEY", "vlm-key")
    monkeypatch.setenv("LLM_API_KEY", "llm-key")

    assert get_api_key() == "EMPTY"


def test_get_vlm_and_llm_api_keys_use_nvidia_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-key")
    monkeypatch.setenv("VLM_API_KEY", "vlm-key")
    monkeypatch.setenv("LLM_API_KEY", "llm-key")

    assert get_vlm_api_key() == "nvapi-key"
    assert get_llm_api_key() == "nvapi-key"


def test_call_chat_raw_expands_tokens_on_first_length_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_openai_client_cache()
    seen_max_tokens: list[int] = []

    def fake_create_client(**_kwargs: Any) -> object:
        return object()

    def fake_completion(_client: object, **kwargs: Any) -> SimpleNamespace:
        seen_max_tokens.append(kwargs["max_tokens"])
        if len(seen_max_tokens) == 1:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(finish_reason="length", message=SimpleNamespace(content=""))
                ]
            )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=[{"text": "ok"}]),
                )
            ]
        )

    monkeypatch.setattr("core.llm.clients.create_openai_client", fake_create_client)
    monkeypatch.setattr("core.llm.clients.create_openai_chat_completion", fake_completion)

    result = call_chat_raw(
        base_url="http://llm",
        model="model",
        messages=[{"role": "user", "content": "hi"}],
        timeout=30,
        max_tokens=128,
        temperature=0.1,
        top_p=0.9,
        retries=1,
        retry_backoff_s=0.0,
        api_key="secret",
    )

    assert result == "ok"
    # Budget doubles on the *first* length truncation instead of wasting a
    # second same-size attempt first.
    assert seen_max_tokens == [128, 256]


def test_call_chat_raw_uses_shared_openai_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_openai_client_cache()
    created: dict[str, Any] = {}
    captured: dict[str, Any] = {}
    fake_client = object()

    def fake_create_client(**kwargs: Any) -> object:
        created.update(kwargs)
        return fake_client

    def fake_completion(client: object, **kwargs: Any) -> SimpleNamespace:
        captured["client"] = client
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=[{"text": "hello"}, {"text": "world"}]),
                )
            ]
        )

    monkeypatch.setattr("core.llm.clients.create_openai_client", fake_create_client)
    monkeypatch.setattr("core.llm.clients.create_openai_chat_completion", fake_completion)

    response_meta: dict[str, Any] = {}
    result = call_chat_raw(
        base_url="http://llm",
        model="model",
        messages=[{"role": "user", "content": "hi"}],
        timeout=30,
        max_tokens=128,
        temperature=0.1,
        top_p=0.9,
        extra_body={"guided_json": {"type": "object"}},
        response_format={"type": "json_object"},
        seed=7,
        retries=0,
        response_meta=response_meta,
        api_key="secret",
    )

    assert result == "hello\nworld"
    assert created == {
        "api_key": "secret",
        "endpoint_url": "http://llm/v1",
        "timeout_s": 30.0,
    }
    assert captured == {
        "client": fake_client,
        "model": "model",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 128,
        "temperature": 0.1,
        "top_p": 0.9,
        "extra_body": {"guided_json": {"type": "object"}},
        "response_format": {"type": "json_object"},
        "seed": 7,
    }
    assert response_meta["finish_reason"] == "stop"
