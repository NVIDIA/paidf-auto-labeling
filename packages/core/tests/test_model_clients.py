# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import urllib.request
from types import SimpleNamespace, TracebackType
from typing import Any, Literal

import pytest
from core.model_clients import (
    GEMINI_API_KEY_ENV,
    NVIDIA_API_KEY_ENV,
    ChatRequest,
    MediaPayload,
    create_endpoint_client,
)
from core.model_clients.gemini import GeminiClient, gemini_generate_url
from core.model_clients.http import post_json
from core.model_clients.openai_compatible import (
    OpenAICompatibleClient,
    content_text,
    create_openai_chat_completion,
    is_retryable_openai_error,
    openai_base_url,
    resolve_api_key,
    validate_openai_endpoint_url,
)


class _Response:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> _Response:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        return False

    def read(self) -> bytes:
        return self.body


def test_openai_messages_preserve_media_order() -> None:
    messages = OpenAICompatibleClient._messages(
        ChatRequest(
            prompt="Describe.",
            media=(
                MediaPayload(mime_type="image/jpeg", data_base64="image"),
                MediaPayload(mime_type="video/mp4", data_base64="video"),
            ),
        )
    )

    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,image"}},
                {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,video"}},
                {"type": "text", "text": "Describe."},
            ],
        }
    ]


def test_openai_client_records_call_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeCompletions:
        def create(self, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content=[{"text": "done"}]),
                    )
                ],
                usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 3}),
            )

    class _FakeOpenAI:
        instances: list[_FakeOpenAI] = []

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.chat = SimpleNamespace(completions=_FakeCompletions())
            self.instances.append(self)

    monkeypatch.setenv(NVIDIA_API_KEY_ENV, "secret")
    monkeypatch.setattr("core.model_clients.openai_compatible.OpenAI", _FakeOpenAI)

    client = OpenAICompatibleClient(
        endpoint_url="http://model.test/v1/chat/completions",
        model="test-model",
        timeout_s=11.0,
        retries=0,
    )

    assert client.generate(ChatRequest(prompt="Run.")) == "done"
    assert _FakeOpenAI.instances[0].kwargs == {
        "api_key": "secret",
        "base_url": "http://model.test/v1",
        "timeout": 11.0,
        "max_retries": 0,
    }
    assert client.last_call_metadata == {
        "provider": "openai-compatible",
        "model": "test-model",
        "endpoint_url": "http://model.test/v1/chat/completions",
        "finish_reason": "stop",
        "usage": {"total_tokens": 3},
    }


def test_openai_client_requires_explicit_endpoint_url() -> None:
    with pytest.raises(ValueError, match="require an explicit endpoint_url"):
        OpenAICompatibleClient(
            endpoint_url=None,
            model="test-model",
            retries=0,
        )


def test_openai_base_url_accepts_base_or_chat_completion_url() -> None:
    assert openai_base_url("http://model.test/v1") == "http://model.test/v1"
    assert openai_base_url("http://model.test/v1/chat/completions") == "http://model.test/v1"


def test_openai_endpoint_url_rejects_non_http_scheme() -> None:
    with pytest.raises(ValueError, match="Unsupported endpoint URL scheme: file"):
        validate_openai_endpoint_url("file:///etc/passwd")


def test_openai_endpoint_url_rejects_missing_host() -> None:
    with pytest.raises(ValueError, match="requires a host"):
        validate_openai_endpoint_url("http:///v1")


def test_openai_endpoint_url_rejects_relative_path() -> None:
    with pytest.raises(ValueError, match="Unsupported endpoint URL scheme: <missing>"):
        validate_openai_endpoint_url("/v1")


def test_openai_chat_completion_helper_preserves_text_options() -> None:
    captured: dict[str, Any] = {}

    class _FakeCompletions:
        def create(self, **kwargs: Any) -> SimpleNamespace:
            captured.update(kwargs)
            return SimpleNamespace(choices=[])

    client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))

    create_openai_chat_completion(
        client,
        model="model",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=12,
        temperature=0.1,
        top_p=0.8,
        extra_body={},
        response_format={},
        seed=7,
    )

    assert captured == {
        "model": "model",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 12,
        "temperature": 0.1,
        "top_p": 0.8,
        "extra_body": {},
        "response_format": {},
        "seed": 7,
    }


def test_openai_retry_classifier_keeps_transient_message_fallback() -> None:
    assert is_retryable_openai_error(RuntimeError("server disconnected before reply"))


def test_openai_client_logs_transient_retry(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class APIConnectionError(Exception):
        pass

    outcomes: list[Exception | SimpleNamespace] = [
        APIConnectionError("connection reset"),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="done"),
                    finish_reason="stop",
                )
            ],
            usage=None,
        ),
    ]
    monkeypatch.setattr(
        "core.model_clients.openai_compatible.create_openai_client",
        lambda **_: object(),
    )

    def fake_completion(*_: Any, **__: Any) -> SimpleNamespace:
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(
        "core.model_clients.openai_compatible.create_openai_chat_completion",
        fake_completion,
    )
    monkeypatch.setattr("core.model_clients.openai_compatible.time.sleep", lambda _: None)
    logger = logging.getLogger("test-openai-retry")
    client = OpenAICompatibleClient(
        endpoint_url="http://model.test/v1",
        model="test-model",
        retries=1,
        retry_backoff_s=0.0,
        logger=logger,
    )

    with caplog.at_level(logging.WARNING):
        assert client.generate(ChatRequest(prompt="Run.")) == "done"

    assert "Transient LLM endpoint error" in caplog.text
    assert "provider=openai-compatible" in caplog.text


def test_content_text_allows_llm_newline_joining() -> None:
    assert content_text([{"text": "first"}, {"text": "second"}], list_separator="\n") == (
        "first\nsecond"
    )


def test_endpoint_client_factory_preserves_provider_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    class _TaskEndpointError(RuntimeError):
        pass

    monkeypatch.setattr("core.model_clients.factory.OpenAICompatibleClient", _FakeClient)

    client = create_endpoint_client(
        provider="openai-compatible",
        endpoint_url="http://model.test/v1",
        model="test-model",
        timeout_s=10.0,
        retries=1,
        retry_backoff_s=0.5,
        error_cls=_TaskEndpointError,
    )

    assert isinstance(client, _FakeClient)
    assert captured == {
        "endpoint_url": "http://model.test/v1",
        "model": "test-model",
        "timeout_s": 10.0,
        "retries": 1,
        "retry_backoff_s": 0.5,
        "parser": "instruct",
        "error_cls": _TaskEndpointError,
        "logger": None,
    }


def test_api_key_env_constants_are_provider_fixed() -> None:
    assert NVIDIA_API_KEY_ENV == "NVIDIA_API_KEY"
    assert GEMINI_API_KEY_ENV == "GEMINI_API_KEY"


def test_resolve_api_key_reads_named_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(NVIDIA_API_KEY_ENV, "nv-secret")
    assert resolve_api_key(NVIDIA_API_KEY_ENV) == "nv-secret"


def test_resolve_api_key_returns_empty_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(NVIDIA_API_KEY_ENV, raising=False)
    assert resolve_api_key(NVIDIA_API_KEY_ENV) == "EMPTY"
    assert resolve_api_key(GEMINI_API_KEY_ENV, empty_value="") == ""


def test_gemini_generate_url_uses_template_and_api_key() -> None:
    assert (
        gemini_generate_url(
            "http://gemini.test/models/{model}:generateContent",
            "model/name",
            "key value",
        )
        == "http://gemini.test/models/model%2Fname:generateContent?key=key%20value"
    )


def test_gemini_client_parses_text(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> _Response:
        captured["url"] = request.full_url
        assert isinstance(request.data, bytes)
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _Response(
            b'{"candidates": [{"content": {"parts": [{"text": "hello"}]}, '
            b'"finishReason": "STOP"}], "usageMetadata": {"totalTokenCount": 2}}'
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv(GEMINI_API_KEY_ENV, "gemini-secret")
    client = GeminiClient(
        endpoint_url="http://gemini.test",
        model="gemini-test",
        timeout_s=10.0,
        retries=0,
    )

    assert client.generate(ChatRequest(prompt="Say hi.")) == "hello"
    assert captured["url"] == (
        "http://gemini.test/v1beta/models/gemini-test:generateContent?key=gemini-secret"
    )
    assert captured["payload"]["contents"][0]["parts"] == [{"text": "Say hi."}]


def test_post_json_rejects_non_http_url() -> None:
    with pytest.raises(ValueError, match="Unsupported endpoint URL scheme: file"):
        post_json(
            "file:///etc/passwd",
            {"ok": True},
            headers={"Content-Type": "application/json"},
            timeout_s=10.0,
            retries=0,
            retry_backoff_s=0.1,
        )
