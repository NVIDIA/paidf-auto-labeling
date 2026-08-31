# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import urllib.error
import urllib.request
from email.message import Message
from types import SimpleNamespace, TracebackType
from typing import Any, Literal

import pytest
from captioning.clients import ChatRequest, MediaPayload
from core.model_clients import OpenAICompatibleClient
from core.model_clients.http import post_json


class _Response:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> "_Response":
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


def _http_error(code: int) -> urllib.error.HTTPError:
    headers: Message = Message()
    return urllib.error.HTTPError(
        url="http://caption.test/v1/chat/completions",
        code=code,
        msg="error",
        hdrs=headers,
        fp=None,
    )


def test_post_json_retries_transient_http_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    sleeps: list[float] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> _Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _http_error(500)
        return _Response(b'{"ok": true}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr("core.model_clients.http.time.sleep", sleeps.append)

    result = post_json(
        "http://caption.test/v1/chat/completions",
        {"model": "test"},
        headers={"Content-Type": "application/json"},
        timeout_s=10.0,
        retries=2,
        retry_backoff_s=0.25,
    )

    assert result == {"ok": True}
    assert calls == 2
    assert sleeps == [0.25]


def test_openai_messages_use_video_url_for_video_payloads() -> None:
    messages = OpenAICompatibleClient._messages(
        ChatRequest(
            prompt="Describe the clip.",
            media=(
                MediaPayload(
                    mime_type="video/mp4",
                    data_base64="AAAA",
                    filename="clip.mp4",
                ),
            ),
        )
    )

    content = messages[0]["content"]
    assert content[0] == {
        "type": "video_url",
        "video_url": {"url": "data:video/mp4;base64,AAAA"},
    }
    assert content[1] == {"type": "text", "text": "Describe the clip."}


def test_openai_messages_keep_image_url_for_image_payloads() -> None:
    messages = OpenAICompatibleClient._messages(
        ChatRequest(
            prompt="Describe the frame.",
            media=(
                MediaPayload(
                    mime_type="image/jpeg",
                    data_base64="BBBB",
                    filename="frame.jpg",
                ),
            ),
        )
    )

    content = messages[0]["content"]
    assert content[0] == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,BBBB"},
    }
    assert content[1] == {"type": "text", "text": "Describe the frame."}


def test_openai_client_uses_sdk_chat_completions(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeCompletions:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def create(self, **kwargs: Any) -> SimpleNamespace:
            self.calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content="caption text"),
                    )
                ],
                usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 7}),
            )

    class _FakeOpenAI:
        instances: list["_FakeOpenAI"] = []

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.chat = SimpleNamespace(completions=_FakeCompletions())
            self.instances.append(self)

    monkeypatch.setenv("NVIDIA_API_KEY", "secret")
    monkeypatch.setattr("core.model_clients.openai_compatible.OpenAI", _FakeOpenAI)

    client = OpenAICompatibleClient(
        endpoint_url="http://caption.test/v1/chat/completions",
        model="vlm-test",
        timeout_s=10.0,
        retries=0,
    )

    result = client.generate(ChatRequest(prompt="Describe.", max_tokens=64, temperature=0.1))

    assert result == "caption text"
    assert _FakeOpenAI.instances[0].kwargs == {
        "api_key": "secret",
        "base_url": "http://caption.test/v1",
        "timeout": 10.0,
        "max_retries": 0,
    }
    call = _FakeOpenAI.instances[0].chat.completions.calls[0]
    assert call["model"] == "vlm-test"
    assert call["max_tokens"] == 64
    assert call["messages"][0]["content"] == [{"type": "text", "text": "Describe."}]
    assert client.last_call_metadata == {
        "provider": "openai-compatible",
        "model": "vlm-test",
        "endpoint_url": "http://caption.test/v1/chat/completions",
        "finish_reason": "stop",
        "usage": {"total_tokens": 7},
    }


def test_post_json_rejects_non_http_schemes_before_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_request(*args: object, **kwargs: object) -> None:
        raise AssertionError("Request should not be created for unsafe schemes")

    monkeypatch.setattr(urllib.request, "Request", fail_request)

    with pytest.raises(ValueError, match="Unsupported endpoint URL scheme: file"):
        post_json(
            "file:///etc/passwd",
            {"model": "test"},
            headers={"Content-Type": "application/json"},
            timeout_s=10.0,
            retries=2,
            retry_backoff_s=0.25,
        )


def test_post_json_does_not_retry_non_transient_http_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    sleeps: list[float] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> _Response:
        nonlocal calls
        calls += 1
        raise _http_error(400)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr("core.model_clients.http.time.sleep", sleeps.append)

    with pytest.raises(urllib.error.HTTPError) as raised:
        post_json(
            "http://caption.test/v1/chat/completions",
            {"model": "test"},
            headers={"Content-Type": "application/json"},
            timeout_s=10.0,
            retries=2,
            retry_backoff_s=0.25,
        )

    assert raised.value.code == 400
    assert calls == 1
    assert sleeps == []


def test_post_json_does_not_retry_json_decode_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    sleeps: list[float] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> _Response:
        nonlocal calls
        calls += 1
        return _Response(b"not-json")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr("core.model_clients.http.time.sleep", sleeps.append)

    with pytest.raises(json.JSONDecodeError):
        post_json(
            "http://caption.test/v1/chat/completions",
            {"model": "test"},
            headers={"Content-Type": "application/json"},
            timeout_s=10.0,
            retries=2,
            retry_backoff_s=0.25,
        )

    assert calls == 1
    assert sleeps == []


def test_post_json_raises_last_retryable_error_after_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    errors = [
        urllib.error.URLError("temporary dns failure"),
        urllib.error.URLError("connection reset"),
    ]

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> _Response:
        nonlocal calls
        error = errors[calls]
        calls += 1
        raise error

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr("core.model_clients.http.time.sleep", lambda seconds: None)

    with pytest.raises(urllib.error.URLError) as raised:
        post_json(
            "http://caption.test/v1/chat/completions",
            {"model": "test"},
            headers={"Content-Type": "application/json"},
            timeout_s=10.0,
            retries=1,
            retry_backoff_s=0.25,
        )

    assert raised.value is errors[-1]
    assert calls == 2
