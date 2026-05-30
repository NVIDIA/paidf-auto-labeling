# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from modules.mcq_generation.mcq.utils import openai as mcq_openai


def test_mcq_call_chat_raw_retries_with_exponential_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify:
    - retries happen on retryable exception
    - backoff uses 5s, 10s, 20s... when retry_backoff_s=5
    - total attempts == retries+1
    """
    sleeps: list[float] = []
    monkeypatch.setattr(mcq_openai.time, "sleep", lambda s: sleeps.append(float(s)))

    class APIConnectionError(Exception):
        pass

    calls = {"n": 0}

    class _FakeClient:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        class chat:
            class completions:
                @staticmethod
                def create(**_: Any):
                    calls["n"] += 1
                    if calls["n"] <= 3:
                        raise APIConnectionError("connection refused")
                    return SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                    )

    monkeypatch.setattr(mcq_openai, "OpenAI", _FakeClient)

    out = mcq_openai.call_chat_raw(
        base_url="http://localhost:1/v1",
        model="x",
        messages=[{"role": "user", "content": "hi"}],
        timeout=1,
        max_tokens=1,
        temperature=0.0,
        top_p=1.0,
        retries=3,
        retry_backoff_s=5.0,
        logger=None,
    )
    assert out == "ok"
    assert calls["n"] == 4
    assert sleeps == [5.0, 10.0, 20.0]


def test_structured_json_parse_retries_with_exponential_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify structured JSON parse retry path in call_chat_json_with_structured_fallback:
    - invalid JSON causes retry
    - backoff is 5s, 10s... when retry_backoff_s=5
    - returns parsed object once valid JSON appears
    """
    sleeps: list[float] = []
    monkeypatch.setattr(mcq_openai.time, "sleep", lambda s: sleeps.append(float(s)))

    calls = {"n": 0}

    def _fake_call_chat_raw(**kwargs: Any) -> str:
        calls["n"] += 1
        # First two attempts are non-JSON; third becomes valid.
        if calls["n"] == 1:
            return "not json"
        if calls["n"] == 2:
            return "still not json"
        # Minimal valid object for MCQ parser flow.
        return '{"mcq": []}'

    monkeypatch.setattr(mcq_openai, "call_chat_raw", _fake_call_chat_raw)

    obj, raw = mcq_openai.call_chat_json_with_structured_fallback(
        base_url="http://localhost:1/v1",
        model="x",
        messages=[{"role": "user", "content": "hi"}],
        timeout=1,
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
        logger=None,
        retries=3,
        retry_backoff_s=5.0,
        structured_output="openai",
    )

    assert obj == {"mcq": []}
    assert raw == '{"mcq": []}'
    assert calls["n"] == 3
    assert sleeps == [5.0, 10.0]
