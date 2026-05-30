# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging

from mcq_generation.mcq.utils import openai as chat


def test_call_chat_structured_accepts_fenced_json_window_module(monkeypatch) -> None:
    def _fake_call_chat_raw(**_kwargs):
        return "```json\n" + '{"mcq":[{"id":"1","question":"q","options":["a","b"],"answer":"a"}]}' + "\n```"

    monkeypatch.setattr(chat, "call_chat_raw", _fake_call_chat_raw)

    obj = chat.call_chat_structured_guided_json(
        base_url="http://example/v1",
        model="x",
        messages=[{"role": "user", "content": "hi"}],
        timeout=1,
        max_tokens=32,
        temperature=0.0,
        top_p=1.0,
        logger=logging.getLogger("test"),
    )
    assert isinstance(obj, dict)
    assert isinstance(obj.get("mcq"), list)
    assert obj["mcq"][0]["id"] == "1"


def test_call_chat_structured_accepts_prefixed_text_llm_module(monkeypatch) -> None:
    def _fake_call_chat_raw(**_kwargs):
        return (
            "Here is the MCQ JSON:\n\n```json\n"
            + '{"mcq":[{"id":"1","question":"q","options":["a","b"],"answer":"a"}]}'
            + "\n```"
        )

    monkeypatch.setattr(chat, "call_chat_raw", _fake_call_chat_raw)

    obj = chat.call_chat_structured_guided_json(
        base_url="http://example/v1",
        model="x",
        messages=[{"role": "user", "content": "hi"}],
        timeout=1,
        max_tokens=32,
        temperature=0.0,
        top_p=1.0,
        logger=logging.getLogger("test"),
    )
    assert isinstance(obj, dict)
    assert isinstance(obj.get("mcq"), list)
    assert obj["mcq"][0]["id"] == "1"
