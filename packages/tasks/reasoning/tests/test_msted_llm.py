# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# mypy: disable-error-code="no-untyped-def,unused-ignore"

"""Tests for the MSTED LLM aggregator.

Mocks ``call_chat_object_with_structured_fallback`` at the module
boundary — the goal is to verify the *adapter logic* (prompt rendering,
input filtering, error handling, schema hint construction). Actual LLM
behavior is out of scope; the converter on the other side validates
whatever JSON comes back."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import patch

import pytest
from reasoning.msted.llm import (
    MstedLLMError,
    generate_msted_with_llm,
    parse_cached_msted,
)
from reasoning.prompts import PromptVariant


def fake_prompt() -> PromptVariant:
    """A minimal prompt that echoes the placeholders into the rendered
    user message — lets tests assert what the adapter passed in."""
    return PromptVariant(
        name="fake",
        system="SYS",
        user_template=("WIN={windows_block}|SD={scene_description_block}|ES={event_summary_block}"),
    )


def windows() -> list[dict[str, Any]]:
    return [
        {"start_s": 0.0, "end_s": 4.0, "caption": "First segment"},
        {"start_s": 4.0, "end_s": 7.0, "caption": "Second segment"},
    ]


@pytest.fixture
def caplog_quiet(caplog):
    caplog.set_level(logging.DEBUG)
    return caplog


class TestHappyPath:
    def test_returns_parsed_dict(self):
        expected = {
            "scene_description": "x",
            "temporal_spatial_localization": [
                {"start": "00:00", "end": "00:04", "description": "y"}
            ],
            "event_description": {"category": "z"},
        }
        with patch(
            "reasoning.msted.llm.call_chat_object_with_structured_fallback",
            return_value=(expected, "{...}"),
        ) as mock_call:
            result = generate_msted_with_llm(
                windows=windows(),
                scene_description="A car drives.",
                event_summary="Right turn.",
                duration=10.0,
                prompt=fake_prompt(),
                llm_url="http://x/v1",
                llm_model="m",
                logger=logging.getLogger("test"),
            )
        assert result == expected
        mock_call.assert_called_once()

    def test_messages_carry_system_and_rendered_user(self):
        captured: dict[str, Any] = {}

        def stub(*, messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return (
                {
                    "scene_description": "s",
                    "temporal_spatial_localization": [
                        {"start": "00:00", "end": "00:01", "description": "d"}
                    ],
                    "event_description": {"k": "v"},
                },
                "",
            )

        with patch(
            "reasoning.msted.llm.call_chat_object_with_structured_fallback", side_effect=stub
        ):
            generate_msted_with_llm(
                windows=windows(),
                scene_description="SCENE_TEXT",
                event_summary="ES_TEXT",
                duration=None,
                prompt=fake_prompt(),
                llm_url="http://x/v1",
                llm_model="m",
            )

        msgs = captured["messages"]
        assert msgs[0] == {"role": "system", "content": "SYS"}
        user = msgs[1]
        assert user["role"] == "user"
        # Adapter formatted timecodes from raw seconds into MM:SS.
        assert "00:00-00:04: First segment" in user["content"]
        assert "00:04-00:07: Second segment" in user["content"]
        # Optional blocks rendered with their headers.
        assert "Scene description:" in user["content"]
        assert "SCENE_TEXT" in user["content"]
        assert "Event summary:" in user["content"]
        assert "ES_TEXT" in user["content"]

    def test_llm_endpoint_args_threaded_through(self):
        captured: dict[str, Any] = {}

        def stub(**kwargs):
            captured.update(kwargs)
            return (
                {
                    "scene_description": "s",
                    "temporal_spatial_localization": [
                        {"start": "00:00", "end": "00:01", "description": "d"}
                    ],
                    "event_description": {"k": "v"},
                },
                "",
            )

        with patch(
            "reasoning.msted.llm.call_chat_object_with_structured_fallback", side_effect=stub
        ):
            generate_msted_with_llm(
                windows=windows(),
                scene_description=None,
                event_summary=None,
                duration=None,
                prompt=fake_prompt(),
                llm_url="http://endpoint/v1",
                llm_model="my-model",
                max_tokens=512,
                temperature=0.7,
                top_p=0.9,
                timeout=42,
                structured_output="nim",
                seed=123,
                retries=4,
                retry_backoff_s=2.5,
                api_key="secret",
            )

        assert captured["base_url"] == "http://endpoint/v1"
        assert captured["model"] == "my-model"
        assert captured["max_tokens"] == 512
        assert captured["temperature"] == 0.7
        assert captured["top_p"] == 0.9
        assert captured["timeout"] == 42
        assert captured["structured_output"] == "nim"
        assert captured["seed"] == 123
        assert captured["retries"] == 4
        assert captured["retry_backoff_s"] == 2.5
        assert captured["api_key"] == "secret"
        assert captured["retry_stage"] == "msted"
        # Schema hint is sent (NIM uses it for guided_json; OpenAI ignores it).
        schema = captured["guided_json_schema"]
        assert schema["type"] == "object"
        assert "scene_description" in schema["properties"]


class TestInputFiltering:
    def test_skips_windows_missing_start_or_end(self, caplog_quiet):
        wins = [
            {"start_s": 0.0, "caption": "no end"},
            {"end_s": 5.0, "caption": "no start"},
            {"start_s": 0.0, "end_s": 5.0, "caption": "good"},
        ]
        captured: dict[str, Any] = {}

        def stub(*, messages, **kwargs):
            captured["messages"] = messages
            return (
                {
                    "scene_description": "s",
                    "temporal_spatial_localization": [
                        {"start": "00:00", "end": "00:05", "description": "good"}
                    ],
                    "event_description": {"k": "v"},
                },
                "",
            )

        with patch(
            "reasoning.msted.llm.call_chat_object_with_structured_fallback", side_effect=stub
        ):
            generate_msted_with_llm(
                windows=wins,
                scene_description=None,
                event_summary=None,
                duration=None,
                prompt=fake_prompt(),
                llm_url="http://x/v1",
                llm_model="m",
                logger=logging.getLogger("test"),
            )
        user = captured["messages"][1]["content"]
        assert "good" in user
        assert "no end" not in user
        assert "no start" not in user

    def test_skips_windows_missing_caption(self):
        wins = [
            {"start_s": 0.0, "end_s": 1.0},  # no caption
            {"start_s": 1.0, "end_s": 2.0, "caption": "real"},
        ]
        captured: dict[str, Any] = {}

        def stub(*, messages, **kwargs):
            captured["messages"] = messages
            return (
                {
                    "scene_description": "s",
                    "temporal_spatial_localization": [
                        {"start": "00:00", "end": "00:01", "description": "d"}
                    ],
                    "event_description": {"k": "v"},
                },
                "",
            )

        with patch(
            "reasoning.msted.llm.call_chat_object_with_structured_fallback", side_effect=stub
        ):
            generate_msted_with_llm(
                windows=wins,
                scene_description=None,
                event_summary=None,
                duration=None,
                prompt=fake_prompt(),
                llm_url="http://x/v1",
                llm_model="m",
            )
        user = captured["messages"][1]["content"]
        assert "real" in user
        # Filtered window doesn't show up.
        assert user.count(":") >= 2  # only the surviving window's "MM:SS-MM:SS:" prefix

    def test_description_keys_override(self):
        wins = [{"start_s": 0.0, "end_s": 1.0, "enhanced_caption": "ENH", "caption": "RAW"}]
        captured: dict[str, Any] = {}

        def stub(*, messages, **kwargs):
            captured["messages"] = messages
            return (
                {
                    "scene_description": "s",
                    "temporal_spatial_localization": [
                        {"start": "00:00", "end": "00:01", "description": "d"}
                    ],
                    "event_description": {"k": "v"},
                },
                "",
            )

        with patch(
            "reasoning.msted.llm.call_chat_object_with_structured_fallback", side_effect=stub
        ):
            generate_msted_with_llm(
                windows=wins,
                scene_description=None,
                event_summary=None,
                duration=None,
                prompt=fake_prompt(),
                llm_url="http://x/v1",
                llm_model="m",
                description_keys=("caption",),  # explicitly prefer raw caption
            )
        assert "RAW" in captured["messages"][1]["content"]
        assert "ENH" not in captured["messages"][1]["content"]

    def test_no_usable_windows_raises(self):
        wins = [{"foo": "bar"}, {"baz": 1}]
        with patch("reasoning.msted.llm.call_chat_object_with_structured_fallback") as mock_call:
            with pytest.raises(MstedLLMError, match="no usable windows"):
                generate_msted_with_llm(
                    windows=wins,
                    scene_description=None,
                    event_summary=None,
                    duration=None,
                    prompt=fake_prompt(),
                    llm_url="http://x/v1",
                    llm_model="m",
                )
        mock_call.assert_not_called()


class TestErrorHandling:
    def test_none_response_raises_with_snippet(self):
        with patch(
            "reasoning.msted.llm.call_chat_object_with_structured_fallback",
            return_value=(None, "garbled output here"),
        ):
            with pytest.raises(MstedLLMError, match="no parseable JSON"):
                generate_msted_with_llm(
                    windows=windows(),
                    scene_description=None,
                    event_summary=None,
                    duration=None,
                    prompt=fake_prompt(),
                    llm_url="http://x/v1",
                    llm_model="m",
                )

    def test_non_dict_response_raises(self):
        with patch(
            "reasoning.msted.llm.call_chat_object_with_structured_fallback",
            return_value=(["a", "list"], "[]"),
        ):
            with pytest.raises(MstedLLMError, match="non-object payload"):
                generate_msted_with_llm(
                    windows=windows(),
                    scene_description=None,
                    event_summary=None,
                    duration=None,
                    prompt=fake_prompt(),
                    llm_url="http://x/v1",
                    llm_model="m",
                )


class TestParseCachedMsted:
    def test_valid_cache_parsed(self):
        out = parse_cached_msted('{"scene_description": "x"}')
        assert out == {"scene_description": "x"}

    def test_invalid_json_raises(self):
        with pytest.raises(MstedLLMError, match="not valid JSON"):
            parse_cached_msted("not json {{}")

    def test_non_object_raises(self):
        with pytest.raises(MstedLLMError, match="must be an object"):
            parse_cached_msted("[1, 2, 3]")
