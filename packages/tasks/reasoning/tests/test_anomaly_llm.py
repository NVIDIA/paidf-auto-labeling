# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# mypy: disable-error-code="no-untyped-def,unused-ignore"

"""Tests for the anomaly classification LLM adapter.

Mocks ``call_chat_object_with_structured_fallback`` at the module
boundary — the goal is to verify adapter logic (prompt rendering, input
filtering, verdict normalization, highlight validation, error handling).
Actual LLM behavior is out of scope."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import patch

import pytest
from reasoning.anomaly.llm import AnomalyLLMError, classify_anomaly_with_llm
from reasoning.prompts import PromptVariant

_PATCH_TARGET = "reasoning.anomaly.llm.call_chat_object_with_structured_fallback"


def fake_prompt() -> PromptVariant:
    return PromptVariant(
        name="fake",
        system="SYS",
        user_template="WIN={windows_block}|SD={scene_description_block}",
    )


def windows() -> list[dict[str, Any]]:
    return [
        {"start_s": 0.0, "end_s": 4.0, "caption": "A car approaches the junction"},
        {"start_s": 4.0, "end_s": 7.0, "caption": "The car collides with a barrier"},
    ]


def _call(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "windows": windows(),
        "scene_description": "A dashcam clip.",
        "prompt": fake_prompt(),
        "llm_url": "http://x/v1",
        "llm_model": "m",
        "logger": logging.getLogger("test"),
    }
    kwargs.update(overrides)
    return classify_anomaly_with_llm(**kwargs)


class TestHappyPath:
    def test_anomaly_verdict_normalized(self):
        raw = {
            "classification": "Anomaly",
            "confidence": "High",
            "reasoning": "  car hits barrier  ",
            "root_cause": " loss of control ",
            "consequence": " collision ",
            "highlight": {"start": "00:04", "end": "00:07", "description": " impact "},
        }
        with patch(_PATCH_TARGET, return_value=(raw, "{...}")):
            out = _call()
        assert out["classification"] == "anomaly"
        assert out["confidence"] == "high"
        assert out["reasoning"] == "car hits barrier"
        assert out["root_cause"] == "loss of control"
        assert out["consequence"] == "collision"
        assert out["highlight"] == {
            "start": "00:04",
            "end": "00:07",
            "description": "impact",
        }

    def test_normal_verdict_drops_anomaly_only_fields(self):
        raw = {
            "classification": "normal",
            "reasoning": "routine traffic",
            # A confused model may still emit these for a normal scene;
            # the adapter must strip them.
            "root_cause": "n/a",
            "consequence": "n/a",
            "highlight": {"start": "00:00", "end": "00:02"},
        }
        with patch(_PATCH_TARGET, return_value=(raw, "{...}")):
            out = _call()
        assert out["classification"] == "normal"
        assert "root_cause" not in out
        assert "consequence" not in out
        assert "highlight" not in out
        assert out["confidence"] is None

    def test_messages_carry_system_and_rendered_user(self):
        captured: dict[str, Any] = {}

        def stub(*, messages, **kwargs):
            captured["messages"] = messages
            return ({"classification": "normal", "reasoning": "x"}, "")

        with patch(_PATCH_TARGET, side_effect=stub):
            _call(scene_description="SCENE_TEXT")

        msgs = captured["messages"]
        assert msgs[0] == {"role": "system", "content": "SYS"}
        user = msgs[1]["content"]
        assert "00:00-00:04: A car approaches the junction" in user
        assert "Scene description:" in user
        assert "SCENE_TEXT" in user

    def test_person_attributes_rendered_when_template_references_block(self):
        captured: dict[str, Any] = {}

        def stub(*, messages, **kwargs):
            captured["messages"] = messages
            return ({"classification": "anomaly", "reasoning": "x"}, "")

        prompt = PromptVariant(
            name="fake_pa",
            system="SYS",
            user_template="WIN={windows_block}|PA={person_attributes_block}",
        )
        with patch(_PATCH_TARGET, side_effect=stub):
            _call(prompt=prompt, person_attributes="Person track 0: potential anomaly=yes")

        user = captured["messages"][1]["content"]
        assert "Person attributes (from person-attribute search):" in user
        assert "Person track 0: potential anomaly=yes" in user

    def test_person_attributes_absent_renders_fallback(self):
        captured: dict[str, Any] = {}

        def stub(*, messages, **kwargs):
            captured["messages"] = messages
            return ({"classification": "normal", "reasoning": "x"}, "")

        prompt = PromptVariant(
            name="fake_pa",
            system="SYS",
            user_template="WIN={windows_block}|PA={person_attributes_block}",
        )
        with patch(_PATCH_TARGET, side_effect=stub):
            _call(prompt=prompt)  # no person_attributes passed

        user = captured["messages"][1]["content"]
        assert "(no person-attribute evidence provided)" in user

    def test_endpoint_args_threaded_through(self):
        captured: dict[str, Any] = {}

        def stub(**kwargs):
            captured.update(kwargs)
            return ({"classification": "anomaly", "reasoning": "x"}, "")

        with patch(_PATCH_TARGET, side_effect=stub):
            _call(
                max_tokens=512,
                temperature=0.5,
                top_p=0.9,
                timeout=42,
                structured_output="nim",
                seed=7,
                retries=4,
                retry_backoff_s=2.5,
                api_key="secret",
            )

        assert captured["base_url"] == "http://x/v1"
        assert captured["max_tokens"] == 512
        assert captured["temperature"] == 0.5
        assert captured["top_p"] == 0.9
        assert captured["timeout"] == 42
        assert captured["structured_output"] == "nim"
        assert captured["seed"] == 7
        assert captured["retries"] == 4
        assert captured["retry_backoff_s"] == 2.5
        assert captured["api_key"] == "secret"
        assert captured["retry_stage"] == "anomaly"
        schema = captured["guided_json_schema"]
        assert schema["properties"]["classification"]["enum"] == ["anomaly", "normal"]


class TestHighlightValidation:
    def test_localize_highlight_false_drops_highlight(self):
        raw = {
            "classification": "anomaly",
            "reasoning": "x",
            "highlight": {"start": "00:04", "end": "00:07"},
        }
        with patch(_PATCH_TARGET, return_value=(raw, "")):
            out = _call(localize_highlight=False)
        assert "highlight" not in out

    def test_reversed_window_dropped(self):
        raw = {
            "classification": "anomaly",
            "reasoning": "x",
            "highlight": {"start": "00:07", "end": "00:04"},
        }
        with patch(_PATCH_TARGET, return_value=(raw, "")):
            out = _call()
        assert "highlight" not in out

    def test_invalid_timecode_dropped(self):
        raw = {
            "classification": "anomaly",
            "reasoning": "x",
            "highlight": {"start": "00:04-00:07", "end": "00:09"},
        }
        with patch(_PATCH_TARGET, return_value=(raw, "")):
            out = _call()
        assert "highlight" not in out


class TestErrorHandling:
    def test_missing_classification_raises(self):
        with patch(_PATCH_TARGET, return_value=({"reasoning": "x"}, "")):
            with pytest.raises(AnomalyLLMError, match="classification"):
                _call()

    def test_invalid_classification_raises(self):
        with patch(_PATCH_TARGET, return_value=({"classification": "weird"}, "")):
            with pytest.raises(AnomalyLLMError, match="classification"):
                _call()

    def test_empty_reasoning_raises(self):
        raw = {"classification": "normal", "reasoning": "  "}
        with patch(_PATCH_TARGET, return_value=(raw, "")):
            with pytest.raises(AnomalyLLMError, match="reasoning"):
                _call()

    def test_missing_reasoning_raises(self):
        with patch(_PATCH_TARGET, return_value=({"classification": "anomaly"}, "")):
            with pytest.raises(AnomalyLLMError, match="reasoning"):
                _call()

    def test_no_usable_windows_raises(self):
        with patch(_PATCH_TARGET) as mock_call:
            with pytest.raises(AnomalyLLMError, match="no usable windows"):
                _call(windows=[{"foo": "bar"}])
        mock_call.assert_not_called()

    def test_none_response_raises(self):
        with patch(_PATCH_TARGET, return_value=(None, "garbled")):
            with pytest.raises(AnomalyLLMError, match="no parseable JSON"):
                _call()

    def test_non_object_response_raises(self):
        with patch(_PATCH_TARGET, return_value=(["a"], "[]")):
            with pytest.raises(AnomalyLLMError, match="non-object payload"):
                _call()
