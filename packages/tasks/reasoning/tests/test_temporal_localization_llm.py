# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# mypy: disable-error-code="no-untyped-def,unused-ignore"

"""Tests for the temporal_localization LLM adapter and query-bank loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml
from reasoning.prompts import PromptVariant
from reasoning.temporal_localization.llm import (
    TemporalLocalizationLLMError,
    generate_temporal_localizations_with_llm,
    load_query_bank,
)


def fake_prompt() -> PromptVariant:
    return PromptVariant(
        name="fake_tl",
        system="TL_SYSTEM",
        user_template=("WIN={windows_block}\nSCENE={scene_description_block}\nQ={queries_block}"),
    )


def windows_fixture() -> list[dict]:
    return [
        {"start_s": 0.0, "end_s": 4.0, "caption": "car approaching intersection"},
        {"start_s": 4.0, "end_s": 7.0, "caption": "car turning right"},
        {"start_s": 7.0, "end_s": 10.0, "caption": "car driving away"},
    ]


def stub_object_response(obj: dict) -> Any:
    """Build a stub for ``call_chat_object_with_structured_fallback`` returning
    ``(parsed, raw)``."""

    def _stub(**kwargs):
        return obj, json.dumps(obj)

    return _stub


class TestQueryBank:
    def test_inline_only(self):
        out = load_query_bank(queries=["a", "b"], query_file=None)
        assert out == ["a", "b"]

    def test_dedupe_and_strip(self):
        out = load_query_bank(queries=["  a  ", "a", "b", "", "  "], query_file=None)
        assert out == ["a", "b"]

    def test_yaml_bare_list(self, tmp_path):
        f = tmp_path / "q.yaml"
        f.write_text(yaml.safe_dump(["q1", "q2"]), encoding="utf-8")
        out = load_query_bank(queries=None, query_file=f)
        assert out == ["q1", "q2"]

    def test_yaml_object_with_queries(self, tmp_path):
        f = tmp_path / "q.yaml"
        f.write_text(yaml.safe_dump({"queries": ["q1", "q2"]}), encoding="utf-8")
        out = load_query_bank(queries=None, query_file=f)
        assert out == ["q1", "q2"]

    def test_mcq_questions_with_text(self, tmp_path):
        f = tmp_path / "q.yaml"
        bank = {
            "questions": [
                {"text": "When does the car turn?", "type": "free"},
                {"text": "What color is the car?"},
            ]
        }
        f.write_text(yaml.safe_dump(bank), encoding="utf-8")
        out = load_query_bank(queries=None, query_file=f)
        assert out == ["When does the car turn?", "What color is the car?"]

    def test_json_file(self, tmp_path):
        f = tmp_path / "q.json"
        f.write_text(json.dumps(["q1"]), encoding="utf-8")
        out = load_query_bank(queries=None, query_file=f)
        assert out == ["q1"]

    def test_inline_plus_file_concatenated_and_deduped(self, tmp_path):
        f = tmp_path / "q.yaml"
        f.write_text(yaml.safe_dump(["b", "c"]), encoding="utf-8")
        out = load_query_bank(queries=["a", "b"], query_file=f)
        assert out == ["a", "b", "c"]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_query_bank(queries=None, query_file=tmp_path / "missing.yaml")

    def test_unknown_shape_raises(self, tmp_path):
        f = tmp_path / "q.yaml"
        f.write_text(yaml.safe_dump({"foo": "bar"}), encoding="utf-8")
        with pytest.raises(ValueError, match="must be a list"):
            load_query_bank(queries=None, query_file=f)

    def test_empty_returns_empty(self):
        assert load_query_bank(queries=None, query_file=None) == []


class TestQueryBankSection:
    """The unified-cookbook bank shape: file is a dict with per-task sections.

    Mirrors :class:`TestQaBankSection` in ``test_qa_llm.py``: when
    ``section="temporal_localization"`` is supplied, the loader pulls
    queries from that section in preference to the legacy bare-list /
    ``{"queries": [...]}`` / ``{"questions": [{"text": "..."}]}``
    shapes -- which keep working unchanged for callers that don't pass
    a section.
    """

    @staticmethod
    def _unified_bank() -> dict:
        return {
            "name": "test_bank",
            "version": "2.0",
            "questions": [
                {"id": "legacy_1", "text": "legacy MCQ question"},
            ],
            "temporal_localization": [
                {"id": "tl_1", "query": "When does the event start?"},
                {"id": "tl_2", "query": "When does the event end?"},
            ],
            "open_qa": [
                {"id": "oq_1", "question": "should NOT be picked up"},
            ],
        }

    def test_section_picks_temporal_localization(self, tmp_path):
        f = tmp_path / "bank.json"
        f.write_text(json.dumps(self._unified_bank()), encoding="utf-8")
        out = load_query_bank(queries=None, query_file=f, section="temporal_localization")
        assert out == [
            "When does the event start?",
            "When does the event end?",
        ]

    def test_no_section_falls_back_to_legacy_questions(self, tmp_path):
        f = tmp_path / "bank.json"
        f.write_text(json.dumps(self._unified_bank()), encoding="utf-8")
        out = load_query_bank(queries=None, query_file=f)
        assert out == ["legacy MCQ question"]

    def test_missing_section_falls_back_to_legacy_questions(self, tmp_path):
        f = tmp_path / "bank.json"
        f.write_text(json.dumps(self._unified_bank()), encoding="utf-8")
        out = load_query_bank(queries=None, query_file=f, section="not_in_file")
        assert out == ["legacy MCQ question"]

    def test_section_value_must_be_a_list(self, tmp_path):
        """If the named section exists but isn't a list, fall back rather than crash."""
        f = tmp_path / "bank.json"
        f.write_text(
            json.dumps(
                {
                    "temporal_localization": "not-a-list",
                    "queries": ["fallback query"],
                }
            ),
            encoding="utf-8",
        )
        out = load_query_bank(queries=None, query_file=f, section="temporal_localization")
        assert out == ["fallback query"]

    def test_section_entries_can_be_plain_strings(self, tmp_path):
        """The section may contain raw strings (skipping the ``query`` key)."""
        f = tmp_path / "bank.json"
        f.write_text(
            json.dumps({"temporal_localization": ["raw string 1", "raw string 2"]}),
            encoding="utf-8",
        )
        out = load_query_bank(queries=None, query_file=f, section="temporal_localization")
        assert out == ["raw string 1", "raw string 2"]

    def test_section_dict_entries_use_query_text_or_question(self, tmp_path):
        """Dict entries contribute ``query`` first, then ``text``, then ``question``."""
        f = tmp_path / "bank.json"
        f.write_text(
            json.dumps(
                {
                    "temporal_localization": [
                        {"query": "from query"},
                        {"text": "from text"},
                        {"question": "from question"},
                        {"unrelated": "ignored"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        out = load_query_bank(queries=None, query_file=f, section="temporal_localization")
        assert out == ["from query", "from text", "from question"]

    def test_yaml_sectioned_bank(self, tmp_path):
        f = tmp_path / "bank.yaml"
        f.write_text(yaml.safe_dump(self._unified_bank()), encoding="utf-8")
        out = load_query_bank(queries=None, query_file=f, section="temporal_localization")
        assert out == [
            "When does the event start?",
            "When does the event end?",
        ]

    def test_inline_plus_section_file_concatenated_and_deduped(self, tmp_path):
        """Inline ``queries`` are prepended to the resolved section list and deduped."""
        f = tmp_path / "bank.json"
        f.write_text(json.dumps(self._unified_bank()), encoding="utf-8")
        out = load_query_bank(
            queries=["When does the event end?", "extra inline query"],
            query_file=f,
            section="temporal_localization",
        )
        assert out == [
            "When does the event end?",
            "extra inline query",
            "When does the event start?",
        ]

    def test_warehouse_cookbook_bank_round_trip(self):
        """Smoke check against the real warehouse cookbook bank.

        Validates that the cookbook JSON parses, has a
        ``temporal_localization`` section, and yields at least one
        query string. This is the regression fence we actually care
        about: if anyone breaks the file structure of
        ``cookbooks/warehouse/question_bank.json``, this test fails.
        """
        bank_path = (
            Path(__file__).resolve().parents[2] / "cookbooks" / "warehouse" / "question_bank.json"
        )
        if not bank_path.is_file():
            pytest.skip(f"warehouse bank not present at {bank_path}")
        out = load_query_bank(
            queries=None,
            query_file=bank_path,
            section="temporal_localization",
        )
        assert out, "warehouse cookbook temporal_localization section resolved empty"
        assert all(isinstance(q, str) and q.strip() for q in out)


class TestHappyPath:
    def test_returns_parsed_items(self):
        good = {
            "items": [
                {
                    "question": "When does the car turn right?",
                    "answer": {"start": "00:04", "end": "00:07"},
                    "reasoning": "Caption at 00:04-00:07 describes a right turn.",
                }
            ]
        }
        with patch(
            "reasoning.temporal_localization.llm.call_chat_object_with_structured_fallback",
            side_effect=stub_object_response(good),
        ):
            items = generate_temporal_localizations_with_llm(
                queries=["When does the car turn right?"],
                windows=windows_fixture(),
                scene_description="Urban intersection.",
                duration=10.0,
                prompt=fake_prompt(),
                llm_url="http://x/v1",
                llm_model="m",
            )
        assert len(items) == 1
        # Adapter defaults t1=0 and t2=duration when LLM omits them.
        assert items[0]["t1"] == 0.0
        assert items[0]["t2"] == 10.0

    def test_messages_have_system_and_rendered_user(self):
        captured: dict[str, Any] = {}

        def stub(*, messages, **kwargs):
            captured["messages"] = messages
            return {
                "items": [{"question": "Q?", "answer": {"start": "00:00", "end": "00:01"}}]
            }, "{}"

        with patch(
            "reasoning.temporal_localization.llm.call_chat_object_with_structured_fallback",
            side_effect=stub,
        ):
            generate_temporal_localizations_with_llm(
                queries=["Q?"],
                windows=windows_fixture(),
                scene_description="hi",
                duration=10.0,
                prompt=fake_prompt(),
                llm_url="http://x/v1",
                llm_model="m",
            )
        assert captured["messages"][0]["content"] == "TL_SYSTEM"
        user = captured["messages"][1]["content"]
        # All windows rendered as "MM:SS-MM:SS: caption" lines.
        assert "00:00-00:04: car approaching intersection" in user
        assert "00:04-00:07: car turning right" in user
        # Scene description block included.
        assert "hi" in user
        # Query bullet present.
        assert "- Q?" in user

    def test_inline_t1_t2_preserved_over_default(self):
        good = {
            "items": [
                {
                    "question": "Q?",
                    "t1": "00:01",
                    "t2": "00:09",
                    "answer": {"start": "00:04", "end": "00:07"},
                }
            ]
        }
        with patch(
            "reasoning.temporal_localization.llm.call_chat_object_with_structured_fallback",
            side_effect=stub_object_response(good),
        ):
            items = generate_temporal_localizations_with_llm(
                queries=["Q?"],
                windows=windows_fixture(),
                scene_description=None,
                duration=10.0,
                prompt=fake_prompt(),
                llm_url="http://x/v1",
                llm_model="m",
            )
        assert items[0]["t1"] == "00:01"
        assert items[0]["t2"] == "00:09"

    def test_no_duration_no_t2_default(self):
        good = {
            "items": [
                {
                    "question": "Q?",
                    "answer": {"start": "00:04", "end": "00:07"},
                }
            ]
        }
        with patch(
            "reasoning.temporal_localization.llm.call_chat_object_with_structured_fallback",
            side_effect=stub_object_response(good),
        ):
            items = generate_temporal_localizations_with_llm(
                queries=["Q?"],
                windows=windows_fixture(),
                scene_description=None,
                duration=None,
                prompt=fake_prompt(),
                llm_url="http://x/v1",
                llm_model="m",
            )
        # Adapter defaults t1=0; t2 unfilled (converter will then raise
        # if duration also missing — that's the converter's job, not ours).
        assert items[0]["t1"] == 0.0
        assert "t2" not in items[0]


class TestErrors:
    def test_empty_queries_raises_before_llm(self):
        with patch(
            "reasoning.temporal_localization.llm.call_chat_object_with_structured_fallback"
        ) as mock:
            with pytest.raises(TemporalLocalizationLLMError, match="no usable queries"):
                generate_temporal_localizations_with_llm(
                    queries=["", "  "],
                    windows=windows_fixture(),
                    scene_description=None,
                    duration=10.0,
                    prompt=fake_prompt(),
                    llm_url="http://x/v1",
                    llm_model="m",
                )
        mock.assert_not_called()

    def test_no_usable_windows_raises(self):
        with patch(
            "reasoning.temporal_localization.llm.call_chat_object_with_structured_fallback"
        ) as mock:
            with pytest.raises(TemporalLocalizationLLMError, match="no usable windows"):
                generate_temporal_localizations_with_llm(
                    queries=["Q?"],
                    windows=[{"caption": "missing times"}, {"start_s": 0.0}],
                    scene_description=None,
                    duration=10.0,
                    prompt=fake_prompt(),
                    llm_url="http://x/v1",
                    llm_model="m",
                )
        mock.assert_not_called()

    def test_unparseable_response_raises(self):
        with patch(
            "reasoning.temporal_localization.llm.call_chat_object_with_structured_fallback",
            return_value=(None, "not-json"),
        ):
            with pytest.raises(TemporalLocalizationLLMError, match="no parseable JSON"):
                generate_temporal_localizations_with_llm(
                    queries=["Q?"],
                    windows=windows_fixture(),
                    scene_description=None,
                    duration=10.0,
                    prompt=fake_prompt(),
                    llm_url="http://x/v1",
                    llm_model="m",
                )

    def test_missing_items_key_raises(self):
        with patch(
            "reasoning.temporal_localization.llm.call_chat_object_with_structured_fallback",
            side_effect=stub_object_response({"results": []}),
        ):
            with pytest.raises(TemporalLocalizationLLMError, match="missing 'items'"):
                generate_temporal_localizations_with_llm(
                    queries=["Q?"],
                    windows=windows_fixture(),
                    scene_description=None,
                    duration=10.0,
                    prompt=fake_prompt(),
                    llm_url="http://x/v1",
                    llm_model="m",
                )

    def test_empty_items_raises(self):
        with patch(
            "reasoning.temporal_localization.llm.call_chat_object_with_structured_fallback",
            side_effect=stub_object_response({"items": []}),
        ):
            with pytest.raises(TemporalLocalizationLLMError, match="empty 'items'"):
                generate_temporal_localizations_with_llm(
                    queries=["Q?"],
                    windows=windows_fixture(),
                    scene_description=None,
                    duration=10.0,
                    prompt=fake_prompt(),
                    llm_url="http://x/v1",
                    llm_model="m",
                )

    def test_non_dict_items_dropped(self):
        good = {
            "items": [
                "garbage string",
                {"question": "Q?", "answer": {"start": "00:00", "end": "00:01"}},
            ]
        }
        with patch(
            "reasoning.temporal_localization.llm.call_chat_object_with_structured_fallback",
            side_effect=stub_object_response(good),
        ):
            items = generate_temporal_localizations_with_llm(
                queries=["Q?"],
                windows=windows_fixture(),
                scene_description=None,
                duration=10.0,
                prompt=fake_prompt(),
                llm_url="http://x/v1",
                llm_model="m",
            )
        assert len(items) == 1
        assert items[0]["question"] == "Q?"

    def test_all_non_dict_items_raises(self):
        with patch(
            "reasoning.temporal_localization.llm.call_chat_object_with_structured_fallback",
            side_effect=stub_object_response({"items": ["a", "b"]}),
        ):
            with pytest.raises(TemporalLocalizationLLMError, match="none were dicts"):
                generate_temporal_localizations_with_llm(
                    queries=["Q?"],
                    windows=windows_fixture(),
                    scene_description=None,
                    duration=10.0,
                    prompt=fake_prompt(),
                    llm_url="http://x/v1",
                    llm_model="m",
                )


class TestPromptInputs:
    def test_no_scene_description_uses_fallback(self):
        captured: dict[str, Any] = {}

        def stub(*, messages, **kwargs):
            captured["messages"] = messages
            return {
                "items": [{"question": "Q?", "answer": {"start": "00:00", "end": "00:01"}}]
            }, "{}"

        with patch(
            "reasoning.temporal_localization.llm.call_chat_object_with_structured_fallback",
            side_effect=stub,
        ):
            generate_temporal_localizations_with_llm(
                queries=["Q?"],
                windows=windows_fixture(),
                scene_description=None,
                duration=10.0,
                prompt=fake_prompt(),
                llm_url="http://x/v1",
                llm_model="m",
            )
        assert "(no scene-level description provided)" in captured["messages"][1]["content"]

    def test_endpoint_args_threaded(self):
        captured: dict[str, Any] = {}

        def stub(**kwargs):
            captured.update(kwargs)
            return {
                "items": [{"question": "Q?", "answer": {"start": "00:00", "end": "00:01"}}]
            }, "{}"

        with patch(
            "reasoning.temporal_localization.llm.call_chat_object_with_structured_fallback",
            side_effect=stub,
        ):
            generate_temporal_localizations_with_llm(
                queries=["Q?"],
                windows=windows_fixture(),
                scene_description=None,
                duration=10.0,
                prompt=fake_prompt(),
                llm_url="http://endpoint/v1",
                llm_model="my-model",
                max_tokens=2048,
                temperature=0.1,
                top_p=0.9,
                timeout=60,
                structured_output="nim",
                seed=11,
                retries=5,
                retry_backoff_s=2.0,
                api_key="apikey",
            )
        assert captured["base_url"] == "http://endpoint/v1"
        assert captured["model"] == "my-model"
        assert captured["max_tokens"] == 2048
        assert captured["temperature"] == 0.1
        assert captured["top_p"] == 0.9
        assert captured["timeout"] == 60
        assert captured["structured_output"] == "nim"
        assert captured["seed"] == 11
        assert captured["retries"] == 5
        assert captured["retry_backoff_s"] == 2.0
        assert captured["api_key"] == "apikey"
        assert captured["retry_stage"] == "temporal_localization"

    def test_custom_description_keys_used(self):
        captured: dict[str, Any] = {}

        def stub(*, messages, **kwargs):
            captured["messages"] = messages
            return {
                "items": [{"question": "Q?", "answer": {"start": "00:00", "end": "00:01"}}]
            }, "{}"

        windows = [
            {"start_s": 0.0, "end_s": 4.0, "custom_text": "from custom field"},
        ]
        with patch(
            "reasoning.temporal_localization.llm.call_chat_object_with_structured_fallback",
            side_effect=stub,
        ):
            generate_temporal_localizations_with_llm(
                queries=["Q?"],
                windows=windows,
                scene_description=None,
                duration=10.0,
                prompt=fake_prompt(),
                llm_url="http://x/v1",
                llm_model="m",
                description_keys=("custom_text",),
            )
        assert "from custom field" in captured["messages"][1]["content"]
