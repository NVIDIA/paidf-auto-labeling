# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# mypy: disable-error-code="no-untyped-def,unused-ignore"

"""Tests for the causal_linkage LLM adapter, pair-derivation, and bank loader."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest
import yaml
from reasoning.causal_linkage.llm import (
    DEFAULT_QUESTION_TEMPLATE,
    CausalLinkageLLMError,
    derive_pairs_from_events,
    generate_causal_linkages_with_llm,
    load_pair_bank,
)
from reasoning.prompts import PromptVariant


def fake_prompt() -> PromptVariant:
    return PromptVariant(
        name="fake_cl",
        system="CL_SYSTEM",
        user_template=("WIN={windows_block}\nSCENE={scene_description_block}\nPAIRS={pairs_block}"),
    )


def windows_fixture() -> list[dict]:
    return [
        {"start_s": 0.0, "end_s": 4.0, "caption": "captionA"},
        {"start_s": 4.0, "end_s": 7.0, "caption": "captionB"},
    ]


def stub_object_response(obj: dict) -> Any:
    def _stub(**kwargs):
        return obj, json.dumps(obj)

    return _stub


# ---------------------------------------------------------------------------
# derive_pairs_from_events
# ---------------------------------------------------------------------------


class TestDerivePairs:
    def test_consecutive_pairs(self):
        events = [
            {"event_id": "e1", "start_time": 1.0, "end_time": 3.0},
            {"event_id": "e2", "start_time": 5.0, "end_time": 7.0},
            {"event_id": "e3", "start_time": 9.0, "end_time": 10.0},
        ]
        pairs = derive_pairs_from_events(events, duration=10.0)
        assert len(pairs) == 3
        assert pairs[0]["t1"] == "00:01"
        assert pairs[0]["t2"] == "00:05"
        assert pairs[1]["t1"] == "00:05"
        assert pairs[1]["t2"] == "00:09"
        # Last event uses duration as the trailing anchor
        assert pairs[2]["t1"] == "00:09"
        assert pairs[2]["t2"] == "00:10"

    def test_question_default_template(self):
        events = [
            {"event_id": "e1", "start_time": 1.0},
            {"event_id": "e2", "start_time": 5.0},
        ]
        pairs = derive_pairs_from_events(events, duration=10.0)
        assert "Explain the relationship" in pairs[0]["question"]
        assert "00:01" in pairs[0]["question"]
        assert "00:05" in pairs[0]["question"]

    def test_question_custom_template(self):
        events = [
            {"event_id": "e1", "start_time": 1.0},
            {"event_id": "e2", "start_time": 5.0},
        ]
        pairs = derive_pairs_from_events(
            events,
            duration=10.0,
            question_template="Why did X at {t1} cause Y at {t2}?",
        )
        assert pairs[0]["question"] == "Why did X at 00:01 cause Y at 00:05?"

    def test_max_pairs_cap(self):
        events = [{"event_id": f"e{i}", "start_time": float(i)} for i in range(20)]
        pairs = derive_pairs_from_events(events, duration=25.0, max_pairs=3)
        assert len(pairs) == 3

    def test_default_video_type_attached(self):
        events = [
            {"start_time": 1.0},
            {"start_time": 5.0},
        ]
        pairs = derive_pairs_from_events(events, duration=10.0, default_video_type="anomaly")
        assert all(p["video_type"] == "anomaly" for p in pairs)

    def test_drops_events_with_no_start(self):
        events = [
            {"start_time": 1.0},
            {"event_id": "no_start"},
            {"start_time": 5.0},
        ]
        pairs = derive_pairs_from_events(events, duration=10.0)
        assert len(pairs) == 2

    def test_string_timecodes_accepted(self):
        events = [
            {"start_time": "00:01"},
            {"start_time": "00:05"},
        ]
        pairs = derive_pairs_from_events(events, duration=10.0)
        assert pairs[0]["t1"] == "00:01"
        assert pairs[0]["t2"] == "00:05"

    def test_no_duration_drops_trailing_pair(self):
        """Sanity guard for the pre-existing fall-through behavior: when
        the trailing event has neither a successor nor a clip duration nor
        a usable ``end_time``, the trailing pair is dropped (rather than
        ship a degenerate ``(t1, t1)``)."""
        events = [
            {"start_time": 1.0},
            {"start_time": 5.0},
        ]
        pairs = derive_pairs_from_events(events, duration=None)
        assert len(pairs) == 1
        assert pairs[0]["t1"] == "00:01" and pairs[0]["t2"] == "00:05"

    def test_no_duration_uses_trailing_event_end_time(self):
        """Documented fallback (was missing in the original implementation):
        when ``duration`` is ``None`` but the trailing event carries an
        ``end_time``, the trailing pair must be emitted using that end as
        ``t2``. Dropping it here meant the LAST event in a clip never got
        a causal-linkage pair, which silently halved coverage on
        single-event clips and made the final 'aftermath' frame
        un-questionable.
        """
        events = [
            {"start_time": 1.0},
            {"start_time": 5.0, "end_time": 9.0},
        ]
        pairs = derive_pairs_from_events(events, duration=None)
        assert len(pairs) == 2
        assert pairs[1]["t1"] == "00:05" and pairs[1]["t2"] == "00:09"

    def test_no_duration_legacy_end_key_also_works(self):
        """The events.json contract uses ``end_time``; some upstream
        producers (and our own legacy MSTED segments) emit ``end``.
        The fallback resolves both, mirroring how ``start`` already
        accepts both keys."""
        events = [
            {"start_time": 5.0, "end": 9.0},
        ]
        pairs = derive_pairs_from_events(events, duration=None)
        assert len(pairs) == 1
        assert pairs[0]["t2"] == "00:09"

    def test_no_duration_end_time_not_after_start_drops_pair(self):
        """Defensive: an ``end_time`` <= ``start_time`` is degenerate
        (would emit a ``(t1, t1)`` pair) and the existing 1-ms collapse
        guard already rejects those further down. Make the upstream
        fallback respect the same invariant up-front: an event whose
        own end isn't strictly after its start can't be the seed of a
        useful causal pair."""
        events = [
            {"start_time": 5.0, "end_time": 5.0},
        ]
        pairs = derive_pairs_from_events(events, duration=None)
        assert pairs == []

    def test_duration_still_preferred_over_end_time(self):
        """Behavior preservation: when both ``duration`` and ``end_time``
        are available, the clip duration anchor wins — that matches the
        pre-fix behavior for two-or-more-event clips and avoids surprising
        callers who set ``duration`` explicitly."""
        events = [
            {"start_time": 5.0, "end_time": 7.0},  # end_time < duration
        ]
        pairs = derive_pairs_from_events(events, duration=10.0)
        assert len(pairs) == 1
        assert pairs[0]["t2"] == "00:10"  # duration anchor, not end_time

    def test_empty_events(self):
        assert derive_pairs_from_events([], duration=10.0) == []


# ---------------------------------------------------------------------------
# load_pair_bank
# ---------------------------------------------------------------------------


class TestPairBankInline:
    def test_inline_pairs_with_question(self):
        out = load_pair_bank(
            pairs=[{"t1": 1.0, "t2": 5.0, "question": "Why?"}],
            pair_file=None,
        )
        assert out == [{"t1": "00:01", "t2": "00:05", "question": "Why?"}]

    def test_inline_pairs_default_question(self):
        out = load_pair_bank(
            pairs=[{"t1": 1.0, "t2": 5.0}],
            pair_file=None,
        )
        assert out[0]["question"] == DEFAULT_QUESTION_TEMPLATE.format(t1="00:01", t2="00:05")

    def test_inline_video_type_passthrough(self):
        out = load_pair_bank(
            pairs=[{"t1": 1.0, "t2": 5.0, "video_type": "anomaly"}],
            pair_file=None,
        )
        assert out[0]["video_type"] == "anomaly"

    def test_dedupe(self):
        out = load_pair_bank(
            pairs=[
                {"t1": 1.0, "t2": 5.0, "question": "Q"},
                {"t1": 1.0, "t2": 5.0, "question": "Q"},
            ],
            pair_file=None,
        )
        assert len(out) == 1

    def test_pairs_missing_t1_dropped(self):
        out = load_pair_bank(
            pairs=[{"t2": 5.0}, {"t1": 1.0, "t2": 5.0}],
            pair_file=None,
        )
        assert len(out) == 1


class TestPairBankFile:
    def test_yaml_bare_list(self, tmp_path):
        f = tmp_path / "pairs.yaml"
        f.write_text(
            yaml.safe_dump([{"t1": 1.0, "t2": 5.0}, {"t1": 5.0, "t2": 9.0}]),
            encoding="utf-8",
        )
        out = load_pair_bank(pairs=None, pair_file=f)
        assert len(out) == 2

    def test_yaml_object_with_pairs(self, tmp_path):
        f = tmp_path / "pairs.yaml"
        f.write_text(
            yaml.safe_dump({"pairs": [{"t1": "00:01", "t2": "00:05"}]}),
            encoding="utf-8",
        )
        out = load_pair_bank(pairs=None, pair_file=f)
        assert out[0]["t1"] == "00:01" and out[0]["t2"] == "00:05"

    def test_yaml_object_with_items(self, tmp_path):
        f = tmp_path / "pairs.yaml"
        f.write_text(
            yaml.safe_dump({"items": [{"t1": 1.0, "t2": 5.0}]}),
            encoding="utf-8",
        )
        out = load_pair_bank(pairs=None, pair_file=f)
        assert len(out) == 1

    def test_json_file(self, tmp_path):
        f = tmp_path / "pairs.json"
        f.write_text(
            json.dumps([{"t1": 1.0, "t2": 5.0}]),
            encoding="utf-8",
        )
        out = load_pair_bank(pairs=None, pair_file=f)
        assert len(out) == 1

    def test_inline_plus_file_concatenated_and_deduped(self, tmp_path):
        f = tmp_path / "pairs.yaml"
        f.write_text(
            yaml.safe_dump([{"t1": 5.0, "t2": 9.0}]),
            encoding="utf-8",
        )
        out = load_pair_bank(
            pairs=[{"t1": 1.0, "t2": 5.0}],
            pair_file=f,
        )
        assert len(out) == 2

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pair_bank(pairs=None, pair_file=tmp_path / "missing.yaml")

    def test_unknown_shape_raises(self, tmp_path):
        f = tmp_path / "pairs.yaml"
        f.write_text(yaml.safe_dump({"foo": "bar"}), encoding="utf-8")
        with pytest.raises(ValueError, match="must be a list"):
            load_pair_bank(pairs=None, pair_file=f)

    def test_empty_returns_empty(self):
        assert load_pair_bank(pairs=None, pair_file=None) == []


# ---------------------------------------------------------------------------
# generate_causal_linkages_with_llm
# ---------------------------------------------------------------------------


class TestGenerateCausalLinkages:
    def test_returns_parsed_items(self):
        good = {
            "items": [
                {
                    "t1": "00:01",
                    "t2": "00:05",
                    "question": "Q",
                    "answer": "Because A.",
                    "reasoning": "captions show A.",
                }
            ]
        }
        with patch(
            "reasoning.causal_linkage.llm.call_chat_object_with_structured_fallback",
            side_effect=stub_object_response(good),
        ):
            items = generate_causal_linkages_with_llm(
                pairs=[{"t1": "00:01", "t2": "00:05", "question": "Q"}],
                windows=windows_fixture(),
                scene_description="scene.",
                duration=10.0,
                prompt=fake_prompt(),
                llm_url="http://x",
                llm_model="m",
                api_key="k",
            )
        assert items[0]["t1"] == "00:01"
        assert items[0]["answer"] == "Because A."

    def test_video_type_reattached_from_bank(self):
        good = {
            "items": [
                {
                    "t1": "00:01",
                    "t2": "00:05",
                    "question": "Q",
                    "answer": "Because A.",
                }
            ]
        }
        with patch(
            "reasoning.causal_linkage.llm.call_chat_object_with_structured_fallback",
            side_effect=stub_object_response(good),
        ):
            items = generate_causal_linkages_with_llm(
                pairs=[
                    {
                        "t1": "00:01",
                        "t2": "00:05",
                        "question": "Q",
                        "video_type": "anomaly",
                    }
                ],
                windows=windows_fixture(),
                scene_description=None,
                duration=10.0,
                prompt=fake_prompt(),
                llm_url="http://x",
                llm_model="m",
                api_key="k",
            )
        assert items[0]["video_type"] == "anomaly"

    def test_empty_pairs_raises(self):
        with pytest.raises(CausalLinkageLLMError, match="no usable causal pairs"):
            generate_causal_linkages_with_llm(
                pairs=[],
                windows=windows_fixture(),
                scene_description=None,
                duration=10.0,
                prompt=fake_prompt(),
                llm_url="http://x",
                llm_model="m",
                api_key="k",
            )

    def test_no_usable_windows_raises(self):
        with pytest.raises(CausalLinkageLLMError, match="no usable windows"):
            generate_causal_linkages_with_llm(
                pairs=[{"t1": "00:01", "t2": "00:05", "question": "Q"}],
                windows=[{"foo": "bar"}],
                scene_description=None,
                duration=10.0,
                prompt=fake_prompt(),
                llm_url="http://x",
                llm_model="m",
                api_key="k",
            )

    def test_unparseable_response_raises(self):
        def stub(**kwargs):
            return None, "not json"

        with patch(
            "reasoning.causal_linkage.llm.call_chat_object_with_structured_fallback",
            side_effect=stub,
        ):
            with pytest.raises(CausalLinkageLLMError, match="no parseable JSON"):
                generate_causal_linkages_with_llm(
                    pairs=[{"t1": "00:01", "t2": "00:05", "question": "Q"}],
                    windows=windows_fixture(),
                    scene_description=None,
                    duration=10.0,
                    prompt=fake_prompt(),
                    llm_url="http://x",
                    llm_model="m",
                    api_key="k",
                )

    def test_empty_items_raises(self):
        with patch(
            "reasoning.causal_linkage.llm.call_chat_object_with_structured_fallback",
            side_effect=stub_object_response({"items": []}),
        ):
            with pytest.raises(CausalLinkageLLMError, match="empty 'items'"):
                generate_causal_linkages_with_llm(
                    pairs=[{"t1": "00:01", "t2": "00:05", "question": "Q"}],
                    windows=windows_fixture(),
                    scene_description=None,
                    duration=10.0,
                    prompt=fake_prompt(),
                    llm_url="http://x",
                    llm_model="m",
                    api_key="k",
                )
