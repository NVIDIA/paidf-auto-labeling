# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re

import pytest
from daft_export.common import DAFT_VERSION, DaftConvertError
from daft_export.task import to_daft_tasks

# Answer regexes from DAFT schemas; every produced answer must match.
MCQ_ANSWER_RE = re.compile(r"^[A-Za-z]$")
BCQ_ANSWER_RE = re.compile(r"^(Yes|No)$")

VIDEO_ID = "main"


def rule_based_item() -> dict:
    # Older/internal format accepted by the DAFT converter: "A. <text>" everywhere.
    return {
        "id": "accident_type",
        "question": "What type of accident?",
        "options": ["A. Rollover", "B. Head-on", "C. Rear-end"],
        "answer": "A. Rollover",
    }


def bank_item() -> dict:
    # Format emitted by LLM-bank runners after filter_mcq_items_strict: bare
    # bank strings in both options and answer.
    return {
        "id": "weather_q",
        "question": "What is the weather?",
        "options": ["Sunny", "Cloudy", "Rainy"],
        "answer": "Cloudy",
    }


def yes_no_item(answer: str = "Yes") -> dict:
    return {
        "id": "collision_q",
        "question": "Does a collision occur?",
        "options": ["Yes", "No"],
        "answer": answer,
    }


class TestEmpty:
    def test_empty_input_returns_none_none(self):
        assert to_daft_tasks([], video_id=VIDEO_ID) == (None, None, None)


class TestMcqConversion:
    def test_rule_based_prefix_stripped(self):
        mcq, bcq, open_qa = to_daft_tasks([rule_based_item()], video_id=VIDEO_ID)
        assert bcq is None
        assert open_qa is None
        [item] = mcq["items"]
        assert item["answer"] == "A"
        assert item["options"] == {"A": "Rollover", "B": "Head-on", "C": "Rear-end"}

    def test_bank_style_routed_by_value(self):
        # "Cloudy" isn't a letter and has no prefix, so the converter must
        # find it by value lookup in the normalized options dict.
        mcq, bcq, open_qa = to_daft_tasks([bank_item()], video_id=VIDEO_ID)
        assert bcq is None
        assert open_qa is None
        [item] = mcq["items"]
        assert item["answer"] == "B"
        assert item["options"] == {"A": "Sunny", "B": "Cloudy", "C": "Rainy"}

    def test_item_shape(self):
        mcq, _, _ = to_daft_tasks([rule_based_item()], video_id=VIDEO_ID)
        [item] = mcq["items"]
        assert set(item.keys()) == {"video_id", "question", "answer", "options"}
        assert item["video_id"] == VIDEO_ID
        assert item["question"] == "What type of accident?"
        assert MCQ_ANSWER_RE.fullmatch(item["answer"])

    def test_extra_fields_stripped(self):
        # DAFT uses additionalProperties: false; converter must not leak auto-labeling-internal fields.
        item = rule_based_item() | {"_debug": "trace", "confidence": 0.9}
        mcq, _, _ = to_daft_tasks([item], video_id=VIDEO_ID)
        [out] = mcq["items"]
        assert "_debug" not in out
        assert "confidence" not in out

    def test_many_options_all_letters_assigned(self):
        # 30 options exercises the A-Z to a-z wraparound in the alphabet table.
        opts = [f"opt{i}" for i in range(30)]
        item = {"id": "q", "question": "?", "options": opts, "answer": "opt27"}
        mcq, _, _ = to_daft_tasks([item], video_id=VIDEO_ID)
        [out] = mcq["items"]
        assert out["answer"] == "b"  # index 27 = A-Z (26) then 'a', 'b'
        assert out["options"]["b"] == "opt27"

    def test_envelope(self):
        mcq, _, _ = to_daft_tasks([rule_based_item()], video_id=VIDEO_ID)
        assert mcq["version"] == DAFT_VERSION
        assert mcq["metadata"]["type"] == "mcq"
        assert len(mcq["items"]) == 1


class TestBcqConversion:
    @pytest.mark.parametrize("answer", ["Yes", "No"])
    def test_yes_no_routed_to_bcq(self, answer):
        mcq, bcq, open_qa = to_daft_tasks([yes_no_item(answer)], video_id=VIDEO_ID)
        assert mcq is None
        assert open_qa is None
        [item] = bcq["items"]
        assert item["answer"] == answer
        assert BCQ_ANSWER_RE.fullmatch(item["answer"])

    def test_reversed_yes_no_options_route_to_bcq(self):
        item = {
            "id": "collision_q",
            "question": "Does a collision occur?",
            "options": ["No", "Yes"],
            "answer": "No",
        }
        mcq, bcq, open_qa = to_daft_tasks([item], video_id=VIDEO_ID)
        assert mcq is None
        assert open_qa is None
        [out] = bcq["items"]
        assert out["answer"] == "No"

    def test_bcq_has_no_options_or_item_metadata(self):
        # BCQ schema forbids both; converter must strip them.
        _, bcq, _ = to_daft_tasks([yes_no_item()], video_id=VIDEO_ID)
        [item] = bcq["items"]
        assert "options" not in item
        assert "item_metadata" not in item
        assert set(item.keys()) == {"video_id", "question", "answer"}

    def test_missing_options_yes_no_answer_routes_to_open_qa(self):
        item = {"id": "q", "question": "?", "answer": "Yes"}
        mcq, bcq, open_qa = to_daft_tasks([item], video_id=VIDEO_ID)
        assert mcq is None
        assert bcq is None
        assert open_qa["metadata"]["type"] == "open_qa"
        assert open_qa["items"] == [{"video_id": VIDEO_ID, "question": "?", "answer": "Yes"}]

    def test_envelope(self):
        _, bcq, _ = to_daft_tasks([yes_no_item()], video_id=VIDEO_ID)
        assert bcq["version"] == DAFT_VERSION
        assert bcq["metadata"]["type"] == "bcq"


class TestSplitAndRoute:
    def test_mixed_items_split(self):
        items = [yes_no_item("Yes"), rule_based_item(), yes_no_item("No"), bank_item()]
        mcq, bcq, open_qa = to_daft_tasks(items, video_id=VIDEO_ID)
        assert len(mcq["items"]) == 2
        assert len(bcq["items"]) == 2
        assert open_qa is None

    def test_yes_no_answer_with_non_yes_no_options_routes_to_mcq(self):
        # Guard against the split rule over-matching: "Yes" as an answer to a
        # real multi-choice question stays in MCQ.
        item = {
            "id": "q",
            "question": "?",
            "options": ["Yes", "No", "Maybe"],
            "answer": "Yes",
        }
        mcq, bcq, open_qa = to_daft_tasks([item], video_id=VIDEO_ID)
        assert bcq is None
        assert open_qa is None
        [out] = mcq["items"]
        assert out["answer"] == "A"

    def test_yes_no_unknown_options_route_to_mcq(self):
        item = {
            "id": "q",
            "question": "?",
            "options": ["No", "Yes", "unknown"],
            "answer": "unknown",
        }
        mcq, bcq, open_qa = to_daft_tasks([item], video_id=VIDEO_ID)
        assert bcq is None
        assert open_qa is None
        [out] = mcq["items"]
        assert out["answer"] == "C"

    def test_duplicate_yes_no_options_dedupe_and_route_to_bcq(self):
        item = {
            "id": "q",
            "question": "?",
            "options": ["Yes", "No", "Yes"],
            "answer": "Yes",
        }
        mcq, bcq, open_qa = to_daft_tasks([item], video_id=VIDEO_ID)
        assert mcq is None
        assert open_qa is None
        [out] = bcq["items"]
        assert out["answer"] == "Yes"

    def test_prefixed_yes_no_options_route_to_mcq(self):
        item = {
            "id": "q",
            "question": "?",
            "options": ["A. Yes", "B. No"],
            "answer": "A. Yes",
        }
        mcq, bcq, open_qa = to_daft_tasks([item], video_id=VIDEO_ID)
        assert bcq is None
        assert open_qa is None
        [out] = mcq["items"]
        assert out["answer"] == "A"

    def test_duplicate_mcq_options_are_deduped(self):
        item = {
            "id": "q",
            "question": "?",
            "options": ["A. Car", "B. Truck", "A. Car"],
            "answer": "B. Truck",
        }
        mcq, bcq, open_qa = to_daft_tasks([item], video_id=VIDEO_ID)
        assert bcq is None
        assert open_qa is None
        [out] = mcq["items"]
        assert out["answer"] == "B"
        assert out["options"] == {"A": "Car", "B": "Truck"}

    def test_prefixed_duplicate_mcq_options_dedupe_by_display_value(self):
        item = {
            "id": "q",
            "question": "?",
            "options": ["A. b", "B. b", "C. c"],
            "answer": "A. b",
        }
        mcq, bcq, open_qa = to_daft_tasks([item], video_id=VIDEO_ID)
        assert bcq is None
        assert open_qa is None
        [out] = mcq["items"]
        assert out["answer"] == "A"
        assert out["options"] == {"A": "b", "B": "c"}

    def test_removed_prefixed_duplicate_mcq_answer_raises(self):
        item = {
            "id": "q",
            "question": "?",
            "options": ["A. b", "B. b", "C. c"],
            "answer": "B. b",
        }
        with pytest.raises(DaftConvertError, match="not in options"):
            to_daft_tasks([item], video_id=VIDEO_ID)

    def test_prefixed_duplicate_mcq_letter_answer_uses_current_letters(self):
        item = {
            "id": "q",
            "question": "?",
            "options": ["A. b", "B. b", "C. c"],
            "answer": "B",
        }
        mcq, bcq, open_qa = to_daft_tasks([item], video_id=VIDEO_ID)
        assert bcq is None
        assert open_qa is None
        [out] = mcq["items"]
        assert out["answer"] == "B"
        assert out["options"] == {"A": "b", "B": "c"}

    def test_compact_prefixed_duplicate_mcq_options_are_deduped(self):
        item = {
            "id": "q",
            "question": "?",
            "options": ["A.b", "B.c", "A.b"],
            "answer": "A.b",
        }
        mcq, bcq, open_qa = to_daft_tasks([item], video_id=VIDEO_ID)
        assert bcq is None
        assert open_qa is None
        [out] = mcq["items"]
        assert out["answer"] == "A"
        assert out["options"] == {"A": "b", "B": "c"}

    def test_duplicate_only_options_raise_after_dedupe(self):
        item = {"id": "q", "question": "?", "options": ["only", "only"], "answer": "only"}
        with pytest.raises(DaftConvertError, match="unique options"):
            to_daft_tasks([item], video_id=VIDEO_ID)


class TestReasoningPassthrough:
    """DAFT v3.0 MCQ and BCQ both allow an optional ``reasoning`` field. When
    ``vlm_verify`` attaches a ``reasoning_trace`` to an internal item, the
    converter surfaces it there; otherwise the key is omitted (empty string
    would fail ``minLength`` in stricter consumers)."""

    def test_mcq_reasoning_trace_passed_through(self):
        item = rule_based_item() | {"reasoning_trace": "The vehicle flips onto its roof at t=2s."}
        mcq, _, _ = to_daft_tasks([item], video_id=VIDEO_ID)
        [out] = mcq["items"]
        assert out["reasoning"] == "The vehicle flips onto its roof at t=2s."

    def test_bcq_reasoning_trace_passed_through(self):
        item = yes_no_item("Yes") | {"reasoning_trace": "Two vehicles make contact at t=3s."}
        _, bcq, _ = to_daft_tasks([item], video_id=VIDEO_ID)
        [out] = bcq["items"]
        assert out["reasoning"] == "Two vehicles make contact at t=3s."

    def test_reasoning_key_used_when_present(self):
        item = rule_based_item() | {"reasoning": "prior trace"}
        mcq, _, _ = to_daft_tasks([item], video_id=VIDEO_ID)
        [out] = mcq["items"]
        assert out["reasoning"] == "prior trace"

    def test_reasoning_trace_preferred_over_reasoning(self):
        # If both are present, prefer the internal name (vlm_verify's output).
        item = rule_based_item() | {"reasoning_trace": "fresh trace", "reasoning": "secondary trace"}
        mcq, _, _ = to_daft_tasks([item], video_id=VIDEO_ID)
        [out] = mcq["items"]
        assert out["reasoning"] == "fresh trace"

    def test_empty_reasoning_omits_key(self):
        # Empty/whitespace-only strings must not be emitted.
        for val in ("", "   ", "\n\t"):
            item = rule_based_item() | {"reasoning_trace": val}
            mcq, _, _ = to_daft_tasks([item], video_id=VIDEO_ID)
            [out] = mcq["items"]
            assert "reasoning" not in out

    def test_absent_reasoning_omits_key(self):
        mcq, _, _ = to_daft_tasks([rule_based_item()], video_id=VIDEO_ID)
        [out] = mcq["items"]
        assert "reasoning" not in out
        _, bcq, _ = to_daft_tasks([yes_no_item()], video_id=VIDEO_ID)
        [out] = bcq["items"]
        assert "reasoning" not in out

    def test_non_string_reasoning_ignored(self):
        # Defensive: non-strings upstream are silently dropped rather than
        # propagated into DAFT where they'd fail schema validation.
        item = rule_based_item() | {"reasoning_trace": 42}
        mcq, _, _ = to_daft_tasks([item], video_id=VIDEO_ID)
        [out] = mcq["items"]
        assert "reasoning" not in out

    def test_open_qa_reasoning_trace_passed_through(self):
        item = {
            "id": "q",
            "question": "Describe the traffic pattern.",
            "options": [],
            "answer": "Traffic flows north-south with a protected left-turn lane.",
            "reasoning_trace": "Lane markings and vehicle movement indicate the pattern.",
        }
        _, _, open_qa = to_daft_tasks([item], video_id=VIDEO_ID)
        [out] = open_qa["items"]
        assert out["reasoning"] == "Lane markings and vehicle movement indicate the pattern."


class TestErrors:
    def test_answer_not_in_options(self):
        item = {"id": "q", "question": "?", "options": ["A", "B"], "answer": "C"}
        with pytest.raises(DaftConvertError, match="not in options"):
            to_daft_tasks([item], video_id=VIDEO_ID)

    def test_missing_options_routes_to_open_qa(self):
        item = {"id": "q", "question": "?", "answer": "free text"}
        mcq, bcq, open_qa = to_daft_tasks([item], video_id=VIDEO_ID)
        assert mcq is None
        assert bcq is None
        assert open_qa["metadata"]["type"] == "open_qa"
        assert open_qa["items"] == [{"video_id": VIDEO_ID, "question": "?", "answer": "free text"}]

    def test_empty_options_routes_to_open_qa(self):
        item = {"id": "q", "question": "?", "options": [], "answer": "free text"}
        mcq, bcq, open_qa = to_daft_tasks([item], video_id=VIDEO_ID)
        assert mcq is None
        assert bcq is None
        assert open_qa["metadata"]["type"] == "open_qa"
        assert open_qa["items"] == [{"video_id": VIDEO_ID, "question": "?", "answer": "free text"}]

    def test_whitespace_only_options_route_to_open_qa(self):
        item = {"id": "q", "question": "?", "options": ["", "   ", "\n\t"], "answer": "free text"}
        mcq, bcq, open_qa = to_daft_tasks([item], video_id=VIDEO_ID)
        assert mcq is None
        assert bcq is None
        assert open_qa["metadata"]["type"] == "open_qa"
        assert open_qa["items"] == [{"video_id": VIDEO_ID, "question": "?", "answer": "free text"}]

    def test_whitespace_around_bcq_options_is_ignored(self):
        item = {"id": "q", "question": "?", "options": [" Yes ", "No", " "], "answer": "Yes"}
        mcq, bcq, open_qa = to_daft_tasks([item], video_id=VIDEO_ID)
        assert mcq is None
        assert open_qa is None
        assert bcq["metadata"]["type"] == "bcq"
        assert bcq["items"] == [{"video_id": VIDEO_ID, "question": "?", "answer": "Yes"}]

    def test_empty_options_yes_no_answer_still_routes_to_open_qa(self):
        item = {"id": "q", "question": "Explain whether police are present.", "options": [], "answer": "No"}
        mcq, bcq, open_qa = to_daft_tasks([item], video_id=VIDEO_ID)
        assert mcq is None
        assert bcq is None
        assert open_qa["metadata"]["type"] == "open_qa"
        assert open_qa["items"] == [
            {"video_id": VIDEO_ID, "question": "Explain whether police are present.", "answer": "No"}
        ]

    def test_empty_options_with_empty_answer_raises(self):
        item = {"id": "q", "question": "?", "options": [], "answer": "   "}
        with pytest.raises(DaftConvertError, match="no options"):
            to_daft_tasks([item], video_id=VIDEO_ID)

    def test_too_few_options(self):
        item = {"id": "q", "question": "?", "options": ["only"], "answer": "only"}
        with pytest.raises(DaftConvertError, match="minItems 2"):
            to_daft_tasks([item], video_id=VIDEO_ID)

    def test_too_many_options(self):
        opts = [f"opt{i}" for i in range(53)]
        item = {"id": "q", "question": "?", "options": opts, "answer": "opt0"}
        with pytest.raises(DaftConvertError, match="caps this at 52"):
            to_daft_tasks([item], video_id=VIDEO_ID)

    @pytest.mark.parametrize("missing", ["question", "answer"])
    def test_missing_required_field_raises(self, missing):
        item = rule_based_item()
        del item[missing]
        with pytest.raises(DaftConvertError, match=missing):
            to_daft_tasks([item], video_id=VIDEO_ID)
