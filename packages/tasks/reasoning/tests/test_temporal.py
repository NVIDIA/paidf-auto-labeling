# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# mypy: disable-error-code="no-untyped-def,unused-ignore"

from __future__ import annotations

import re

import pytest
from reasoning.common import DAFT_VERSION, DaftConvertError, SceneContext
from reasoning.temporal.converter import (
    DEFAULT_QUESTION_TEMPLATE,
    to_daft_temporal_description,
)

VIDEO_CTX = SceneContext(media_id="clip_42", iso_date="2026-04-20")
IMAGE_CTX = SceneContext(media_id="frame_001", is_image=True, iso_date="2026-04-20")

# Schema regex from temporal_description.schema.json — every emitted
# timecode must match this pattern.
TIMECODE_RE = re.compile(r"^(\d{2}:)?\d{2}:\d{2}(\.\d+)?$")


def pl_window(**overrides) -> dict:
    """Window shape PL's window_vlm_llm runner writes to sidecars/metadata.json.

    Same fixture shape as test_chunks.py — the converter accepts the same
    input so callers can re-use one window list for both task-side
    temporal_description.json and contextual-side chunks.json.
    """
    base = {
        "start_s": 0.0,
        "end_s": 4.0,
        "start_frame": 0,
        "end_frame": 119,
        "caption": "Traffic flows normally.",
        "enhanced_caption": "A white sedan and a gray sedan drive in adjacent lanes.",
    }
    base.update(overrides)
    return base


class TestEmpty:
    def test_empty_iter_returns_none(self):
        assert to_daft_temporal_description([], ctx=VIDEO_CTX) is None

    def test_image_ctx_rejected(self):
        with pytest.raises(DaftConvertError, match="image"):
            to_daft_temporal_description([pl_window()], ctx=IMAGE_CTX)


class TestEnvelope:
    def test_envelope_shape(self):
        out = to_daft_temporal_description([pl_window()], ctx=VIDEO_CTX)
        assert out["version"] == DAFT_VERSION
        assert out["metadata"]["type"] == "temporal_description"
        assert out["metadata"]["date"] == "2026-04-20"
        assert "items" in out
        assert isinstance(out["items"], list) and len(out["items"]) == 1

    def test_no_top_level_video_id(self):
        # Schema's additionalProperties:false rejects any top-level scene id.
        out = to_daft_temporal_description([pl_window()], ctx=VIDEO_CTX)
        assert "video_id" not in out
        assert "image_id" not in out

    def test_metadata_threads_optional_fields(self):
        ctx = SceneContext(media_id="clip", license_str="CC-BY-4.0", tags=("urban",))
        out = to_daft_temporal_description([pl_window()], ctx=ctx)
        assert out["metadata"]["license"] == "CC-BY-4.0"
        assert out["metadata"]["tags"] == ["urban"]


class TestItemShape:
    def test_minimal_item_uses_default_question_template(self):
        # The default template is the schema example — use-case agnostic.
        out = to_daft_temporal_description([pl_window()], ctx=VIDEO_CTX)
        [item] = out["items"]
        assert item["video_id"] == "clip_42"
        assert item["t1"] == "00:00"
        assert item["t2"] == "00:04"
        assert item["question"] == DEFAULT_QUESTION_TEMPLATE.format(t1="00:00", t2="00:04")
        # Default description preference: "description" first; this fixture
        # only sets caption + enhanced_caption, so the answer comes from
        # caption (the second key in the default tuple).
        assert item["answer"] == "Traffic flows normally."
        assert "video_type" not in item
        assert "reasoning" not in item

    def test_description_keys_override_picks_enhanced_caption(self):
        # Use-case agnostic knob: caller decides which key is the "best" answer.
        out = to_daft_temporal_description(
            [pl_window()],
            ctx=VIDEO_CTX,
            description_keys=("enhanced_caption", "caption"),
        )
        [item] = out["items"]
        assert item["answer"] == "A white sedan and a gray sedan drive in adjacent lanes."

    def test_custom_question_template_is_formatted_with_t1_t2(self):
        # Caller phrases the question for their domain. Both placeholders
        # must be filled with the DAFT timecode strings (not raw seconds).
        template = "Describe the safety-relevant activity from {t1} to {t2}."
        out = to_daft_temporal_description(
            [pl_window(start_s=12.5, end_s=18.0)],
            ctx=VIDEO_CTX,
            question_template=template,
        )
        [item] = out["items"]
        assert item["t1"] == "00:12.500"
        assert item["t2"] == "00:18"
        assert item["question"] == "Describe the safety-relevant activity from 00:12.500 to 00:18."

    def test_question_template_unknown_placeholder_raises(self):
        # Catches typos in caller-supplied templates early instead of writing
        # a half-formatted string to disk.
        with pytest.raises(DaftConvertError, match="placeholder"):
            to_daft_temporal_description(
                [pl_window()],
                ctx=VIDEO_CTX,
                question_template="What about {missing_key}?",
            )

    def test_video_type_emitted_when_supplied(self):
        out = to_daft_temporal_description(
            [pl_window()],
            ctx=VIDEO_CTX,
            video_type="anomaly",
        )
        [item] = out["items"]
        assert item["video_type"] == "anomaly"

    def test_video_type_invalid_value_rejected(self):
        with pytest.raises(DaftConvertError, match="video_type"):
            to_daft_temporal_description(
                [pl_window()],
                ctx=VIDEO_CTX,
                video_type="weird",  # type: ignore[arg-type]
            )

    def test_reasoning_emitted_when_key_present(self):
        # Use-case agnostic: caller picks which window field to thread as
        # reasoning. This lets future LLM-enrichment passes attach traces
        # without the converter knowing the field name in advance.
        out = to_daft_temporal_description(
            [pl_window(reasoning_trace="The vehicles maintain steady speed.")],
            ctx=VIDEO_CTX,
            reasoning_key="reasoning_trace",
        )
        [item] = out["items"]
        assert item["reasoning"] == "The vehicles maintain steady speed."

    def test_reasoning_key_absent_means_no_reasoning(self):
        # Even when the window has a "reasoning" key, the converter ignores
        # it unless the caller explicitly opts in via reasoning_key. Keeps
        # output deterministic and avoids leaking unrelated fields.
        out = to_daft_temporal_description(
            [pl_window(reasoning="Should be ignored.")],
            ctx=VIDEO_CTX,
        )
        [item] = out["items"]
        assert "reasoning" not in item

    def test_reasoning_blank_value_skipped(self):
        out = to_daft_temporal_description(
            [pl_window(reasoning_trace="   ")],
            ctx=VIDEO_CTX,
            reasoning_key="reasoning_trace",
        )
        [item] = out["items"]
        assert "reasoning" not in item


class TestKeyAliases:
    def test_start_end_aliases_accepted(self):
        # Match the chunks-converter alias families; PL window writers vary.
        win = {"start": 1.0, "end": 5.0, "description": "x"}
        out = to_daft_temporal_description([win], ctx=VIDEO_CTX)
        [item] = out["items"]
        assert item["t1"] == "00:01"
        assert item["t2"] == "00:05"

    def test_summary_alias_accepted_as_answer(self):
        win = {"start_s": 0.0, "end_s": 1.0, "summary": "A short blurb."}
        out = to_daft_temporal_description([win], ctx=VIDEO_CTX)
        [item] = out["items"]
        assert item["answer"] == "A short blurb."


class TestTimecodes:
    def test_fractional_seconds_emitted_with_milliseconds(self):
        out = to_daft_temporal_description(
            [pl_window(start_s=0.250, end_s=4.500)],
            ctx=VIDEO_CTX,
        )
        [item] = out["items"]
        assert item["t1"] == "00:00.250"
        assert item["t2"] == "00:04.500"
        for tc in (item["t1"], item["t2"]):
            assert TIMECODE_RE.match(tc)

    def test_clamp_to_duration(self):
        out = to_daft_temporal_description(
            [pl_window(start_s=58.0, end_s=65.0)],
            ctx=VIDEO_CTX,
            duration=60.0,
        )
        [item] = out["items"]
        assert item["t1"] == "00:58"
        assert item["t2"] == "01:00"

    def test_long_video_uses_hh_mm_ss(self):
        out = to_daft_temporal_description(
            [pl_window(start_s=3600.0, end_s=3700.0)],
            ctx=VIDEO_CTX,
        )
        [item] = out["items"]
        assert item["t1"] == "01:00:00"
        assert item["t2"] == "01:01:40"


class TestErrors:
    def test_window_must_be_dict(self):
        with pytest.raises(DaftConvertError, match="dict"):
            to_daft_temporal_description(["not a window"], ctx=VIDEO_CTX)  # type: ignore[list-item]

    def test_missing_start_raises(self):
        with pytest.raises(DaftConvertError, match="start"):
            to_daft_temporal_description([{"end_s": 1.0, "caption": "x"}], ctx=VIDEO_CTX)

    def test_missing_end_raises(self):
        with pytest.raises(DaftConvertError, match="end"):
            to_daft_temporal_description([{"start_s": 0.0, "caption": "x"}], ctx=VIDEO_CTX)

    def test_missing_answer_raises(self):
        with pytest.raises(DaftConvertError, match="answer"):
            to_daft_temporal_description([{"start_s": 0.0, "end_s": 1.0}], ctx=VIDEO_CTX)

    def test_non_numeric_start_raises(self):
        with pytest.raises(DaftConvertError, match="number"):
            to_daft_temporal_description(
                [{"start_s": "zero", "end_s": 1.0, "caption": "x"}], ctx=VIDEO_CTX
            )

    def test_bool_is_not_a_valid_number(self):
        # Same trap as elsewhere: bool is a subclass of int in Python.
        with pytest.raises(DaftConvertError, match="number"):
            to_daft_temporal_description(
                [{"start_s": True, "end_s": 1.0, "caption": "x"}], ctx=VIDEO_CTX
            )

    def test_end_before_start_raises(self):
        with pytest.raises(DaftConvertError, match="end .* < start"):
            to_daft_temporal_description([pl_window(start_s=5.0, end_s=2.0)], ctx=VIDEO_CTX)


class TestMultipleItems:
    def test_items_preserve_order_and_per_window_data(self):
        wins = [
            pl_window(start_s=0.0, end_s=2.0, caption="A"),
            pl_window(start_s=2.0, end_s=4.0, caption="B"),
            pl_window(start_s=4.0, end_s=6.0, caption="C"),
        ]
        out = to_daft_temporal_description(wins, ctx=VIDEO_CTX)
        assert [it["answer"] for it in out["items"]] == ["A", "B", "C"]
        assert [it["t1"] for it in out["items"]] == ["00:00", "00:02", "00:04"]
        assert [it["t2"] for it in out["items"]] == ["00:02", "00:04", "00:06"]
        for it in out["items"]:
            assert it["video_id"] == "clip_42"
