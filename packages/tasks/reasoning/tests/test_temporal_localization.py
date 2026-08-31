# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# mypy: disable-error-code="no-untyped-def,unused-ignore"

from __future__ import annotations

import re

import pytest
from reasoning.common import DAFT_VERSION, DaftConvertError, SceneContext
from reasoning.temporal_localization.converter import to_daft_temporal_localization

VIDEO_CTX = SceneContext(media_id="clip_42", iso_date="2026-04-20")
IMAGE_CTX = SceneContext(media_id="frame_001", is_image=True, iso_date="2026-04-20")
TIMECODE_RE = re.compile(r"^(\d{2}:)?\d{2}:\d{2}(\.\d+)?$")


def base_item(**overrides) -> dict:
    item = {
        "question": "When does the car turn right?",
        "answer": {"start": 4.0, "end": 7.0},
    }
    item.update(overrides)
    return item


class TestEmpty:
    def test_empty_iter_returns_none(self):
        assert to_daft_temporal_localization([], ctx=VIDEO_CTX) is None

    def test_image_ctx_rejected(self):
        with pytest.raises(DaftConvertError, match="image"):
            to_daft_temporal_localization([base_item()], ctx=IMAGE_CTX, duration=10.0)


class TestEnvelope:
    def test_envelope_shape(self):
        out = to_daft_temporal_localization([base_item()], ctx=VIDEO_CTX, duration=10.0)
        assert out["version"] == DAFT_VERSION
        assert out["metadata"]["type"] == "temporal_localization"
        assert out["metadata"]["date"] == "2026-04-20"
        assert isinstance(out["items"], list) and len(out["items"]) == 1

    def test_no_top_level_video_id(self):
        # Schema's additionalProperties:false rejects top-level scene id;
        # video_id is per-item only.
        out = to_daft_temporal_localization([base_item()], ctx=VIDEO_CTX, duration=10.0)
        assert "video_id" not in out
        assert "image_id" not in out

    def test_metadata_threads_optional_fields(self):
        ctx = SceneContext(media_id="clip", license_str="CC-BY-4.0", tags=("urban",))
        out = to_daft_temporal_localization([base_item()], ctx=ctx, duration=10.0)
        assert out["metadata"]["license"] == "CC-BY-4.0"
        assert out["metadata"]["tags"] == ["urban"]


class TestItemShape:
    def test_minimal_item(self):
        out = to_daft_temporal_localization([base_item()], ctx=VIDEO_CTX, duration=10.0)
        [item] = out["items"]
        assert item["video_id"] == "clip_42"
        assert item["t1"] == "00:00"  # defaulted from missing t1
        assert item["t2"] == "00:10"  # defaulted from duration
        assert item["question"] == "When does the car turn right?"
        assert item["answer"] == {"start": "00:04", "end": "00:07"}
        assert "video_type" not in item
        assert "reasoning" not in item

    def test_explicit_t1_t2_preserved(self):
        out = to_daft_temporal_localization(
            [base_item(t1=2.0, t2=8.0)], ctx=VIDEO_CTX, duration=10.0
        )
        [item] = out["items"]
        assert item["t1"] == "00:02"
        assert item["t2"] == "00:08"

    def test_t2_missing_without_duration_raises(self):
        with pytest.raises(DaftConvertError, match="missing t2 and no fallback duration"):
            to_daft_temporal_localization([base_item()], ctx=VIDEO_CTX)

    def test_string_timecodes_accepted(self):
        out = to_daft_temporal_localization(
            [
                {
                    "question": "Q?",
                    "answer": {"start": "00:04.500", "end": "00:07"},
                    "t1": "00:00",
                    "t2": "00:10",
                }
            ],
            ctx=VIDEO_CTX,
        )
        item = out["items"][0]
        assert item["answer"]["start"] == "00:04.500"
        assert item["answer"]["end"] == "00:07"

    def test_invalid_string_timecode_rejected(self):
        with pytest.raises(DaftConvertError, match="timecode"):
            to_daft_temporal_localization(
                [
                    {
                        "question": "Q?",
                        "answer": {"start": "not-a-timecode", "end": "00:07"},
                    }
                ],
                ctx=VIDEO_CTX,
                duration=10.0,
            )

    def test_negative_seconds_rejected(self):
        with pytest.raises(DaftConvertError, match=">= 0"):
            to_daft_temporal_localization(
                [
                    {
                        "question": "Q?",
                        "answer": {"start": -1.0, "end": 5.0},
                    }
                ],
                ctx=VIDEO_CTX,
                duration=10.0,
            )

    def test_negative_duration_rejected(self):
        with pytest.raises(DaftConvertError, match="duration"):
            to_daft_temporal_localization(
                [
                    {
                        "question": "Q?",
                        "answer": {"start": 1.0, "end": 5.0},
                    }
                ],
                ctx=VIDEO_CTX,
                duration=-1.0,
            )

    def test_bool_rejected_as_timecode(self):
        with pytest.raises(DaftConvertError, match="number or timecode"):
            to_daft_temporal_localization(
                [
                    {
                        "question": "Q?",
                        "answer": {"start": True, "end": 5.0},
                    }
                ],
                ctx=VIDEO_CTX,
                duration=10.0,
            )

    def test_clamp_to_duration(self):
        out = to_daft_temporal_localization(
            [
                {
                    "question": "Q?",
                    "answer": {"start": 58.0, "end": 65.0},
                    "t1": 0.0,
                    "t2": 70.0,
                }
            ],
            ctx=VIDEO_CTX,
            duration=60.0,
        )
        item = out["items"][0]
        assert item["t2"] == "01:00"
        assert item["answer"]["end"] == "01:00"

    def test_answer_end_before_start_rejected(self):
        with pytest.raises(DaftConvertError, match="answer.end .* < answer.start"):
            to_daft_temporal_localization(
                [
                    {
                        "question": "Q?",
                        "answer": {"start": 5.0, "end": 2.0},
                    }
                ],
                ctx=VIDEO_CTX,
                duration=10.0,
            )

    def test_t2_before_t1_rejected(self):
        with pytest.raises(DaftConvertError, match="t2 .* < t1"):
            to_daft_temporal_localization(
                [
                    {
                        "question": "Q?",
                        "answer": {"start": 0.0, "end": 1.0},
                        "t1": 5.0,
                        "t2": 2.0,
                    }
                ],
                ctx=VIDEO_CTX,
                duration=10.0,
            )

    def test_video_type_emitted_when_supplied(self):
        out = to_daft_temporal_localization(
            [base_item(video_type="anomaly")], ctx=VIDEO_CTX, duration=10.0
        )
        assert out["items"][0]["video_type"] == "anomaly"

    def test_video_type_invalid_value_rejected(self):
        with pytest.raises(DaftConvertError, match="video_type"):
            to_daft_temporal_localization(
                [base_item(video_type="weird")], ctx=VIDEO_CTX, duration=10.0
            )

    def test_reasoning_kept_when_present(self):
        out = to_daft_temporal_localization(
            [base_item(reasoning="The captions show the right turn at 00:04-00:07.")],
            ctx=VIDEO_CTX,
            duration=10.0,
        )
        assert out["items"][0]["reasoning"] == "The captions show the right turn at 00:04-00:07."

    def test_blank_reasoning_dropped(self):
        out = to_daft_temporal_localization(
            [base_item(reasoning="   ")], ctx=VIDEO_CTX, duration=10.0
        )
        assert "reasoning" not in out["items"][0]

    def test_reasoning_strips_whitespace(self):
        out = to_daft_temporal_localization(
            [base_item(reasoning="  trimmed  ")],
            ctx=VIDEO_CTX,
            duration=10.0,
        )
        assert out["items"][0]["reasoning"] == "trimmed"

    def test_non_string_reasoning_rejected(self):
        with pytest.raises(DaftConvertError, match="reasoning"):
            to_daft_temporal_localization([base_item(reasoning=42)], ctx=VIDEO_CTX, duration=10.0)


class TestQuestionValidation:
    def test_blank_question_rejected(self):
        with pytest.raises(DaftConvertError, match="question"):
            to_daft_temporal_localization([base_item(question="   ")], ctx=VIDEO_CTX, duration=10.0)

    def test_non_string_question_rejected(self):
        with pytest.raises(DaftConvertError, match="question"):
            to_daft_temporal_localization([base_item(question=42)], ctx=VIDEO_CTX, duration=10.0)


class TestAnswerValidation:
    def test_non_dict_answer_rejected(self):
        with pytest.raises(DaftConvertError, match="answer must be a dict"):
            to_daft_temporal_localization(
                [{"question": "Q?", "answer": "not a dict"}],
                ctx=VIDEO_CTX,
                duration=10.0,
            )

    def test_missing_start_rejected(self):
        with pytest.raises(DaftConvertError, match="answer.start"):
            to_daft_temporal_localization(
                [{"question": "Q?", "answer": {"end": 5.0}}],
                ctx=VIDEO_CTX,
                duration=10.0,
            )

    def test_missing_end_rejected(self):
        with pytest.raises(DaftConvertError, match="answer.end"):
            to_daft_temporal_localization(
                [{"question": "Q?", "answer": {"start": 0.0}}],
                ctx=VIDEO_CTX,
                duration=10.0,
            )


class TestKeyOrderingStability:
    def test_item_keys_stable(self):
        out = to_daft_temporal_localization(
            [base_item(video_type="normal", reasoning="trace")],
            ctx=VIDEO_CTX,
            duration=10.0,
        )
        item = out["items"][0]
        # Required first, optionals appended in declared order.
        keys = list(item.keys())
        assert keys[:5] == ["video_id", "t1", "t2", "question", "answer"]
        assert "video_type" in keys
        assert "reasoning" in keys


class TestTimecodeOutput:
    def test_all_timecodes_match_schema_regex(self):
        out = to_daft_temporal_localization(
            [base_item(t1=0.0, t2=10.0)], ctx=VIDEO_CTX, duration=10.0
        )
        item = out["items"][0]
        assert TIMECODE_RE.match(item["t1"])
        assert TIMECODE_RE.match(item["t2"])
        assert TIMECODE_RE.match(item["answer"]["start"])
        assert TIMECODE_RE.match(item["answer"]["end"])

    def test_long_clip_uses_hh_mm_ss(self):
        out = to_daft_temporal_localization(
            [
                {
                    "question": "Q?",
                    "answer": {"start": 3600.0, "end": 3700.0},
                }
            ],
            ctx=VIDEO_CTX,
            duration=4000.0,
        )
        item = out["items"][0]
        assert item["answer"]["start"] == "01:00:00"
        assert item["answer"]["end"] == "01:01:40"


class TestMultipleItems:
    def test_items_preserve_order(self):
        items = [
            base_item(question="Q1?", answer={"start": 0.0, "end": 2.0}),
            base_item(question="Q2?", answer={"start": 2.0, "end": 4.0}),
            base_item(question="Q3?", answer={"start": 4.0, "end": 6.0}),
        ]
        out = to_daft_temporal_localization(items, ctx=VIDEO_CTX, duration=10.0)
        assert [it["question"] for it in out["items"]] == ["Q1?", "Q2?", "Q3?"]
