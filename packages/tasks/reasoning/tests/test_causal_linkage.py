# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# mypy: disable-error-code="no-untyped-def,unused-ignore"

"""Tests for the causal_linkage pure converter."""

from __future__ import annotations

import pytest
from reasoning.causal_linkage.converter import to_daft_causal_linkage
from reasoning.common import DAFT_VERSION, DaftConvertError, SceneContext

VIDEO_CTX = SceneContext(media_id="clip_001", iso_date="2026-04-20")
IMAGE_CTX = SceneContext(media_id="frame_001", is_image=True, iso_date="2026-04-20")


def _basic_item(**overrides) -> dict:
    base = {
        "t1": 1.0,
        "t2": 5.0,
        "question": "Explain the relationship between 00:01 and 00:05.",
        "answer": "The car at t1 caused the situation at t2 by ...",
    }
    base.update(overrides)
    return base


class TestEnvelope:
    def test_envelope(self):
        out = to_daft_causal_linkage([_basic_item()], ctx=VIDEO_CTX)
        assert out["version"] == DAFT_VERSION
        assert out["metadata"]["type"] == "causal_linkage"
        assert out["metadata"]["date"] == "2026-04-20"
        assert "video_id" not in out and "image_id" not in out
        assert isinstance(out["items"], list) and len(out["items"]) == 1

    def test_empty_returns_none(self):
        assert to_daft_causal_linkage([], ctx=VIDEO_CTX) is None

    def test_image_raises(self):
        with pytest.raises(DaftConvertError, match="image scenes"):
            to_daft_causal_linkage([_basic_item()], ctx=IMAGE_CTX)


class TestItemRequiredFields:
    def test_basic(self):
        out = to_daft_causal_linkage([_basic_item()], ctx=VIDEO_CTX)
        item = out["items"][0]
        assert item["video_id"] == VIDEO_CTX.media_id
        assert item["t1"] == "00:01"
        assert item["t2"] == "00:05"
        assert item["question"].startswith("Explain")
        assert item["answer"].startswith("The car")
        assert "video_type" not in item
        assert "reasoning" not in item

    def test_missing_t1_raises(self):
        with pytest.raises(DaftConvertError, match="t1"):
            to_daft_causal_linkage([_basic_item(t1=None)], ctx=VIDEO_CTX)

    def test_missing_t2_raises(self):
        with pytest.raises(DaftConvertError, match="t2"):
            to_daft_causal_linkage([_basic_item(t2=None)], ctx=VIDEO_CTX)

    def test_missing_question_raises(self):
        item = _basic_item()
        del item["question"]
        with pytest.raises(DaftConvertError, match="question"):
            to_daft_causal_linkage([item], ctx=VIDEO_CTX)

    def test_missing_answer_raises(self):
        item = _basic_item()
        del item["answer"]
        with pytest.raises(DaftConvertError, match="answer"):
            to_daft_causal_linkage([item], ctx=VIDEO_CTX)

    def test_non_dict_item_raises(self):
        with pytest.raises(DaftConvertError, match="must be a dict"):
            to_daft_causal_linkage(["nope"], ctx=VIDEO_CTX)


class TestTimecodes:
    def test_numeric_seconds_converted(self):
        out = to_daft_causal_linkage([_basic_item(t1=2.5, t2=7.125)], ctx=VIDEO_CTX)
        item = out["items"][0]
        assert item["t1"] == "00:02.500"
        assert item["t2"] == "00:07.125"

    def test_string_timecodes_normalized(self):
        out = to_daft_causal_linkage([_basic_item(t1="00:00:01", t2="00:00:05")], ctx=VIDEO_CTX)
        item = out["items"][0]
        # Both round-trip through timecode_to_seconds + seconds_to_timecode
        # so outputs are normalized to MM:SS form.
        assert item["t1"] == "00:01"
        assert item["t2"] == "00:05"

    def test_t2_less_than_t1_raises(self):
        with pytest.raises(DaftConvertError, match="t2.*< t1"):
            to_daft_causal_linkage([_basic_item(t1=5.0, t2=1.0)], ctx=VIDEO_CTX)

    def test_negative_seconds_raises(self):
        with pytest.raises(DaftConvertError, match=">= 0"):
            to_daft_causal_linkage([_basic_item(t1=-1.0)], ctx=VIDEO_CTX)

    def test_invalid_timecode_string_raises(self):
        with pytest.raises(DaftConvertError, match="not a valid DAFT"):
            to_daft_causal_linkage([_basic_item(t1="not a time")], ctx=VIDEO_CTX)

    def test_clamped_to_duration(self):
        out = to_daft_causal_linkage(
            [_basic_item(t1=2.0, t2=15.0)],
            ctx=VIDEO_CTX,
            duration=10.0,
        )
        item = out["items"][0]
        assert item["t2"] == "00:10"

    def test_non_finite_timestamp_raises(self):
        with pytest.raises(DaftConvertError, match="finite"):
            to_daft_causal_linkage([_basic_item(t1=float("inf"))], ctx=VIDEO_CTX)

    def test_invalid_duration_raises(self):
        with pytest.raises(DaftConvertError, match="duration"):
            to_daft_causal_linkage([_basic_item(t1=2.0)], ctx=VIDEO_CTX, duration=float("nan"))

    def test_negative_duration_raises(self):
        with pytest.raises(DaftConvertError, match="duration"):
            to_daft_causal_linkage([_basic_item(t1=2.0)], ctx=VIDEO_CTX, duration=-1.0)

    def test_bool_t1_raises(self):
        with pytest.raises(DaftConvertError, match="number or"):
            to_daft_causal_linkage([_basic_item(t1=True)], ctx=VIDEO_CTX)


class TestVideoType:
    def test_anomaly_passthrough(self):
        out = to_daft_causal_linkage([_basic_item(video_type="anomaly")], ctx=VIDEO_CTX)
        item = out["items"][0]
        assert item["video_type"] == "anomaly"
        # video_type sits between video_id and t1/t2 in the schema example;
        # respect that key order in the on-disk file.
        keys = list(item.keys())
        assert keys.index("video_id") < keys.index("video_type") < keys.index("t1")

    def test_normal_passthrough(self):
        out = to_daft_causal_linkage([_basic_item(video_type="normal")], ctx=VIDEO_CTX)
        assert out["items"][0]["video_type"] == "normal"

    def test_unknown_video_type_raises(self):
        with pytest.raises(DaftConvertError, match="video_type"):
            to_daft_causal_linkage([_basic_item(video_type="weird")], ctx=VIDEO_CTX)


class TestReasoning:
    def test_passthrough(self):
        out = to_daft_causal_linkage(
            [_basic_item(reasoning="Captions support this.")],
            ctx=VIDEO_CTX,
        )
        assert out["items"][0]["reasoning"] == "Captions support this."

    def test_empty_dropped(self):
        out = to_daft_causal_linkage(
            [_basic_item(reasoning="   ")],
            ctx=VIDEO_CTX,
        )
        assert "reasoning" not in out["items"][0]

    def test_non_string_raises(self):
        with pytest.raises(DaftConvertError, match="reasoning"):
            to_daft_causal_linkage(
                [_basic_item(reasoning=42)],
                ctx=VIDEO_CTX,
            )
