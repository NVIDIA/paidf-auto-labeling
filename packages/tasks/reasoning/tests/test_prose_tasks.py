# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# mypy: disable-error-code="no-untyped-def,unused-ignore"

from __future__ import annotations

import pytest
from reasoning.common import DAFT_VERSION, DaftConvertError, SceneContext
from reasoning.prose_tasks.converter import to_daft_scene_description, to_daft_video_summarization

VIDEO_CTX = SceneContext(media_id="clip_001", iso_date="2026-04-20")
IMAGE_CTX = SceneContext(media_id="frame_001", is_image=True, iso_date="2026-04-20")


class TestSceneDescriptionEnvelope:
    def test_video_envelope(self):
        out = to_daft_scene_description("A car drives through an intersection.", ctx=VIDEO_CTX)
        assert out["version"] == DAFT_VERSION
        assert out["metadata"]["type"] == "scene_description"
        assert out["metadata"]["date"] == "2026-04-20"
        # Top-level scene id is omitted on task files; per-item carries it instead.
        assert "video_id" not in out and "image_id" not in out
        assert isinstance(out["items"], list) and len(out["items"]) == 1

    def test_image_envelope(self):
        out = to_daft_scene_description("A truck parked on a sunny street.", ctx=IMAGE_CTX)
        assert out["version"] == DAFT_VERSION
        assert out["metadata"]["type"] == "scene_description"

    def test_metadata_threads_optional_fields(self):
        ctx = SceneContext(
            media_id="x",
            iso_date="2026-04-20",
            license_str="CC-BY-4.0",
            tags=("test", "synthetic"),
        )
        out = to_daft_scene_description("desc", ctx=ctx)
        assert out["metadata"]["license"] == "CC-BY-4.0"
        assert out["metadata"]["tags"] == ["test", "synthetic"]


class TestSceneDescriptionItem:
    def test_video_item_uses_video_id(self):
        out = to_daft_scene_description("A car drives through an intersection.", ctx=VIDEO_CTX)
        [item] = out["items"]
        assert item["video_id"] == VIDEO_CTX.media_id
        assert "image_id" not in item
        assert item["question"] == "Describe the scene."
        assert item["answer"] == "A car drives through an intersection."
        assert "reasoning" not in item

    def test_image_item_uses_image_id(self):
        # DAFT v3 scene_description.json item is ``oneOf(video_id, image_id)``;
        # image scenes must carry image_id so the validator's oneOf passes.
        out = to_daft_scene_description("A still frame of a parked car.", ctx=IMAGE_CTX)
        [item] = out["items"]
        assert item["image_id"] == IMAGE_CTX.media_id
        assert "video_id" not in item

    def test_custom_question_passthrough(self):
        out = to_daft_scene_description(
            "It is dawn.",
            ctx=VIDEO_CTX,
            question="What time of day is shown?",
        )
        [item] = out["items"]
        assert item["question"] == "What time of day is shown?"

    def test_reasoning_passthrough(self):
        out = to_daft_scene_description(
            "A red sedan turns left.",
            ctx=VIDEO_CTX,
            reasoning="Vehicle wheels rotate left while moving.",
        )
        [item] = out["items"]
        assert item["reasoning"] == "Vehicle wheels rotate left while moving."

    def test_text_is_stripped(self):
        out = to_daft_scene_description("  hello  ", ctx=VIDEO_CTX)
        [item] = out["items"]
        assert item["answer"] == "hello"


class TestSceneDescriptionEmpty:
    @pytest.mark.parametrize("empty", ["", "   ", "\n\t"])
    def test_empty_answer_returns_none(self, empty):
        # DAFT requires ``minItems: 1`` on items; converter must signal "skip"
        # rather than emit an illegal stub.
        assert to_daft_scene_description(empty, ctx=VIDEO_CTX) is None

    def test_none_answer_returns_none(self):
        assert to_daft_scene_description(None, ctx=VIDEO_CTX) is None  # type: ignore[arg-type]

    def test_empty_question_raises(self):
        with pytest.raises(DaftConvertError, match="question"):
            to_daft_scene_description("answer", ctx=VIDEO_CTX, question="   ")

    def test_empty_reasoning_omits_key(self):
        out = to_daft_scene_description("answer", ctx=VIDEO_CTX, reasoning="   ")
        [item] = out["items"]
        assert "reasoning" not in item


class TestVideoSummarizationEnvelope:
    def test_envelope(self):
        out = to_daft_video_summarization(
            "Two cars approach an intersection; one runs the light.",
            ctx=VIDEO_CTX,
        )
        assert out["version"] == DAFT_VERSION
        assert out["metadata"]["type"] == "video_summarization"
        assert "video_id" not in out and "image_id" not in out
        assert len(out["items"]) == 1

    def test_image_ctx_rejected(self):
        # Summarization is inherently temporal; the schema would technically
        # allow image_id-only items but it is semantically meaningless.
        with pytest.raises(DaftConvertError, match="image"):
            to_daft_video_summarization("summary", ctx=IMAGE_CTX)


class TestVideoSummarizationItem:
    def test_default_question(self):
        out = to_daft_video_summarization("A summary.", ctx=VIDEO_CTX)
        [item] = out["items"]
        assert item["question"] == "Summarize the events in the video."
        assert item["answer"] == "A summary."
        assert item["video_id"] == VIDEO_CTX.media_id
        assert "timestamp" not in item
        assert "reasoning" not in item

    def test_custom_question(self):
        out = to_daft_video_summarization(
            "answer",
            ctx=VIDEO_CTX,
            question="Briefly describe what happens.",
        )
        [item] = out["items"]
        assert item["question"] == "Briefly describe what happens."

    def test_reasoning_passthrough(self):
        out = to_daft_video_summarization(
            "summary",
            ctx=VIDEO_CTX,
            reasoning="Aggregated event captions sequentially.",
        )
        [item] = out["items"]
        assert item["reasoning"] == "Aggregated event captions sequentially."

    def test_timestamp_emitted(self):
        out = to_daft_video_summarization(
            "summary",
            ctx=VIDEO_CTX,
            timestamp=(0.0, 12.5),
        )
        [item] = out["items"]
        assert item["timestamp"] == {"start": 0.0, "end": 12.5}


class TestVideoSummarizationEmpty:
    @pytest.mark.parametrize("empty", ["", "   ", "\n\t"])
    def test_empty_returns_none(self, empty):
        assert to_daft_video_summarization(empty, ctx=VIDEO_CTX) is None

    def test_empty_question_raises(self):
        with pytest.raises(DaftConvertError, match="question"):
            to_daft_video_summarization("answer", ctx=VIDEO_CTX, question="   ")


class TestVideoSummarizationTimestampValidation:
    def test_non_finite_timestamp_raises(self):
        with pytest.raises(DaftConvertError, match="finite"):
            to_daft_video_summarization("s", ctx=VIDEO_CTX, timestamp=(float("inf"), 5.0))

    def test_negative_start_raises(self):
        with pytest.raises(DaftConvertError, match=">= 0"):
            to_daft_video_summarization("s", ctx=VIDEO_CTX, timestamp=(-1.0, 5.0))

    def test_negative_end_raises(self):
        with pytest.raises(DaftConvertError, match=">= 0"):
            to_daft_video_summarization("s", ctx=VIDEO_CTX, timestamp=(0.0, -1.0))

    def test_end_before_start_raises(self):
        with pytest.raises(DaftConvertError, match="end < start"):
            to_daft_video_summarization("s", ctx=VIDEO_CTX, timestamp=(5.0, 1.0))

    def test_zero_window_allowed(self):
        # An instantaneous event (start == end) is valid in the schema.
        out = to_daft_video_summarization("s", ctx=VIDEO_CTX, timestamp=(2.0, 2.0))
        [item] = out["items"]
        assert item["timestamp"] == {"start": 2.0, "end": 2.0}
