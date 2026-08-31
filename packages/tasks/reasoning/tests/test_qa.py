# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# mypy: disable-error-code="no-untyped-def,unused-ignore"

"""Tests for the open-ended QA family converters (open_qa, mcq_openended,
bcq_openended). Pure-converter tests — no LLM, no I/O."""

from __future__ import annotations

import pytest
from reasoning.common import DAFT_VERSION, DaftConvertError, SceneContext
from reasoning.qa.converter import (
    to_daft_bcq_openended,
    to_daft_mcq_openended,
    to_daft_open_qa,
)

VIDEO_CTX = SceneContext(media_id="clip_001", iso_date="2026-04-20")
IMAGE_CTX = SceneContext(media_id="frame_001", is_image=True, iso_date="2026-04-20")


# ---------------------------------------------------------------------------
# open_qa
# ---------------------------------------------------------------------------


class TestOpenQaEnvelope:
    def test_envelope(self):
        out = to_daft_open_qa(
            [{"question": "Q?", "answer": "A."}],
            ctx=VIDEO_CTX,
        )
        assert out["version"] == DAFT_VERSION
        assert out["metadata"]["type"] == "open_qa"
        assert out["metadata"]["date"] == "2026-04-20"
        assert "video_id" not in out and "image_id" not in out
        assert isinstance(out["items"], list) and len(out["items"]) == 1

    def test_empty_returns_none(self):
        assert to_daft_open_qa([], ctx=VIDEO_CTX) is None


class TestOpenQaItem:
    def test_video_item_uses_video_id(self):
        out = to_daft_open_qa([{"question": "Q?", "answer": "A."}], ctx=VIDEO_CTX)
        [item] = out["items"]
        assert item["video_id"] == VIDEO_CTX.media_id
        assert "image_id" not in item
        assert item["question"] == "Q?"
        assert item["answer"] == "A."
        assert "reasoning" not in item

    def test_image_item_uses_image_id(self):
        out = to_daft_open_qa([{"question": "Q?", "answer": "A."}], ctx=IMAGE_CTX)
        [item] = out["items"]
        assert item["image_id"] == IMAGE_CTX.media_id
        assert "video_id" not in item

    def test_reasoning_passthrough(self):
        out = to_daft_open_qa(
            [{"question": "Q?", "answer": "A.", "reasoning": "Because X."}],
            ctx=VIDEO_CTX,
        )
        assert out["items"][0]["reasoning"] == "Because X."

    def test_empty_reasoning_dropped(self):
        out = to_daft_open_qa(
            [{"question": "Q?", "answer": "A.", "reasoning": "   "}],
            ctx=VIDEO_CTX,
        )
        assert "reasoning" not in out["items"][0]

    def test_text_is_stripped(self):
        out = to_daft_open_qa(
            [{"question": "  Q?  ", "answer": "  A.  "}],
            ctx=VIDEO_CTX,
        )
        item = out["items"][0]
        assert item["question"] == "Q?"
        assert item["answer"] == "A."

    def test_missing_question_raises(self):
        with pytest.raises(DaftConvertError, match="question"):
            to_daft_open_qa([{"answer": "A."}], ctx=VIDEO_CTX)

    def test_missing_answer_raises(self):
        with pytest.raises(DaftConvertError, match="answer"):
            to_daft_open_qa([{"question": "Q?"}], ctx=VIDEO_CTX)

    def test_non_dict_item_raises(self):
        with pytest.raises(DaftConvertError, match="must be a dict"):
            to_daft_open_qa(["just a string"], ctx=VIDEO_CTX)

    def test_non_string_reasoning_raises(self):
        with pytest.raises(DaftConvertError, match="reasoning must be a string"):
            to_daft_open_qa(
                [{"question": "Q?", "answer": "A.", "reasoning": 123}],
                ctx=VIDEO_CTX,
            )


# ---------------------------------------------------------------------------
# mcq_openended
# ---------------------------------------------------------------------------


class TestMcqOpenendedEnvelope:
    def test_envelope(self):
        out = to_daft_mcq_openended(
            [{"question": "Q?", "answer": "A. Yes it is."}],
            ctx=VIDEO_CTX,
        )
        assert out["version"] == DAFT_VERSION
        assert out["metadata"]["type"] == "mcq_openended"
        assert isinstance(out["items"], list) and len(out["items"]) == 1

    def test_empty_returns_none(self):
        assert to_daft_mcq_openended([], ctx=VIDEO_CTX) is None


class TestMcqOpenendedItem:
    def test_basic_no_options(self):
        out = to_daft_mcq_openended(
            [{"question": "Q?", "answer": "A. Reason text."}],
            ctx=VIDEO_CTX,
        )
        [item] = out["items"]
        assert item["video_id"] == VIDEO_CTX.media_id
        assert item["question"] == "Q?"
        assert item["answer"] == "A. Reason text."
        assert "options" not in item
        assert "reasoning" not in item

    def test_image_uses_image_id(self):
        out = to_daft_mcq_openended(
            [{"question": "Q?", "answer": "A. ok."}],
            ctx=IMAGE_CTX,
        )
        [item] = out["items"]
        assert item["image_id"] == IMAGE_CTX.media_id

    def test_options_passthrough(self):
        out = to_daft_mcq_openended(
            [
                {
                    "question": "Q?",
                    "answer": "B. The truck is large.",
                    "options": {"A": "car", "B": "truck"},
                }
            ],
            ctx=VIDEO_CTX,
        )
        [item] = out["items"]
        assert item["options"] == {"A": "car", "B": "truck"}

    def test_options_letter_must_match_answer(self):
        with pytest.raises(DaftConvertError, match="not a key in options"):
            to_daft_mcq_openended(
                [
                    {
                        "question": "Q?",
                        "answer": "C. Out of range.",
                        "options": {"A": "car", "B": "truck"},
                    }
                ],
                ctx=VIDEO_CTX,
            )

    def test_options_below_min_size_raises(self):
        with pytest.raises(DaftConvertError, match="minProperties"):
            to_daft_mcq_openended(
                [
                    {
                        "question": "Q?",
                        "answer": "A. text.",
                        "options": {"A": "car"},
                    }
                ],
                ctx=VIDEO_CTX,
            )

    def test_options_non_letter_key_raises(self):
        with pytest.raises(DaftConvertError, match="single uppercase letter"):
            to_daft_mcq_openended(
                [
                    {
                        "question": "Q?",
                        "answer": "A. text.",
                        "options": {"A": "car", "1": "truck"},
                    }
                ],
                ctx=VIDEO_CTX,
            )

    def test_options_non_string_value_raises(self):
        with pytest.raises(DaftConvertError, match="must be a string"):
            to_daft_mcq_openended(
                [
                    {
                        "question": "Q?",
                        "answer": "A. text.",
                        "options": {"A": "car", "B": 42},
                    }
                ],
                ctx=VIDEO_CTX,
            )

    def test_options_empty_string_value_raises(self):
        with pytest.raises(DaftConvertError, match="empty after strip"):
            to_daft_mcq_openended(
                [
                    {
                        "question": "Q?",
                        "answer": "A. text.",
                        "options": {"A": "car", "B": "  "},
                    }
                ],
                ctx=VIDEO_CTX,
            )

    @pytest.mark.parametrize(
        "bad_answer",
        [
            "A",  # missing the explanation tail
            "A.",  # missing the space + explanation
            "Option A. text.",  # extra prefix
            "a. lowercase",  # wrong case
            "AB. two letters",  # two letters
            "1. number",  # number not letter
        ],
    )
    def test_answer_regex_enforced(self, bad_answer):
        with pytest.raises(DaftConvertError, match="must match"):
            to_daft_mcq_openended(
                [{"question": "Q?", "answer": bad_answer}],
                ctx=VIDEO_CTX,
            )

    def test_reasoning_passthrough(self):
        out = to_daft_mcq_openended(
            [
                {
                    "question": "Q?",
                    "answer": "A. text.",
                    "reasoning": "Because the captions said so.",
                }
            ],
            ctx=VIDEO_CTX,
        )
        assert out["items"][0]["reasoning"] == "Because the captions said so."


# ---------------------------------------------------------------------------
# bcq_openended
# ---------------------------------------------------------------------------


class TestBcqOpenendedEnvelope:
    def test_envelope(self):
        out = to_daft_bcq_openended(
            [{"question": "Q?", "answer": "Yes. Because X."}],
            ctx=VIDEO_CTX,
        )
        assert out["version"] == DAFT_VERSION
        assert out["metadata"]["type"] == "bcq_openended"
        assert isinstance(out["items"], list) and len(out["items"]) == 1

    def test_empty_returns_none(self):
        assert to_daft_bcq_openended([], ctx=VIDEO_CTX) is None


class TestBcqOpenendedItem:
    def test_yes_answer(self):
        out = to_daft_bcq_openended(
            [{"question": "Q?", "answer": "Yes. Affirmative because A."}],
            ctx=VIDEO_CTX,
        )
        [item] = out["items"]
        assert item["video_id"] == VIDEO_CTX.media_id
        assert item["answer"].startswith("Yes. ")

    def test_no_answer(self):
        out = to_daft_bcq_openended(
            [{"question": "Q?", "answer": "No. Negative because B."}],
            ctx=VIDEO_CTX,
        )
        assert out["items"][0]["answer"].startswith("No. ")

    def test_image_uses_image_id(self):
        out = to_daft_bcq_openended(
            [{"question": "Q?", "answer": "Yes. ok."}],
            ctx=IMAGE_CTX,
        )
        [item] = out["items"]
        assert item["image_id"] == IMAGE_CTX.media_id

    @pytest.mark.parametrize(
        "bad_answer",
        [
            "Yes",  # missing tail
            "Yes.",  # missing space + tail
            "yes. lowercase",  # wrong case
            "YES. all caps",  # wrong case
            "Maybe. not in alphabet",  # wrong token
            "True. not in alphabet",
            "It is. missing leading verdict",
        ],
    )
    def test_answer_regex_enforced(self, bad_answer):
        with pytest.raises(DaftConvertError, match=r"\(Yes\|No\)"):
            to_daft_bcq_openended(
                [{"question": "Q?", "answer": bad_answer}],
                ctx=VIDEO_CTX,
            )

    def test_reasoning_passthrough(self):
        out = to_daft_bcq_openended(
            [
                {
                    "question": "Q?",
                    "answer": "No. Empty.",
                    "reasoning": "Captions silent.",
                }
            ],
            ctx=VIDEO_CTX,
        )
        assert out["items"][0]["reasoning"] == "Captions silent."

    def test_missing_question_raises(self):
        with pytest.raises(DaftConvertError, match="question"):
            to_daft_bcq_openended([{"answer": "Yes. ok."}], ctx=VIDEO_CTX)

    def test_missing_answer_raises(self):
        with pytest.raises(DaftConvertError, match="answer"):
            to_daft_bcq_openended([{"question": "Q?"}], ctx=VIDEO_CTX)
