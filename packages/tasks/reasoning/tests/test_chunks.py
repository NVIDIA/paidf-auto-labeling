# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# mypy: disable-error-code="no-untyped-def,unused-ignore"

from __future__ import annotations

import re

import pytest
from reasoning.chunks.converter import to_daft_chunks
from reasoning.common import DAFT_VERSION, DaftConvertError, SceneContext

VIDEO_CTX = SceneContext(media_id="clip_001", iso_date="2026-04-20")
IMAGE_CTX = SceneContext(media_id="frame_001", is_image=True, iso_date="2026-04-20")

# Schema regex from chunks.schema.json — every emitted timecode must match.
TIMECODE_RE = re.compile(r"^(\d{2}:)?\d{2}:\d{2}(\.\d+)?$")


def pl_window(**overrides) -> dict:
    """Window shape PL's window_vlm_llm runner writes to sidecars/metadata.json."""
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
        # DAFT requires minItems: 1 on chunks; converter signals "skip writing".
        assert to_daft_chunks([], ctx=VIDEO_CTX) is None

    def test_image_ctx_rejected(self):
        with pytest.raises(DaftConvertError, match="image"):
            to_daft_chunks([pl_window()], ctx=IMAGE_CTX)


class TestEnvelope:
    def test_envelope_shape(self):
        out = to_daft_chunks([pl_window()], ctx=VIDEO_CTX)
        assert out["version"] == DAFT_VERSION
        # chunks.json carries top-level video_id (unlike instances.json).
        assert out["video_id"] == "clip_001"
        assert out["metadata"]["type"] == "chunks"
        assert out["metadata"]["date"] == "2026-04-20"

    def test_metadata_threads_optional_fields(self):
        ctx = SceneContext(media_id="clip", license_str="CC-BY-4.0", tags=("urban",))
        out = to_daft_chunks([pl_window()], ctx=ctx)
        assert out["metadata"]["license"] == "CC-BY-4.0"
        assert out["metadata"]["tags"] == ["urban"]


class TestChunkShape:
    def test_minimal_chunk(self):
        out = to_daft_chunks([pl_window()], ctx=VIDEO_CTX)
        [chunk] = out["chunks"]
        # description preference defaults to "description" first, then caption.
        assert chunk == {
            "chunk_id": "chunk_001",
            "start": "00:00",
            "end": "00:04",
            "description": "Traffic flows normally.",
        }

    def test_auto_chunk_id_padding(self):
        wins = [pl_window(start_s=i, end_s=i + 1.0) for i in range(12)]
        out = to_daft_chunks(wins, ctx=VIDEO_CTX)
        ids = [c["chunk_id"] for c in out["chunks"]]
        assert ids[0] == "chunk_001"
        assert ids[9] == "chunk_010"
        assert ids[11] == "chunk_012"

    def test_explicit_chunk_id_preserved(self):
        out = to_daft_chunks([pl_window(chunk_id="my_special_chunk")], ctx=VIDEO_CTX)
        [chunk] = out["chunks"]
        assert chunk["chunk_id"] == "my_special_chunk"

    def test_blank_chunk_id_falls_back_to_auto(self):
        out = to_daft_chunks([pl_window(chunk_id="   ")], ctx=VIDEO_CTX)
        [chunk] = out["chunks"]
        assert chunk["chunk_id"] == "chunk_001"

    def test_extra_per_window_keys_stripped(self):
        # additionalProperties: false at the chunk level — no PL bookkeeping leak.
        win = pl_window(start_frame=10, end_frame=119, _debug="x", vlm_verify={"hits": 1})
        out = to_daft_chunks([win], ctx=VIDEO_CTX)
        [chunk] = out["chunks"]
        assert set(chunk.keys()) == {"chunk_id", "start", "end", "description"}

    def test_tags_passthrough(self):
        out = to_daft_chunks([pl_window(tags=["normal_flow", "urban"])], ctx=VIDEO_CTX)
        [chunk] = out["chunks"]
        assert chunk["tags"] == ["normal_flow", "urban"]

    def test_empty_or_blank_tags_stripped(self):
        out = to_daft_chunks([pl_window(tags=["", "  ", "ok"])], ctx=VIDEO_CTX)
        [chunk] = out["chunks"]
        assert chunk["tags"] == ["ok"]

    def test_all_blank_tags_omits_key(self):
        out = to_daft_chunks([pl_window(tags=["", "  "])], ctx=VIDEO_CTX)
        [chunk] = out["chunks"]
        assert "tags" not in chunk

    def test_empty_tags_list_omits_key(self):
        out = to_daft_chunks([pl_window(tags=[])], ctx=VIDEO_CTX)
        [chunk] = out["chunks"]
        assert "tags" not in chunk


class TestTimecodes:
    def test_seconds_to_mmss(self):
        out = to_daft_chunks([pl_window(start_s=0.0, end_s=2.7)], ctx=VIDEO_CTX)
        [chunk] = out["chunks"]
        assert chunk["start"] == "00:00"
        assert chunk["end"] == "00:02.700"
        for tc in (chunk["start"], chunk["end"]):
            assert TIMECODE_RE.fullmatch(tc), f"timecode does not match schema regex: {tc!r}"

    def test_long_video_uses_hhmmss(self):
        # > 1 hour triggers the HH:MM:SS branch.
        out = to_daft_chunks([pl_window(start_s=3600.0, end_s=3725.5)], ctx=VIDEO_CTX)
        [chunk] = out["chunks"]
        assert chunk["start"] == "01:00:00"
        assert chunk["end"] == "01:02:05.500"
        for tc in (chunk["start"], chunk["end"]):
            assert TIMECODE_RE.fullmatch(tc)

    def test_clamp_to_duration(self):
        out = to_daft_chunks(
            [pl_window(start_s=0.0, end_s=20.0)],
            ctx=VIDEO_CTX,
            duration=10.0,
        )
        [chunk] = out["chunks"]
        assert chunk["end"] == "00:10"

    def test_non_finite_timestamp_rejected(self):
        with pytest.raises(DaftConvertError, match="must be finite"):
            to_daft_chunks([pl_window(start_s=float("nan"))], ctx=VIDEO_CTX)

    def test_negative_duration_rejected(self):
        with pytest.raises(DaftConvertError, match="duration"):
            to_daft_chunks([pl_window()], ctx=VIDEO_CTX, duration=-1.0)

    def test_negative_start_clamped_to_zero(self):
        out = to_daft_chunks(
            [pl_window(start_s=-1.0, end_s=2.0)],
            ctx=VIDEO_CTX,
        )
        [chunk] = out["chunks"]
        assert chunk["start"] == "00:00"


class TestKeyAliases:
    def test_alternative_start_end_keys(self):
        win = {
            "start_seconds": 1.0,
            "end_seconds": 2.0,
            "description": "x",
        }
        out = to_daft_chunks([win], ctx=VIDEO_CTX)
        [chunk] = out["chunks"]
        assert chunk["start"] == "00:01"
        assert chunk["end"] == "00:02"

    def test_caption_key_used_when_no_description(self):
        win = {"start_s": 0.0, "end_s": 1.0, "caption": "from caption"}
        out = to_daft_chunks([win], ctx=VIDEO_CTX)
        [chunk] = out["chunks"]
        assert chunk["description"] == "from caption"

    def test_description_keys_override_preference(self):
        # Default order prefers "description" then "caption" then "enhanced_caption";
        # callers pass description_keys to flip the preference (e.g. prefer LM-rewritten).
        win = pl_window()  # has both caption + enhanced_caption
        out = to_daft_chunks([win], ctx=VIDEO_CTX, description_keys=("enhanced_caption", "caption"))
        [chunk] = out["chunks"]
        assert chunk["description"] == "A white sedan and a gray sedan drive in adjacent lanes."

    def test_description_strip_whitespace(self):
        win = {"start_s": 0.0, "end_s": 1.0, "description": "  with whitespace  "}
        out = to_daft_chunks([win], ctx=VIDEO_CTX)
        [chunk] = out["chunks"]
        assert chunk["description"] == "with whitespace"

    def test_blank_description_falls_back_to_next_key(self):
        # description present but blank; converter must fall through to caption.
        win = {"start_s": 0.0, "end_s": 1.0, "description": "   ", "caption": "real text"}
        out = to_daft_chunks([win], ctx=VIDEO_CTX)
        [chunk] = out["chunks"]
        assert chunk["description"] == "real text"


class TestErrors:
    def test_window_must_be_dict(self):
        with pytest.raises(DaftConvertError, match="must be a dict"):
            to_daft_chunks(["not a dict"], ctx=VIDEO_CTX)  # type: ignore[list-item]

    def test_missing_start(self):
        win = {"end_s": 1.0, "caption": "x"}
        with pytest.raises(DaftConvertError, match="start"):
            to_daft_chunks([win], ctx=VIDEO_CTX)

    def test_missing_end(self):
        win = {"start_s": 0.0, "caption": "x"}
        with pytest.raises(DaftConvertError, match="end"):
            to_daft_chunks([win], ctx=VIDEO_CTX)

    def test_missing_description(self):
        win = {"start_s": 0.0, "end_s": 1.0}
        with pytest.raises(DaftConvertError, match="description"):
            to_daft_chunks([win], ctx=VIDEO_CTX)

    def test_non_numeric_start(self):
        win = {"start_s": "zero", "end_s": 1.0, "caption": "x"}
        with pytest.raises(DaftConvertError, match="must be a number"):
            to_daft_chunks([win], ctx=VIDEO_CTX)

    def test_bool_start_rejected(self):
        # isinstance(True, int) is True; converter must filter bools.
        win = {"start_s": True, "end_s": 1.0, "caption": "x"}
        with pytest.raises(DaftConvertError, match="must be a number"):
            to_daft_chunks([win], ctx=VIDEO_CTX)

    def test_end_before_start(self):
        win = pl_window(start_s=5.0, end_s=2.0)
        with pytest.raises(DaftConvertError, match="end .* < start"):
            to_daft_chunks([win], ctx=VIDEO_CTX)

    def test_zero_window_allowed(self):
        # An instantaneous chunk (start == end) is technically valid in the schema.
        win = pl_window(start_s=2.0, end_s=2.0)
        out = to_daft_chunks([win], ctx=VIDEO_CTX)
        [chunk] = out["chunks"]
        assert chunk["start"] == chunk["end"] == "00:02"
