# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# mypy: disable-error-code="no-untyped-def,unused-ignore"

from __future__ import annotations

import re

import pytest
from reasoning.common import DAFT_VERSION, DaftConvertError, SceneContext
from reasoning.msted.converter import to_daft_msted

VIDEO_CTX = SceneContext(media_id="clip_42", iso_date="2026-04-20")
IMAGE_CTX = SceneContext(media_id="frame_001", is_image=True, iso_date="2026-04-20")

# Schema regex from msted.schema.json — every emitted timecode must match.
TIMECODE_RE = re.compile(r"^(\d{2}:)?\d{2}:\d{2}(\.\d+)?$")


def base_structured(**overrides) -> dict:
    """Minimal valid LLM output shape — same as the bundled prompt asks for."""
    obj = {
        "scene_description": "A car drives down a residential street and turns right.",
        "temporal_spatial_localization": [
            {
                "start": 0.0,
                "end": 4.0,
                "description": "Car accelerates from rest in the right lane.",
            },
            {
                "start": 4.0,
                "end": 7.0,
                "description": "Car turns right at the intersection.",
                "spatial_region": "center frame",
            },
        ],
        "event_description": {
            "category": "lane change and turn",
            "description": "A sedan accelerates and executes a right turn.",
            "cause": "anticipated turn at the intersection",
        },
    }
    obj.update(overrides)
    return obj


class TestEnvelope:
    def test_version_and_metadata(self):
        out = to_daft_msted(base_structured(), ctx=VIDEO_CTX)
        assert out["version"] == DAFT_VERSION
        assert out["metadata"]["type"] == "msted"
        assert out["metadata"]["date"] == "2026-04-20"

    def test_top_level_video_id(self):
        # MSTED schema requires top-level video_id (unlike, e.g., temporal_description).
        out = to_daft_msted(base_structured(), ctx=VIDEO_CTX)
        assert out["video_id"] == "clip_42"
        assert "image_id" not in out

    def test_image_ctx_rejected(self):
        with pytest.raises(DaftConvertError, match="image"):
            to_daft_msted(base_structured(), ctx=IMAGE_CTX)

    def test_metadata_threads_optional_fields(self):
        ctx = SceneContext(media_id="clip", license_str="CC-BY-4.0", tags=("urban",))
        out = to_daft_msted(base_structured(), ctx=ctx)
        assert out["metadata"]["license"] == "CC-BY-4.0"
        assert out["metadata"]["tags"] == ["urban"]

    def test_sources_emitted_when_supplied(self):
        out = to_daft_msted(
            base_structured(),
            ctx=VIDEO_CTX,
            sources=["sidecars/metadata.json", "contextual/video.json"],
        )
        assert out["sources"] == ["sidecars/metadata.json", "contextual/video.json"]

    def test_sources_omitted_when_none(self):
        out = to_daft_msted(base_structured(), ctx=VIDEO_CTX)
        assert "sources" not in out


class TestSceneDescription:
    def test_minimal_scene_description(self):
        out = to_daft_msted(base_structured(), ctx=VIDEO_CTX)
        assert out["scene_description"].startswith("A car drives")

    def test_strips_whitespace(self):
        out = to_daft_msted(
            base_structured(scene_description="   trimmed   "),
            ctx=VIDEO_CTX,
        )
        assert out["scene_description"] == "trimmed"

    def test_empty_scene_description_rejected(self):
        with pytest.raises(DaftConvertError, match="scene_description"):
            to_daft_msted(
                base_structured(scene_description="   "),
                ctx=VIDEO_CTX,
            )

    def test_non_string_scene_description_rejected(self):
        with pytest.raises(DaftConvertError, match="scene_description"):
            to_daft_msted(
                base_structured(scene_description=123),
                ctx=VIDEO_CTX,
            )


class TestTemporalSpatialLocalization:
    def test_segments_converted_to_timecodes(self):
        out = to_daft_msted(base_structured(), ctx=VIDEO_CTX)
        segs = out["temporal_spatial_localization"]
        assert len(segs) == 2
        assert segs[0]["start"] == "00:00"
        assert segs[0]["end"] == "00:04"
        assert segs[1]["start"] == "00:04"
        assert segs[1]["end"] == "00:07"
        for s in segs:
            assert TIMECODE_RE.match(s["start"])
            assert TIMECODE_RE.match(s["end"])

    def test_string_timecodes_accepted(self):
        out = to_daft_msted(
            base_structured(
                temporal_spatial_localization=[
                    {"start": "00:00.500", "end": "00:04", "description": "x"},
                ]
            ),
            ctx=VIDEO_CTX,
        )
        seg = out["temporal_spatial_localization"][0]
        assert seg["start"] == "00:00.500"
        assert seg["end"] == "00:04"

    def test_invalid_string_timecode_rejected(self):
        with pytest.raises(DaftConvertError, match="timecode"):
            to_daft_msted(
                base_structured(
                    temporal_spatial_localization=[
                        {"start": "not-a-timecode", "end": "00:04", "description": "x"},
                    ]
                ),
                ctx=VIDEO_CTX,
            )

    def test_negative_seconds_rejected(self):
        with pytest.raises(DaftConvertError, match=">= 0"):
            to_daft_msted(
                base_structured(
                    temporal_spatial_localization=[
                        {"start": -1.0, "end": 4.0, "description": "x"},
                    ]
                ),
                ctx=VIDEO_CTX,
            )

    def test_negative_duration_rejected(self):
        with pytest.raises(DaftConvertError, match="duration"):
            to_daft_msted(
                base_structured(
                    temporal_spatial_localization=[
                        {"start": 1.0, "end": 4.0, "description": "x"},
                    ]
                ),
                ctx=VIDEO_CTX,
                duration=-1.0,
            )

    def test_bool_rejected_as_timecode(self):
        with pytest.raises(DaftConvertError, match="number or timecode"):
            to_daft_msted(
                base_structured(
                    temporal_spatial_localization=[
                        {"start": True, "end": 4.0, "description": "x"},
                    ]
                ),
                ctx=VIDEO_CTX,
            )

    def test_reversed_segment_range_rejected(self):
        """LLM regression guard: the legacy "00:00-00:04.839" concatenated-
        timecode bug surfaced as reversed start/end pairs in MSTED.
        Even with the patched ``msted_default.yaml`` prompt, the converter
        is the last line of defense against an adversarial LLM swapping
        the fields. Match on the stable error code so log-grepping in
        prod stays cheap."""
        with pytest.raises(DaftConvertError, match=r"segment\[0\] has end .* earlier than start"):
            to_daft_msted(
                base_structured(
                    temporal_spatial_localization=[
                        {"start": 5.0, "end": 2.0, "description": "x"},
                    ]
                ),
                ctx=VIDEO_CTX,
                duration=10.0,
            )

    def test_equal_start_and_end_segment_allowed(self):
        """Zero-duration segments (instant marker frames) are valid; only
        ``end < start`` is rejected by the monotonicity check."""
        out = to_daft_msted(
            base_structured(
                temporal_spatial_localization=[
                    {"start": 2.5, "end": 2.5, "description": "x"},
                ]
            ),
            ctx=VIDEO_CTX,
        )
        seg = out["temporal_spatial_localization"][0]
        assert seg["start"] == seg["end"] == "00:02.500"

    def test_reversed_segment_with_string_timecodes_rejected(self):
        """The check operates on post-coercion seconds, so it must
        catch reversed ranges expressed as strings just as it does
        numeric inputs."""
        with pytest.raises(DaftConvertError, match=r"segment\[0\] has end .* earlier than start"):
            to_daft_msted(
                base_structured(
                    temporal_spatial_localization=[
                        {"start": "00:05", "end": "00:02", "description": "x"},
                    ]
                ),
                ctx=VIDEO_CTX,
            )

    def test_clamp_to_duration(self):
        out = to_daft_msted(
            base_structured(
                temporal_spatial_localization=[
                    {"start": 58.0, "end": 65.0, "description": "x"},
                ]
            ),
            ctx=VIDEO_CTX,
            duration=60.0,
        )
        seg = out["temporal_spatial_localization"][0]
        assert seg["start"] == "00:58"
        assert seg["end"] == "01:00"

    def test_non_finite_segment_time_rejected(self):
        with pytest.raises(DaftConvertError, match="finite"):
            to_daft_msted(
                base_structured(
                    temporal_spatial_localization=[
                        {"start": float("nan"), "end": 2.0, "description": "x"},
                    ]
                ),
                ctx=VIDEO_CTX,
            )

    def test_spatial_region_kept_when_present(self):
        out = to_daft_msted(base_structured(), ctx=VIDEO_CTX)
        assert out["temporal_spatial_localization"][1]["spatial_region"] == "center frame"

    def test_spatial_region_omitted_when_blank(self):
        out = to_daft_msted(
            base_structured(
                temporal_spatial_localization=[
                    {"start": 0.0, "end": 1.0, "description": "x", "spatial_region": "   "},
                ]
            ),
            ctx=VIDEO_CTX,
        )
        assert "spatial_region" not in out["temporal_spatial_localization"][0]

    def test_spatial_region_strips(self):
        out = to_daft_msted(
            base_structured(
                temporal_spatial_localization=[
                    {"start": 0.0, "end": 1.0, "description": "x", "spatial_region": "  left  "},
                ]
            ),
            ctx=VIDEO_CTX,
        )
        assert out["temporal_spatial_localization"][0]["spatial_region"] == "left"

    def test_non_string_spatial_region_rejected(self):
        with pytest.raises(DaftConvertError, match="spatial_region"):
            to_daft_msted(
                base_structured(
                    temporal_spatial_localization=[
                        {"start": 0.0, "end": 1.0, "description": "x", "spatial_region": 42},
                    ]
                ),
                ctx=VIDEO_CTX,
            )

    def test_empty_segment_list_rejected(self):
        with pytest.raises(DaftConvertError, match="non-empty"):
            to_daft_msted(
                base_structured(temporal_spatial_localization=[]),
                ctx=VIDEO_CTX,
            )

    def test_non_list_segments_rejected(self):
        with pytest.raises(DaftConvertError, match="must be a list"):
            to_daft_msted(
                base_structured(temporal_spatial_localization={}),
                ctx=VIDEO_CTX,
            )

    def test_segment_must_be_dict(self):
        with pytest.raises(DaftConvertError, match="segment\\[0\\]"):
            to_daft_msted(
                base_structured(temporal_spatial_localization=["not a dict"]),
                ctx=VIDEO_CTX,
            )

    def test_missing_segment_description_rejected(self):
        with pytest.raises(DaftConvertError, match="segment\\[0\\].description"):
            to_daft_msted(
                base_structured(
                    temporal_spatial_localization=[{"start": 0.0, "end": 1.0}],
                ),
                ctx=VIDEO_CTX,
            )

    def test_max_segments_enforced(self):
        many = [{"start": float(i), "end": float(i + 1), "description": f"s{i}"} for i in range(5)]
        with pytest.raises(DaftConvertError, match="exceeds max_segments"):
            to_daft_msted(
                base_structured(temporal_spatial_localization=many),
                ctx=VIDEO_CTX,
                max_segments=3,
            )

    def test_extra_keys_in_segment_dropped(self):
        # additionalProperties: false in the schema — the converter must
        # strip anything outside the known shape so the validator passes.
        out = to_daft_msted(
            base_structured(
                temporal_spatial_localization=[
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "description": "x",
                        "spatial_region": "left",
                        "extra_key": "should not appear",
                        "another": 99,
                    }
                ]
            ),
            ctx=VIDEO_CTX,
        )
        seg = out["temporal_spatial_localization"][0]
        assert set(seg.keys()) == {"start", "end", "description", "spatial_region"}


class TestEventDescription:
    def test_event_description_passes_through(self):
        out = to_daft_msted(base_structured(), ctx=VIDEO_CTX)
        ed = out["event_description"]
        assert ed["category"] == "lane change and turn"
        assert ed["description"].startswith("A sedan")
        assert ed["cause"] == "anticipated turn at the intersection"

    def test_use_case_agnostic_keys(self):
        # Schema allows ANY string-keyed/string-valued shape — the
        # converter never enforces a fixed taxonomy. This is the explicit
        # use-case-agnostic hook the user asked for.
        out = to_daft_msted(
            base_structured(
                event_description={
                    "domain_specific_field": "value1",
                    "another_custom_thing": "value2",
                }
            ),
            ctx=VIDEO_CTX,
        )
        assert out["event_description"] == {
            "domain_specific_field": "value1",
            "another_custom_thing": "value2",
        }

    def test_blank_values_dropped(self):
        out = to_daft_msted(
            base_structured(event_description={"category": "x", "blank": "   ", "filled": "y"}),
            ctx=VIDEO_CTX,
        )
        assert out["event_description"] == {"category": "x", "filled": "y"}

    def test_all_blank_event_description_rejected(self):
        with pytest.raises(DaftConvertError, match="empty after dropping"):
            to_daft_msted(
                base_structured(event_description={"a": "  ", "b": ""}),
                ctx=VIDEO_CTX,
            )

    def test_non_string_value_rejected(self):
        with pytest.raises(DaftConvertError, match="must be a string"):
            to_daft_msted(
                base_structured(event_description={"category": 123}),
                ctx=VIDEO_CTX,
            )

    def test_non_dict_event_description_rejected(self):
        with pytest.raises(DaftConvertError, match="must be a dict"):
            to_daft_msted(
                base_structured(event_description=["a", "b"]),
                ctx=VIDEO_CTX,
            )

    def test_empty_event_description_rejected(self):
        with pytest.raises(DaftConvertError, match="at least one key"):
            to_daft_msted(
                base_structured(event_description={}),
                ctx=VIDEO_CTX,
            )

    def test_blank_key_rejected(self):
        with pytest.raises(DaftConvertError, match="non-empty strings"):
            to_daft_msted(
                base_structured(event_description={"": "value"}),
                ctx=VIDEO_CTX,
            )


class TestSources:
    def test_string_sources_rejected(self):
        # Common mistake: passing a single string instead of a list.
        with pytest.raises(DaftConvertError, match="list of strings"):
            to_daft_msted(
                base_structured(),
                ctx=VIDEO_CTX,
                sources="sidecars/metadata.json",  # type: ignore[arg-type]
            )

    def test_non_iterable_sources_rejected(self):
        with pytest.raises(DaftConvertError, match="iterable"):
            to_daft_msted(
                base_structured(),
                ctx=VIDEO_CTX,
                sources=42,  # type: ignore[arg-type]
            )

    def test_blank_source_entry_rejected(self):
        with pytest.raises(DaftConvertError, match="empty after strip"):
            to_daft_msted(
                base_structured(),
                ctx=VIDEO_CTX,
                sources=["valid", "   "],
            )

    def test_empty_source_list_rejected(self):
        with pytest.raises(DaftConvertError, match="omit the field"):
            to_daft_msted(base_structured(), ctx=VIDEO_CTX, sources=[])

    def test_sources_strips_whitespace(self):
        out = to_daft_msted(
            base_structured(),
            ctx=VIDEO_CTX,
            sources=["  one  ", "two"],
        )
        assert out["sources"] == ["one", "two"]


class TestNonDictInput:
    def test_non_dict_structured_rejected(self):
        with pytest.raises(DaftConvertError, match="expects a dict"):
            to_daft_msted("not a dict", ctx=VIDEO_CTX)  # type: ignore[arg-type]


class TestKeyOrderingStability:
    """The schema doesn't care about key order, but stable ordering makes
    on-disk diffs readable — important when MSTED becomes reviewable
    artifact in datasets."""

    def test_envelope_keys_stable(self):
        out = to_daft_msted(
            base_structured(),
            ctx=VIDEO_CTX,
            sources=["sidecars/metadata.json"],
        )
        # version + envelope first, then payload fields in declared order.
        assert list(out.keys()) == [
            "version",
            "video_id",
            "metadata",
            "sources",
            "scene_description",
            "temporal_spatial_localization",
            "event_description",
        ]
