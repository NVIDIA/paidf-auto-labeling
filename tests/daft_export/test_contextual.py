# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re

import pytest
from daft_export.common import DAFT_VERSION, DaftConvertError
from daft_export.contextual import to_daft_events, to_daft_image, to_daft_video

# DAFT timecode regex; every start/end_time we emit must match.
TIMECODE_RE = re.compile(r"^(\d{2}:)?\d{2}:\d{2}(\.\d+)?$")

VIDEO_ID = "main"
IMAGE_ID = "main"


def al_video_dict(**overrides) -> dict:
    # Shape the auto-labeling VLM runner produces after `_normalize_video_json_obj`.
    base = {
        "video_id": "whatever-the-vlm-said",  # converter overrides with VIDEO_ID
        "format": "mp4",
        "fps": 29.97,
        "duration": 30.5,
        "height": 720,
        "width": 1280,
        "rectified": False,
        "scenario_info": "SURVEILLANCE_CAMERA",
        "scene_description": "...",
        "event_summary": "...",
        "source_video": "/input/clip.mp4",
        "generated_at": "2026-04-20T00:00:00Z",
    }
    base.update(overrides)
    return base


def al_event(**overrides) -> dict:
    base = {
        "event_id": "e1",
        "start_time": 1.0,
        "end_time": 4.5,
        "category": "collision",
        "sub_category": ["rear_end"],
        "instances": ["id_7"],
        "event_caption": "Vehicle A rear-ends vehicle B.",
    }
    base.update(overrides)
    return base


class TestVideo:
    def test_happy_path_only_allowed_keys(self):
        # Strips source_video/generated_at and overrides video_id with the
        # scene-canonical one.
        out = to_daft_video(al_video_dict(), video_id=VIDEO_ID)
        assert set(out.keys()) == {
            "version",
            "video_id",
            "format",
            "fps",
            "duration",
            "height",
            "width",
            "rectified",
            "scenario_info",
            "scene_description",
            "event_summary",
            "metadata",
        }
        assert out["version"] == DAFT_VERSION
        assert out["video_id"] == VIDEO_ID
        assert out["metadata"]["type"] == "video"

    def test_fps_float_rounded_to_int(self):
        assert to_daft_video(al_video_dict(fps=29.97), video_id=VIDEO_ID)["fps"] == 30
        assert to_daft_video(al_video_dict(fps=24.0), video_id=VIDEO_ID)["fps"] == 24

    def test_height_width_accept_whole_floats(self):
        # VLM occasionally emits 720.0 instead of 720.
        out = to_daft_video(al_video_dict(height=720.0, width=1280.0), video_id=VIDEO_ID)
        assert out["height"] == 720
        assert out["width"] == 1280

    def test_strips_pl_bookkeeping(self):
        out = to_daft_video(al_video_dict(), video_id=VIDEO_ID)
        assert "source_video" not in out
        assert "generated_at" not in out

    def test_strips_description_extra(self):
        # VLM sometimes leaks a top-level `description` — DAFT rejects it.
        out = to_daft_video(al_video_dict(description="blah"), video_id=VIDEO_ID)
        assert "description" not in out

    @pytest.mark.parametrize("fmt", ["avi", "mov", "mkv", "webm"])
    def test_format_enum_accepted(self, fmt):
        out = to_daft_video(al_video_dict(format=fmt), video_id=VIDEO_ID)
        assert out["format"] == fmt

    def test_bad_format_rejected(self):
        with pytest.raises(DaftConvertError, match="not in DAFT enum"):
            to_daft_video(al_video_dict(format="mxf"), video_id=VIDEO_ID)

    @pytest.mark.parametrize("bad_fps", [0, 241, -1, "thirty"])
    def test_bad_fps_rejected(self, bad_fps):
        with pytest.raises(DaftConvertError):
            to_daft_video(al_video_dict(fps=bad_fps), video_id=VIDEO_ID)

    def test_zero_duration_rejected(self):
        with pytest.raises(DaftConvertError, match="duration"):
            to_daft_video(al_video_dict(duration=0), video_id=VIDEO_ID)

    def test_fractional_dim_rejected(self):
        with pytest.raises(DaftConvertError, match="height"):
            to_daft_video(al_video_dict(height=720.5), video_id=VIDEO_ID)

    @pytest.mark.parametrize("missing", ["format", "fps", "duration", "height", "width"])
    def test_missing_required_field_raises(self, missing):
        d = al_video_dict()
        del d[missing]
        with pytest.raises(DaftConvertError, match=missing):
            to_daft_video(d, video_id=VIDEO_ID)


class TestEvents:
    def test_happy_path(self):
        out = to_daft_events({"events": [al_event()]}, video_id=VIDEO_ID, duration=10.0)
        assert out["version"] == DAFT_VERSION
        assert out["video_id"] == VIDEO_ID
        assert out["metadata"]["type"] == "events"
        [ev] = out["events"]
        assert TIMECODE_RE.fullmatch(ev["start_time"])
        assert TIMECODE_RE.fullmatch(ev["end_time"])
        assert ev["event_id"] == "e1"
        assert ev["event_caption"] == "Vehicle A rear-ends vehicle B."

    def test_accepts_bare_list(self):
        out = to_daft_events([al_event()], video_id=VIDEO_ID)
        assert len(out["events"]) == 1

    def test_empty_list(self):
        out = to_daft_events([], video_id=VIDEO_ID)
        assert out["events"] == []

    def test_seconds_converted_to_timecodes(self):
        out = to_daft_events([al_event(start_time=0.0, end_time=95.5)], video_id=VIDEO_ID)
        [ev] = out["events"]
        assert ev["start_time"] == "00:00"
        assert ev["end_time"] == "01:35.500"

    def test_accepts_seconds_aliases(self):
        ev_in = al_event()
        del ev_in["start_time"]
        del ev_in["end_time"]
        ev_in["start_time_sec"] = 2.0
        ev_in["end_time_sec"] = 5.0
        out = to_daft_events(
            [ev_in],
            video_id=VIDEO_ID,
            duration=6.0,
        )
        [ev] = out["events"]
        assert ev["start_time"] == "00:02"
        assert ev["end_time"] == "00:05"

    def test_clamping_to_duration(self):
        # end_time past duration gets clamped.
        out = to_daft_events(
            [al_event(start_time=-1.0, end_time=999.0)],
            video_id=VIDEO_ID,
            duration=10.0,
        )
        [ev] = out["events"]
        assert ev["start_time"] == "00:00"
        assert ev["end_time"] == "00:10"

    def test_description_remapped_to_event_caption(self):
        # Older prompts or upstream code that still emits `description`.
        ev_in = al_event()
        del ev_in["event_caption"]
        ev_in["description"] = "older caption"
        out = to_daft_events([ev_in], video_id=VIDEO_ID)
        [ev] = out["events"]
        assert ev["event_caption"] == "older caption"
        assert "description" not in ev

    def test_event_caption_wins_over_description(self):
        ev_in = al_event(description="older")
        out = to_daft_events([ev_in], video_id=VIDEO_ID)
        [ev] = out["events"]
        assert ev["event_caption"] == "Vehicle A rear-ends vehicle B."
        assert "description" not in ev

    def test_strips_per_event_extras(self):
        ev_in = al_event(_debug="trace", confidence=0.92)
        out = to_daft_events([ev_in], video_id=VIDEO_ID)
        [ev] = out["events"]
        assert "_debug" not in ev
        assert "confidence" not in ev

    def test_accepts_string_timecodes_too(self):
        # Idempotent: feeding already-formatted timecodes round-trips cleanly.
        ev_in = al_event(start_time="00:01.500", end_time="00:04.500")
        out = to_daft_events([ev_in], video_id=VIDEO_ID)
        [ev] = out["events"]
        assert ev["start_time"] == "00:01.500"
        assert ev["end_time"] == "00:04.500"

    @pytest.mark.parametrize("missing", ["event_id", "start_time", "end_time"])
    def test_missing_required_field_raises(self, missing):
        ev_in = al_event()
        del ev_in[missing]
        with pytest.raises(DaftConvertError, match=missing):
            to_daft_events([ev_in], video_id=VIDEO_ID)

    def test_bad_timestamp_raises(self):
        with pytest.raises(DaftConvertError, match="start_time"):
            to_daft_events([al_event(start_time="nope")], video_id=VIDEO_ID)


class TestEventsIdTranslation:
    """``to_daft_events`` runs the id translator when ``instances_keys`` is
    given. These tests verify the wiring; the translator's own behavior is
    covered exhaustively in ``test_id_translator``."""

    def test_no_instances_keys_passes_through(self):
        # Backward-compat path: callers without a tracker catalogue (tests,
        # older code) should see the VLM's ``id_<n>`` shape preserved.
        out = to_daft_events(
            [al_event(instances=["id_7", "id_42"])],
            video_id=VIDEO_ID,
        )
        [ev] = out["events"]
        assert ev["instances"] == ["id_7", "id_42"]

    def test_translates_with_instances_keys(self):
        out = to_daft_events(
            [al_event(instances=["id_7", "id_42"])],
            video_id=VIDEO_ID,
            instances_keys=["car_7", "motorcycle_42", "car_99"],
        )
        [ev] = out["events"]
        assert ev["instances"] == ["car_7", "motorcycle_42"]

    def test_translates_bare_numeric_instance_refs(self):
        out = to_daft_events(
            [al_event(instances=["007", 42])],
            video_id=VIDEO_ID,
            instances_keys=["car_7", "motorcycle_42"],
        )
        [ev] = out["events"]
        assert ev["instances"] == ["car_7", "motorcycle_42"]

    def test_drops_ungrounded_ids(self):
        # Mismatch case: id_8 has no <class>_8 in the catalogue.
        out = to_daft_events(
            [al_event(instances=["id_7", "id_8"])],
            video_id=VIDEO_ID,
            instances_keys=["car_7"],
        )
        [ev] = out["events"]
        assert ev["instances"] == ["car_7"]

    def test_empty_instances_keys_drops_all(self):
        # Catalogue empty (e.g. det/track disabled or produced no instances);
        # nothing the VLM emitted can be grounded.
        out = to_daft_events(
            [al_event(instances=["id_6131", "id_6127"])],
            video_id=VIDEO_ID,
            instances_keys=[],
        )
        [ev] = out["events"]
        assert ev["instances"] == []

    def test_multiple_events_share_one_index(self):
        # Internal optimization is invisible to the caller, but the per-event
        # translation must still produce the right per-event result.
        out = to_daft_events(
            [
                al_event(event_id="e1", instances=["id_7"]),
                al_event(event_id="e2", instances=["id_42", "id_999"]),
            ],
            video_id=VIDEO_ID,
            instances_keys=["car_7", "motorcycle_42"],
        )
        evs = out["events"]
        assert evs[0]["instances"] == ["car_7"]
        assert evs[1]["instances"] == ["motorcycle_42"]

    def test_event_without_instances_field_is_untouched(self):
        ev_in = al_event()
        del ev_in["instances"]
        out = to_daft_events(
            [ev_in],
            video_id=VIDEO_ID,
            instances_keys=["car_7"],
        )
        [ev] = out["events"]
        assert "instances" not in ev


class TestEventCaptionStripping:
    """``to_daft_events`` removes ungrounded ``{id: <n>}`` from ``event_caption`` when
    ``instances_keys`` is provided. Wires
    :func:`strip_ungrounded_id_annotations` against the same catalogue used
    for structured grounding so a number dropped from ``instances`` is also
    dropped from any ``{id: <n>}`` clause in the caption."""

    def test_strips_ungrounded_caption_with_keys(self):
        # Mixed leak: ``{id: 7}`` is real, ``{id: 6131}`` is a few-shot
        # example number — only the latter is removed.
        out = to_daft_events(
            [
                al_event(
                    instances=["id_7"],
                    event_caption="sedan {id: 7} hits motorcycle {id: 6131}",
                )
            ],
            video_id=VIDEO_ID,
            instances_keys=["car_7"],
        )
        [ev] = out["events"]
        assert ev["event_caption"] == "sedan {id: 7} hits motorcycle"

    def test_caption_unchanged_without_instances_keys(self):
        # Cleanup is opt-in via ``instances_keys`` to keep test fixtures
        # and older callers untouched.
        ev_in = al_event(event_caption="sedan {id: 6131} fabricated")
        out = to_daft_events([ev_in], video_id=VIDEO_ID)
        [ev] = out["events"]
        assert ev["event_caption"] == "sedan {id: 6131} fabricated"

    def test_empty_catalogue_strips_all_caption_ids(self):
        # det/track OFF case at the integration boundary: every ``{id: <n>}``
        # in the caption is necessarily a leak.
        out = to_daft_events(
            [
                al_event(
                    instances=["id_6131"],
                    event_caption="vehicle {id: 6131} and motorcycle {id: 6127}",
                )
            ],
            video_id=VIDEO_ID,
            instances_keys=[],
        )
        [ev] = out["events"]
        assert ev["event_caption"] == "vehicle and motorcycle"
        assert "{id:" not in ev["event_caption"]


class TestVideoProseStripping:
    """``to_daft_video`` removes ungrounded ``{id: <n>}`` from ``scene_description`` and
    ``event_summary`` when ``instances_keys`` is provided. Mirror of
    :class:`TestEventCaptionStripping` for the video.json side."""

    def test_strips_ungrounded_in_scene_description(self):
        out = to_daft_video(
            al_video_dict(scene_description="Wide view; sedan {id: 6131} parked."),
            video_id=VIDEO_ID,
            instances_keys=["car_7"],
        )
        assert out["scene_description"] == "Wide view; sedan parked."

    def test_strips_ungrounded_in_event_summary(self):
        out = to_daft_video(
            al_video_dict(event_summary="Motorcycle {id: 6127} hit by sedan."),
            video_id=VIDEO_ID,
            instances_keys=["car_7"],
        )
        assert out["event_summary"] == "Motorcycle hit by sedan."

    def test_keeps_grounded_in_prose(self):
        out = to_daft_video(
            al_video_dict(
                scene_description="sedan {id: 7} parked at corner",
                event_summary="vehicle {id: 7} stationary",
            ),
            video_id=VIDEO_ID,
            instances_keys=["car_7"],
        )
        assert out["scene_description"] == "sedan {id: 7} parked at corner"
        assert out["event_summary"] == "vehicle {id: 7} stationary"

    def test_prose_unchanged_without_instances_keys(self):
        out = to_daft_video(
            al_video_dict(
                scene_description="sedan {id: 6131} parked",
                event_summary="motorcycle {id: 6127} moving",
            ),
            video_id=VIDEO_ID,
        )
        assert out["scene_description"] == "sedan {id: 6131} parked"
        assert out["event_summary"] == "motorcycle {id: 6127} moving"

    def test_empty_catalogue_strips_all_prose_ids(self):
        out = to_daft_video(
            al_video_dict(
                scene_description="sedan {id: 6131} parked",
                event_summary="motorcycle {id: 6127} moving",
            ),
            video_id=VIDEO_ID,
            instances_keys=[],
        )
        assert out["scene_description"] == "sedan parked"
        assert out["event_summary"] == "motorcycle moving"


def al_image_dict(**overrides) -> dict:
    # Shape the auto-labeling image-input runner produces after `_enrich_image_with_probe`:
    # video schema's scene_description/event_summary collapse into `caption`.
    base = {
        "format": "jpg",
        "height": 1080,
        "width": 1920,
        "rectified": False,
        "scenario_info": "SURVEILLANCE_CAMERA",
        "caption": "A red car at an intersection.",
        "source_video": "/input/frame.jpg",  # auto-labeling bookkeeping; must be stripped
        "generated_at": "2026-04-25T00:00:00Z",
    }
    base.update(overrides)
    return base


class TestImage:
    def test_happy_path_only_allowed_keys(self):
        out = to_daft_image(al_image_dict(), image_id=IMAGE_ID)
        assert set(out.keys()) == {
            "version",
            "image_id",
            "format",
            "height",
            "width",
            "rectified",
            "scenario_info",
            "caption",
            "metadata",
        }
        assert out["version"] == DAFT_VERSION
        assert out["image_id"] == IMAGE_ID
        assert out["metadata"]["type"] == "image"
        # No video-only fields leaked through.
        assert "fps" not in out
        assert "duration" not in out
        assert "video_id" not in out

    def test_strips_pl_bookkeeping(self):
        out = to_daft_image(al_image_dict(), image_id=IMAGE_ID)
        assert "source_video" not in out
        assert "generated_at" not in out

    @pytest.mark.parametrize("fmt", ["png", "jpg", "jpeg", "bmp", "tiff", "webp"])
    def test_format_enum_accepted(self, fmt):
        out = to_daft_image(al_image_dict(format=fmt), image_id=IMAGE_ID)
        assert out["format"] == fmt

    @pytest.mark.parametrize("bad_fmt", ["mp4", "gif", "tif", ""])
    def test_bad_format_rejected(self, bad_fmt):
        with pytest.raises(DaftConvertError, match="not in DAFT enum"):
            to_daft_image(al_image_dict(format=bad_fmt), image_id=IMAGE_ID)

    def test_height_width_accept_whole_floats(self):
        out = to_daft_image(al_image_dict(height=1080.0, width=1920.0), image_id=IMAGE_ID)
        assert out["height"] == 1080
        assert out["width"] == 1920

    def test_fractional_dim_rejected(self):
        with pytest.raises(DaftConvertError, match="height"):
            to_daft_image(al_image_dict(height=1080.5), image_id=IMAGE_ID)

    @pytest.mark.parametrize("missing", ["format", "height", "width"])
    def test_missing_required_field_raises(self, missing):
        d = al_image_dict()
        del d[missing]
        # `format=None` → enum rejection; missing height/width → numeric coercion.
        with pytest.raises(DaftConvertError):
            to_daft_image(d, image_id=IMAGE_ID)

    def test_optional_passthroughs_only_when_present(self):
        out = to_daft_image(
            {"format": "png", "height": 100, "width": 100},
            image_id=IMAGE_ID,
        )
        for k in ("rectified", "scenario_info", "caption", "timestamp"):
            assert k not in out

    def test_image_id_overrides_anything_in_payload(self):
        # The converter is the source of truth for the scene-anchor id.
        out = to_daft_image(
            {**al_image_dict(), "image_id": "whatever-the-vlm-said"},
            image_id=IMAGE_ID,
        )
        assert out["image_id"] == IMAGE_ID
