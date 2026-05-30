# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from daft_export.common import DAFT_VERSION, DaftConvertError
from daft_export.tracking import to_daft_instances, to_daft_objects

VIDEO_ID = "main"


def al_instance(**overrides) -> dict:
    # Shape the auto-labeling rfdetr_tracking produces per instance.
    base = {
        "object_type": "car",
        "instance_id": 1,
        "semantic_id": 2,
        "color": [255, 128, 0],
        "caption": "car (track 7)",
        "track_id": 7,
        "first_frame": 0,
        "last_frame": 29,
        "confidence_avg": 0.85,
        "frame_count": 30,
    }
    base.update(overrides)
    return base


def al_detection(**overrides) -> dict:
    base = {
        "object_id": "car_7",
        "instance_id": 1,
        "semantic_id": 2,
        "bounding_box_2d_tight": [10.0, 20.0, 100.0, 200.0],
        "bounding_box_2d_loose": [5.0, 15.0, 105.0, 205.0],
        "confidence": 0.95,
    }
    base.update(overrides)
    return base


def al_frame(**overrides) -> dict:
    base = {
        "format": "png",
        "frame_number": 1,
        "width": 1920,
        "height": 1080,
        "instances": [al_detection()],
        "detection_count": 1,
    }
    base.update(overrides)
    return base


def al_instances_dict(instances: dict | None = None) -> dict:
    return {
        "version": "al-internal",
        "video_info": {"source": "/in.mp4", "fps": 30, "width": 1920, "height": 1080, "total_frames": 30},
        "instances": instances if instances is not None else {"car_7": al_instance()},
    }


def al_objects_dict(frames: dict | None = None) -> dict:
    return {
        "version": "al-internal",
        "frames": frames if frames is not None else {"frame_000001": al_frame()},
    }


class TestInstances:
    def test_envelope(self):
        # No video_id at top level: instances.json is a scene-level catalog and
        # DAFT's additionalProperties:false would reject it.
        out = to_daft_instances(al_instances_dict())
        assert out["version"] == DAFT_VERSION
        assert out["metadata"]["type"] == "instances"
        assert set(out.keys()) == {"version", "instances", "metadata"}

    def test_keeps_daft_fields_only(self):
        # Exact key set is the contract; tripwire on auto-labeling-internal leaks
        # (track_id/first_frame/... ) and accidental drops of DAFT fields.
        out = to_daft_instances(al_instances_dict())
        inst = out["instances"]["car_7"]
        assert set(inst.keys()) == {
            "object_type",
            "instance_id",
            "semantic_id",
            "color",
            "caption",
        }

    def test_omits_unset_optionals(self):
        bare = al_instance()
        for f in ("color", "caption"):
            del bare[f]
        out = to_daft_instances(al_instances_dict({"car_7": bare}))
        inst = out["instances"]["car_7"]
        assert "color" not in inst
        assert "caption" not in inst

    def test_empty_instances_dict(self):
        out = to_daft_instances(al_instances_dict({}))
        assert out["instances"] == {}

    def test_missing_instances_dict_raises(self):
        with pytest.raises(DaftConvertError, match="must be a dict"):
            to_daft_instances({})

    @pytest.mark.parametrize("missing", ["object_type", "instance_id", "semantic_id"])
    def test_missing_required_field_raises(self, missing):
        entry = al_instance()
        del entry[missing]
        with pytest.raises(DaftConvertError, match=missing):
            to_daft_instances(al_instances_dict({"car_7": entry}))

    @pytest.mark.parametrize("bad", [-1, 0.5, "1", True, None])
    def test_bad_instance_id_rejected(self, bad):
        entry = al_instance(instance_id=bad)
        with pytest.raises(DaftConvertError, match="instance_id"):
            to_daft_instances(al_instances_dict({"car_7": entry}))

    def test_non_string_object_type_rejected(self):
        entry = al_instance(object_type=7)
        with pytest.raises(DaftConvertError, match="object_type"):
            to_daft_instances(al_instances_dict({"car_7": entry}))


class TestObjects:
    def test_envelope(self):
        out = to_daft_objects(al_objects_dict(), video_id=VIDEO_ID)
        assert out["version"] == DAFT_VERSION
        assert out["video_id"] == VIDEO_ID
        assert out["metadata"]["type"] == "objects"
        assert set(out.keys()) == {"version", "video_id", "frames", "metadata"}

    def test_frame_strips_bookkeeping(self):
        out = to_daft_objects(al_objects_dict(), video_id=VIDEO_ID)
        frame = out["frames"]["frame_000001"]
        assert set(frame.keys()) == {"format", "frame_number", "instances"}

    def test_detection_strips_bookkeeping(self):
        out = to_daft_objects(al_objects_dict(), video_id=VIDEO_ID)
        [det] = out["frames"]["frame_000001"]["instances"]
        # instance_id/semantic_id/confidence are recovered via cross-ref to
        # instances.json; they must not be duplicated here.
        assert set(det.keys()) == {"object_id", "bounding_box_2d_tight", "bounding_box_2d_loose"}

    def test_detection_without_loose_bbox(self):
        det_in = al_detection()
        del det_in["bounding_box_2d_loose"]
        frame = al_frame(instances=[det_in])
        out = to_daft_objects(al_objects_dict({"f1": frame}), video_id=VIDEO_ID)
        [det] = out["frames"]["f1"]["instances"]
        assert set(det.keys()) == {"object_id", "bounding_box_2d_tight"}

    def test_empty_frame_instances(self):
        frame = al_frame(instances=[])
        out = to_daft_objects(al_objects_dict({"f1": frame}), video_id=VIDEO_ID)
        assert out["frames"]["f1"]["instances"] == []

    def test_empty_frames_dict(self):
        out = to_daft_objects(al_objects_dict({}), video_id=VIDEO_ID)
        assert out["frames"] == {}

    def test_missing_frames_dict_raises(self):
        with pytest.raises(DaftConvertError, match="must be a dict"):
            to_daft_objects({}, video_id=VIDEO_ID)

    @pytest.mark.parametrize("missing", ["format", "frame_number", "instances"])
    def test_frame_missing_required_field_raises(self, missing):
        frame = al_frame()
        del frame[missing]
        with pytest.raises(DaftConvertError, match=missing):
            to_daft_objects(al_objects_dict({"f1": frame}), video_id=VIDEO_ID)

    @pytest.mark.parametrize("missing", ["object_id", "bounding_box_2d_tight"])
    def test_detection_missing_required_field_raises(self, missing):
        det = al_detection()
        del det[missing]
        frame = al_frame(instances=[det])
        with pytest.raises(DaftConvertError, match=missing):
            to_daft_objects(al_objects_dict({"f1": frame}), video_id=VIDEO_ID)

    @pytest.mark.parametrize("fmt", ["jpg", "jpeg", "bmp"])
    def test_allowed_frame_formats(self, fmt):
        frame = al_frame(format=fmt)
        out = to_daft_objects(al_objects_dict({"f1": frame}), video_id=VIDEO_ID)
        assert out["frames"]["f1"]["format"] == fmt

    def test_rejects_video_formats(self):
        # mp4 is allowed by video.json's format enum but NOT by objects.json's.
        frame = al_frame(format="mp4")
        with pytest.raises(DaftConvertError, match="format"):
            to_daft_objects(al_objects_dict({"f1": frame}), video_id=VIDEO_ID)

    def test_rejects_non_list_instances(self):
        frame = al_frame(instances={"not": "a list"})
        with pytest.raises(DaftConvertError, match="must be a list"):
            to_daft_objects(al_objects_dict({"f1": frame}), video_id=VIDEO_ID)
