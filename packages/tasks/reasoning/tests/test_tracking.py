# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# mypy: disable-error-code="no-untyped-def,unused-ignore"

from __future__ import annotations

import pytest
from reasoning.common import DAFT_VERSION, DaftConvertError, SceneContext
from reasoning.tracking import to_daft_instances, to_daft_objects, to_daft_tracking

VIDEO_CTX = SceneContext(media_id="main", iso_date="2026-04-20")
IMAGE_CTX = SceneContext(media_id="main", is_image=True, iso_date="2026-04-20")


def pl_instance(**overrides) -> dict:
    # Shape PL's rfdetr_tracking produces per instance.
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


def pl_detection(**overrides) -> dict:
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


def pl_frame(**overrides) -> dict:
    base = {
        "format": "png",
        "frame_number": 1,
        "width": 1920,
        "height": 1080,
        "instances": [pl_detection()],
        "detection_count": 1,
    }
    base.update(overrides)
    return base


def pl_instances_dict(instances: dict | None = None) -> dict:
    return {
        "version": "pl-internal",
        "video_info": {
            "source": "/in.mp4",
            "fps": 30,
            "width": 1920,
            "height": 1080,
            "total_frames": 30,
        },
        "instances": instances if instances is not None else {"car_7": pl_instance()},
    }


def pl_objects_dict(frames: dict | None = None) -> dict:
    return {
        "version": "pl-internal",
        "frames": frames if frames is not None else {"frame_000001": pl_frame()},
    }


class TestInstances:
    def test_envelope(self):
        # No video_id at top level: instances.json is a scene-level catalog and
        # DAFT's additionalProperties:false would reject it.
        out = to_daft_instances(pl_instances_dict(), ctx=VIDEO_CTX)
        assert out["version"] == DAFT_VERSION
        assert out["metadata"]["type"] == "instances"
        assert set(out.keys()) == {"version", "instances", "metadata"}

    def test_keeps_daft_fields_only(self):
        # Exact key set is the contract; tripwire on PL-internal leaks
        # (track_id/first_frame/... ) and accidental drops of DAFT fields.
        out = to_daft_instances(pl_instances_dict(), ctx=VIDEO_CTX)
        inst = out["instances"]["car_7"]
        assert set(inst.keys()) == {
            "object_type",
            "instance_id",
            "semantic_id",
            "color",
            "caption",
        }

    def test_omits_unset_optionals(self):
        bare = pl_instance()
        for f in ("color", "caption"):
            del bare[f]
        out = to_daft_instances(pl_instances_dict({"car_7": bare}), ctx=VIDEO_CTX)
        inst = out["instances"]["car_7"]
        assert "color" not in inst
        assert "caption" not in inst

    def test_empty_instances_dict(self):
        out = to_daft_instances(pl_instances_dict({}), ctx=VIDEO_CTX)
        assert out["instances"] == {}

    def test_missing_instances_dict_raises(self):
        with pytest.raises(DaftConvertError, match="must be a dict"):
            to_daft_instances({}, ctx=VIDEO_CTX)

    @pytest.mark.parametrize("missing", ["object_type", "instance_id", "semantic_id"])
    def test_missing_required_field_raises(self, missing):
        entry = pl_instance()
        del entry[missing]
        with pytest.raises(DaftConvertError, match=missing):
            to_daft_instances(pl_instances_dict({"car_7": entry}), ctx=VIDEO_CTX)

    @pytest.mark.parametrize("bad", [-1, 0.5, "1", True, None])
    def test_bad_instance_id_rejected(self, bad):
        entry = pl_instance(instance_id=bad)
        with pytest.raises(DaftConvertError, match="instance_id"):
            to_daft_instances(pl_instances_dict({"car_7": entry}), ctx=VIDEO_CTX)

    def test_non_string_object_type_rejected(self):
        entry = pl_instance(object_type=7)
        with pytest.raises(DaftConvertError, match="object_type"):
            to_daft_instances(pl_instances_dict({"car_7": entry}), ctx=VIDEO_CTX)

    def test_image_ctx_succeeds_and_auto_binds_images(self):
        """``instances.schema.json`` defines a per-instance ``images``
        array (parallel to ``videos``) precisely so an image scene's
        instances.json can bind back to the originating image(s). For
        image scenes that don't pre-supply ``images`` / ``videos``,
        the converter auto-binds ``images=[ctx.media_id]``.

        Validator nuance: default ``tao-daft validate --raw image
        --strict`` ignores ``instances.json`` (it's not in the default
        contextual set ``[objects, tracking]``), so emitting it never
        costs default validation. Only explicit
        ``--contextual instances`` rejects it — see
        ``test_validate_scene.py`` for the empirical CLI contract.
        """
        out = to_daft_instances(pl_instances_dict(), ctx=IMAGE_CTX)
        assert out["version"] == DAFT_VERSION
        assert out["metadata"]["type"] == "instances"
        # Top-level shape matches the video flavor — no scene-id field.
        assert set(out.keys()) == {"version", "instances", "metadata"}
        inst = out["instances"]["car_7"]
        # Auto-bound to the image scene via the schema's ``images`` field.
        assert inst.get("images") == [IMAGE_CTX.media_id], (
            "image-scene instance must auto-bind to ``images=[media_id]`` "
            "so the DAFT cross-reference is explicit"
        )
        # No accidental ``videos`` binding for image scenes.
        assert "videos" not in inst

    def test_image_ctx_preserves_caller_supplied_images(self):
        """Caller-supplied ``images`` (e.g. multi-image scenes) is the
        source of truth — the converter must not overwrite it."""
        entry = pl_instance(images=["scene_a", "scene_b"])
        out = to_daft_instances(pl_instances_dict({"car_7": entry}), ctx=IMAGE_CTX)
        assert out["instances"]["car_7"]["images"] == ["scene_a", "scene_b"]

    def test_image_ctx_preserves_caller_supplied_videos(self):
        """If caller already bound the instance via ``videos`` (e.g. a
        composite image-+-video scene), don't auto-add ``images``."""
        entry = pl_instance(videos=["camera_01"])
        out = to_daft_instances(pl_instances_dict({"car_7": entry}), ctx=IMAGE_CTX)
        inst = out["instances"]["car_7"]
        assert inst["videos"] == ["camera_01"]
        assert "images" not in inst, "explicit ``videos`` binding suppresses the image auto-bind"

    def test_video_ctx_does_not_auto_bind(self):
        """Video scenes already have ``video_id`` at the scene-tree
        level via ``video.json``. Don't perturb the existing video
        contract by adding a per-instance ``videos`` array — caller
        opt-in only."""
        out = to_daft_instances(pl_instances_dict(), ctx=VIDEO_CTX)
        inst = out["instances"]["car_7"]
        assert "images" not in inst
        assert "videos" not in inst


class TestObjects:
    def test_envelope(self):
        out = to_daft_objects(pl_objects_dict(), ctx=VIDEO_CTX)
        assert out["version"] == DAFT_VERSION
        assert out["video_id"] == VIDEO_CTX.media_id
        assert out["metadata"]["type"] == "objects"
        assert set(out.keys()) == {"version", "video_id", "frames", "metadata"}

    def test_image_ctx_succeeds_without_scene_id(self):
        """Image scenes are first-class for objects.json: ``tao-daft
        validate --raw image`` lists ``objects`` in its default
        contextual set. The schema's ``video_id`` is optional and
        ``additionalProperties: false`` would reject ``image_id``, so
        the converter omits the scene-id field entirely for images."""
        out = to_daft_objects(pl_objects_dict(), ctx=IMAGE_CTX)
        assert out["version"] == DAFT_VERSION
        assert out["metadata"]["type"] == "objects"
        # Neither scene-id field appears.
        assert "video_id" not in out, (
            "image-scene objects.json must omit video_id (no video to "
            "reference; would mislead downstream consumers)"
        )
        assert "image_id" not in out, (
            "image-scene objects.json must omit image_id (the schema's "
            "additionalProperties: false would reject it)"
        )
        # And the per-frame payload still flows through unchanged.
        assert isinstance(out["frames"], dict)
        assert set(out["frames"]["frame_000001"].keys()) == {"format", "frame_number", "instances"}

    def test_frame_strips_bookkeeping(self):
        out = to_daft_objects(pl_objects_dict(), ctx=VIDEO_CTX)
        frame = out["frames"]["frame_000001"]
        assert set(frame.keys()) == {"format", "frame_number", "instances"}

    def test_detection_strips_bookkeeping(self):
        out = to_daft_objects(pl_objects_dict(), ctx=VIDEO_CTX)
        [det] = out["frames"]["frame_000001"]["instances"]
        # instance_id/semantic_id/confidence are recovered via cross-ref to
        # instances.json; they must not be duplicated here.
        assert set(det.keys()) == {"object_id", "bounding_box_2d_tight", "bounding_box_2d_loose"}

    def test_detection_without_loose_bbox(self):
        det_in = pl_detection()
        del det_in["bounding_box_2d_loose"]
        frame = pl_frame(instances=[det_in])
        out = to_daft_objects(pl_objects_dict({"f1": frame}), ctx=VIDEO_CTX)
        [det] = out["frames"]["f1"]["instances"]
        assert set(det.keys()) == {"object_id", "bounding_box_2d_tight"}

    def test_empty_frame_instances(self):
        frame = pl_frame(instances=[])
        out = to_daft_objects(pl_objects_dict({"f1": frame}), ctx=VIDEO_CTX)
        assert out["frames"]["f1"]["instances"] == []

    def test_empty_frames_dict(self):
        out = to_daft_objects(pl_objects_dict({}), ctx=VIDEO_CTX)
        assert out["frames"] == {}

    def test_missing_frames_dict_raises(self):
        with pytest.raises(DaftConvertError, match="must be a dict"):
            to_daft_objects({}, ctx=VIDEO_CTX)

    @pytest.mark.parametrize("missing", ["format", "frame_number", "instances"])
    def test_frame_missing_required_field_raises(self, missing):
        frame = pl_frame()
        del frame[missing]
        with pytest.raises(DaftConvertError, match=missing):
            to_daft_objects(pl_objects_dict({"f1": frame}), ctx=VIDEO_CTX)

    @pytest.mark.parametrize("missing", ["object_id", "bounding_box_2d_tight"])
    def test_detection_missing_required_field_raises(self, missing):
        det = pl_detection()
        del det[missing]
        frame = pl_frame(instances=[det])
        with pytest.raises(DaftConvertError, match=missing):
            to_daft_objects(pl_objects_dict({"f1": frame}), ctx=VIDEO_CTX)

    @pytest.mark.parametrize("fmt", ["jpg", "jpeg", "bmp"])
    def test_allowed_frame_formats(self, fmt):
        frame = pl_frame(format=fmt)
        out = to_daft_objects(pl_objects_dict({"f1": frame}), ctx=VIDEO_CTX)
        assert out["frames"]["f1"]["format"] == fmt

    def test_rejects_video_formats(self):
        # mp4 is allowed by video.json's format enum but NOT by objects.json's.
        frame = pl_frame(format="mp4")
        with pytest.raises(DaftConvertError, match="format"):
            to_daft_objects(pl_objects_dict({"f1": frame}), ctx=VIDEO_CTX)

    def test_rejects_non_list_instances(self):
        frame = pl_frame(instances={"not": "a list"})
        with pytest.raises(DaftConvertError, match="must be a list"):
            to_daft_objects(pl_objects_dict({"f1": frame}), ctx=VIDEO_CTX)

    def test_instances_source_threaded_from_ctx(self):
        ctx = SceneContext(media_id="main", instances_source="instances_auto.json")
        out = to_daft_objects(pl_objects_dict(), ctx=ctx)
        assert out["instances_source"] == "instances_auto.json"


def track_det(**overrides) -> dict:
    """Minimal valid 4D-MOT detection: object_id + 3d_location."""
    base = {"object_id": "car_1", "3d_location": [5.2, 10.3, 0.0]}
    base.update(overrides)
    return base


def track_input(frames: dict | None = None) -> dict:
    if frames is None:
        frames = {"frame_000001": [track_det()]}
    return {"frames": frames}


class TestTrackingEnvelope:
    def test_envelope(self):
        out = to_daft_tracking(track_input(), ctx=VIDEO_CTX)
        assert out["version"] == DAFT_VERSION
        assert out["metadata"]["type"] == "tracking"
        assert out["metadata"]["date"] == "2026-04-20"
        # Top-level scene id is omitted; instances catalogue cross-reference is the link.
        assert "video_id" not in out and "image_id" not in out

    def test_image_ctx_rejected(self):
        with pytest.raises(DaftConvertError, match="image"):
            to_daft_tracking(track_input(), ctx=IMAGE_CTX)

    def test_metadata_threads_optional_fields(self):
        ctx = SceneContext(media_id="main", license_str="CC-BY-4.0", tags=("test",))
        out = to_daft_tracking(track_input(), ctx=ctx)
        assert out["metadata"]["license"] == "CC-BY-4.0"
        assert out["metadata"]["tags"] == ["test"]


class TestTrackingFrames:
    def test_minimal_detection(self):
        out = to_daft_tracking(track_input(), ctx=VIDEO_CTX)
        [det] = out["frames"]["frame_000001"]
        assert det == {"object_id": "car_1", "3d_location": [5.2, 10.3, 0.0]}

    def test_optional_3d_box_fields(self):
        det = track_det(
            **{
                "3d_bounding_box_scale": [2.0, 3.5, 2.0],
                "3d_bounding_box_rotation": [0.0, 0.0, 1.57],
            }
        )
        out = to_daft_tracking(track_input({"f1": [det]}), ctx=VIDEO_CTX)
        [out_det] = out["frames"]["f1"]
        assert out_det["3d_bounding_box_scale"] == [2.0, 3.5, 2.0]
        assert out_det["3d_bounding_box_rotation"] == [0.0, 0.0, 1.57]

    def test_2d_visible_passthrough(self):
        det = track_det(
            **{
                "2d_bounding_box_visible": {
                    "cam_001": [100, 200, 300, 500],
                    "cam_002": [450.5, 320.8, 680.2, 550.1],
                }
            }
        )
        out = to_daft_tracking(track_input({"f1": [det]}), ctx=VIDEO_CTX)
        [out_det] = out["frames"]["f1"]
        # All numbers coerced to float; key set preserved.
        assert out_det["2d_bounding_box_visible"]["cam_001"] == [100.0, 200.0, 300.0, 500.0]
        assert out_det["2d_bounding_box_visible"]["cam_002"] == [450.5, 320.8, 680.2, 550.1]

    def test_extra_per_detection_keys_stripped(self):
        # additionalProperties: false at the detection level — no PL-internal leakage.
        det = track_det(track_id=7, confidence=0.9, _debug="x")
        out = to_daft_tracking(track_input({"f1": [det]}), ctx=VIDEO_CTX)
        [out_det] = out["frames"]["f1"]
        assert set(out_det.keys()) == {"object_id", "3d_location"}

    def test_multi_frame_multi_object(self):
        frames = {
            "frame_000001": [
                track_det(object_id="car_1", **{"3d_location": [1.0, 2.0, 0.0]}),
                track_det(object_id="worker_1", **{"3d_location": [3.0, 4.0, 0.0]}),
            ],
            "frame_000002": [
                track_det(object_id="car_1", **{"3d_location": [1.5, 2.0, 0.0]}),
            ],
        }
        out = to_daft_tracking(track_input(frames), ctx=VIDEO_CTX)
        assert len(out["frames"]) == 2
        assert len(out["frames"]["frame_000001"]) == 2
        assert len(out["frames"]["frame_000002"]) == 1


class TestTrackingErrors:
    def test_frames_must_be_dict(self):
        with pytest.raises(DaftConvertError, match="must be a dict"):
            to_daft_tracking({"frames": []}, ctx=VIDEO_CTX)

    def test_frame_value_must_be_list(self):
        with pytest.raises(DaftConvertError, match="list of detections"):
            to_daft_tracking({"frames": {"f1": {"not": "a list"}}}, ctx=VIDEO_CTX)

    def test_detection_must_be_dict(self):
        with pytest.raises(DaftConvertError, match="must be a dict"):
            to_daft_tracking({"frames": {"f1": ["not a dict"]}}, ctx=VIDEO_CTX)

    @pytest.mark.parametrize("missing", ["object_id", "3d_location"])
    def test_missing_required_field(self, missing):
        det = track_det()
        del det[missing]
        with pytest.raises(DaftConvertError, match=missing):
            to_daft_tracking(track_input({"f1": [det]}), ctx=VIDEO_CTX)

    def test_object_id_must_be_string(self):
        det = track_det(object_id=42)
        with pytest.raises(DaftConvertError, match="object_id"):
            to_daft_tracking(track_input({"f1": [det]}), ctx=VIDEO_CTX)

    def test_object_id_must_be_nonempty(self):
        det = track_det(object_id="")
        with pytest.raises(DaftConvertError, match="object_id"):
            to_daft_tracking(track_input({"f1": [det]}), ctx=VIDEO_CTX)

    @pytest.mark.parametrize(
        "bad",
        [[1, 2], [1, 2, 3, 4], "not a list", None, {"x": 1}],
    )
    def test_3d_location_shape(self, bad):
        det = track_det(**{"3d_location": bad})
        with pytest.raises(DaftConvertError, match="3-element list"):
            to_daft_tracking(track_input({"f1": [det]}), ctx=VIDEO_CTX)

    def test_3d_location_non_numeric_rejected(self):
        det = track_det(**{"3d_location": [1.0, "y", 3.0]})
        with pytest.raises(DaftConvertError, match="not a number"):
            to_daft_tracking(track_input({"f1": [det]}), ctx=VIDEO_CTX)

    def test_3d_location_bool_rejected(self):
        # ``isinstance(True, int)`` is True in Python; converter must filter bools.
        det = track_det(**{"3d_location": [True, 0, 0]})
        with pytest.raises(DaftConvertError, match="not a number"):
            to_daft_tracking(track_input({"f1": [det]}), ctx=VIDEO_CTX)

    def test_scale_must_be_nonneg(self):
        det = track_det(
            **{"3d_location": [0.0, 0.0, 0.0], "3d_bounding_box_scale": [1.0, -0.5, 1.0]}
        )
        with pytest.raises(DaftConvertError, match=">= 0"):
            to_daft_tracking(track_input({"f1": [det]}), ctx=VIDEO_CTX)

    def test_rotation_can_be_negative(self):
        # Rotation is not constrained nonneg in the schema (pitch/roll/yaw can be negative).
        det = track_det(**{"3d_bounding_box_rotation": [-0.1, 0.0, -3.14]})
        out = to_daft_tracking(track_input({"f1": [det]}), ctx=VIDEO_CTX)
        assert out["frames"]["f1"][0]["3d_bounding_box_rotation"] == [-0.1, 0.0, -3.14]

    def test_2d_visible_must_be_dict(self):
        det = track_det(**{"2d_bounding_box_visible": [1, 2, 3, 4]})
        with pytest.raises(DaftConvertError, match="2d_bounding_box_visible"):
            to_daft_tracking(track_input({"f1": [det]}), ctx=VIDEO_CTX)

    def test_2d_visible_bbox_shape(self):
        det = track_det(**{"2d_bounding_box_visible": {"cam_001": [1, 2, 3]}})
        with pytest.raises(DaftConvertError, match="4-element list"):
            to_daft_tracking(track_input({"f1": [det]}), ctx=VIDEO_CTX)

    def test_2d_visible_camera_id_must_be_string(self):
        det = track_det(**{"2d_bounding_box_visible": {"": [1, 2, 3, 4]}})
        with pytest.raises(DaftConvertError, match="camera id"):
            to_daft_tracking(track_input({"f1": [det]}), ctx=VIDEO_CTX)
