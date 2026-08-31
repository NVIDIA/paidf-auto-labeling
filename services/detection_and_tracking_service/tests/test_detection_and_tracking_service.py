# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from core import DataEntry, read_pipeline_state
from core.policy import EmptyOutputPolicy
from detection_and_tracking.artifacts import TRACKING_ARTIFACTS_KEY
from detection_and_tracking.config import DetectionAndTrackingConfig
from detection_and_tracking_service.main import (
    DetectionAndTrackingService,
    _validate_tracker_arg,
    build_config,
)
from pydantic import ValidationError


def test_detection_and_tracking_service_empty_input_exits_cleanly() -> None:
    service = DetectionAndTrackingService()

    with pytest.raises(SystemExit) as exc_info:
        service.execute(argparse.Namespace(), [])

    assert exc_info.value.code == "Pass --input or --input-file with at least one DataEntry."


def test_add_service_args_does_not_list_trackers(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_list_trackers() -> list[str]:
        raise AssertionError("list_trackers should not run during argument construction")

    monkeypatch.setattr("detection_and_tracking_service.main.list_trackers", fail_list_trackers)
    parser = argparse.ArgumentParser()

    DetectionAndTrackingService().add_service_args(parser)
    args = parser.parse_args([])

    assert args.tracker == "rfdetr-boosttrack"


def test_validate_tracker_arg_reports_invalid_tracker(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("detection_and_tracking_service.main.list_trackers", lambda: ["stub"])
    parser = argparse.ArgumentParser(prog="detection-service")

    with pytest.raises(SystemExit) as exc_info:
        _validate_tracker_arg(parser, "missing")

    assert exc_info.value.code == 2
    assert "Invalid tracker: missing" in capsys.readouterr().err


def test_detection_and_tracking_service_disabled_skips_task_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = DetectionAndTrackingService()

    def fail_task_creation(*args: object, **kwargs: object) -> object:
        raise AssertionError("DetectionAndTrackingTask should not be constructed")

    monkeypatch.setattr(
        "detection_and_tracking_service.main.DetectionAndTrackingTask",
        fail_task_creation,
    )

    service.execute(
        argparse.Namespace(disabled=True),
        [DataEntry(media_path="video.mp4", data_path="data")],
    )


def test_detection_and_tracking_service_run_disabled_skips_input_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = DetectionAndTrackingService()

    def fail_get_data_entries(args: argparse.Namespace) -> list[DataEntry]:
        _ = args
        raise AssertionError("disabled service should not load input data")

    def fail_task_creation(*args: object, **kwargs: object) -> object:
        raise AssertionError("DetectionAndTrackingTask should not be constructed")

    def fail_list_trackers() -> list[str]:
        raise AssertionError("list_trackers should not run when disabled")

    monkeypatch.setattr(sys, "argv", ["detection-service", "--disabled"])
    monkeypatch.setattr(service, "_get_data_entries", fail_get_data_entries)
    monkeypatch.setattr(
        "detection_and_tracking_service.main.DetectionAndTrackingTask",
        fail_task_creation,
    )
    monkeypatch.setattr("detection_and_tracking_service.main.list_trackers", fail_list_trackers)

    service.run()


def test_execute_invokes_linear_pipeline_run() -> None:
    service = DetectionAndTrackingService()
    parser = argparse.ArgumentParser()
    service.add_service_args(parser)
    args = parser.parse_args(["--tracker", "stub"])
    entries = [DataEntry(id="entry-1", media_path="/data/source/clip.mp4", data_path="/data/scene")]

    with (
        patch("detection_and_tracking_service.main.DetectionAndTrackingTask") as task_cls,
        patch("detection_and_tracking_service.main.DaftValidationTask") as validation_task_cls,
        patch("detection_and_tracking_service.main.LinearPipeline") as pipeline_cls,
    ):
        pipeline = pipeline_cls.return_value
        pipeline.run.return_value = entries

        service.execute(args, entries)

    task_cls.assert_called_once()
    pipeline_cls.assert_called_once()
    _, kwargs = pipeline_cls.call_args
    assert kwargs["name"] == "detection_and_tracking_pipeline"
    assert kwargs["policy"] is EmptyOutputPolicy.FAIL
    assert kwargs["tasks"] == [task_cls.return_value, validation_task_cls.return_value]
    pipeline.run.assert_called_once_with(entries)


def test_execute_stub_pipeline_writes_tracking_artifacts_to_scene(tmp_path: Path) -> None:
    service = DetectionAndTrackingService()
    parser = argparse.ArgumentParser()
    service.add_service_args(parser)
    args = parser.parse_args(["--tracker", "stub"])
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"media")
    scene = tmp_path / "scene"

    service.execute(args, [DataEntry(media_path=str(media), data_path=str(scene))])

    assert (scene / "contextual" / "objects.json").exists()
    assert (scene / "contextual" / "instances.json").exists()
    assert (scene / "sidecars" / "active.mp4").exists()
    state = read_pipeline_state(scene)
    assert TRACKING_ARTIFACTS_KEY in state.task_artifacts


def test_detection_and_tracking_service_run_applies_dev_data_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = DetectionAndTrackingService()
    original = [DataEntry(media_path="video.mp4", data_path="source")]
    copied = [DataEntry(media_path="video.mp4", data_path="dev/entry")]
    received: list[DataEntry] = []

    monkeypatch.setattr("detection_and_tracking_service.main.list_trackers", lambda: ["stub"])
    monkeypatch.setattr(sys, "argv", ["detection-service", "--tracker", "stub"])
    monkeypatch.setattr(service, "_get_data_entries", lambda _args: original)
    monkeypatch.setattr(
        service,
        "_copy_data_entries_to_dev_root",
        lambda entries, dev_root: copied if entries == original and dev_root == "dev" else [],
    )

    def capture_execute(_args: argparse.Namespace, data_entries: list[DataEntry]) -> None:
        received.extend(data_entries)

    monkeypatch.setattr(service, "execute", capture_execute)
    monkeypatch.setattr(
        sys,
        "argv",
        ["detection-service", "--tracker", "stub", "--dev-data-root", "dev"],
    )

    service.run()

    assert received == copied


def test_build_config_maps_cli_args() -> None:
    args = argparse.Namespace(
        tracker="sam3",
        disabled=False,
        model_cache_path="/models",
        gpu_ids="0",
        classes=["person"],
        threshold=0.45,
        iou_threshold=0.5,
        per_class=True,
        min_hits=4,
        max_age=80,
        min_track_frames=3,
        bbox_expansion_ratio=0.15,
        save_video=True,
        save_red_id_overlay=True,
        save_rgb=True,
        copy_media=True,
        allow_model_download=True,
        sam3_prompts=["forklift"],
        sam3_version="sam3.1",
        sam3_runtime="native",
        sam3_tracking_mode="continuous",
        sam3_target_fps=8.0,
        sam3_session_reset_s=12.0,
        sam3_max_duration_s=60.0,
        sam3_write_annotated_video=True,
        sam3_annotated_video_trails=True,
        sam3_annotated_video_label_style="track",
        sam3_annotated_video_mask_opacity=30,
        sam3_annotated_video_shape="box",
        sam3_write_masks=True,
        sam3_score_threshold_detection=0.6,
        sam3_det_nms_thresh=0.5,
        sam3_new_det_thresh=0.7,
        sam3_fill_hole_area=64,
        sam3_recondition_every_nth_frame=8,
        sam3_recondition_on_trk_masks=True,
        sam3_high_conf_thresh=0.8,
        sam3_high_iou_thresh=0.9,
        sam3_multiplex_count=8,
        sam3_max_num_objects=32,
        sam3_compile=True,
        extract_crops=True,
        crop_classes=["person"],
        crops_per_track=8,
        crop_padding=0.1,
        min_crop_size=32,
        min_detection_score=0.75,
        min_track_seconds=3.0,
        crop_format="png",
        crop_subdir="tracks/crops",
        tracks_sidecar="detection_and_tracking/tracks.json",
    )

    config = build_config(args)

    assert config.enabled is True
    assert config.tracker == "sam3"
    assert config.extract_crops is True
    assert config.crop_classes == ("person",)
    assert config.crops_per_track == 8
    assert config.crop_padding == 0.1
    assert config.min_crop_size == 32
    assert config.min_detection_score == 0.75
    assert config.min_track_seconds == 3.0
    assert config.crop_format == "png"
    assert config.crop_subdir == "tracks/crops"
    assert config.tracks_sidecar == "detection_and_tracking/tracks.json"
    assert config.model_cache_path == "/models"
    assert config.gpu_ids == "0"
    assert config.classes == ("person",)
    assert config.threshold == 0.45
    assert config.iou_threshold == 0.5
    assert config.per_class is True
    assert config.min_hits == 4
    assert config.max_age == 80
    assert config.min_track_frames == 3
    assert config.bbox_expansion_ratio == 0.15
    assert config.save_video is True
    assert config.save_red_id_overlay is True
    assert config.save_rgb is True
    assert config.copy_media is True
    assert config.allow_model_download is True
    assert config.sam3_prompts == ("forklift",)
    assert config.sam3_version == "sam3.1"
    assert config.sam3_runtime == "native"
    assert config.sam3_tracking_mode == "continuous"
    assert config.sam3_target_fps == 8.0
    assert config.sam3_session_reset_s == 12.0
    assert config.sam3_max_duration_s == 60.0
    assert config.sam3_write_annotated_video is True
    assert config.sam3_annotated_video_trails is True
    assert config.sam3_annotated_video_label_style == "track"
    assert config.sam3_annotated_video_mask_opacity == 30
    assert config.sam3_annotated_video_shape == "box"
    assert config.sam3_write_masks is True
    assert config.sam3_score_threshold_detection == 0.6
    assert config.sam3_det_nms_thresh == 0.5
    assert config.sam3_new_det_thresh == 0.7
    assert config.sam3_fill_hole_area == 64
    assert config.sam3_recondition_every_nth_frame == 8
    assert config.sam3_recondition_on_trk_masks is True
    assert config.sam3_high_conf_thresh == 0.8
    assert config.sam3_high_iou_thresh == 0.9
    assert config.sam3_multiplex_count == 8
    assert config.sam3_max_num_objects == 32
    assert config.sam3_compile is True


def test_cli_defaults_match_legacy_rfdetr_boosttrack() -> None:
    parser = argparse.ArgumentParser()
    DetectionAndTrackingService().add_service_args(parser)

    args = parser.parse_args([])

    assert args.threshold == 0.2
    assert args.iou_threshold == 0.3
    assert args.per_class is True
    assert args.min_hits == 3
    assert args.max_age == 60
    assert args.min_track_frames == 5
    assert args.bbox_expansion_ratio == 0.1


def test_cli_no_per_class_disables_legacy_default() -> None:
    parser = argparse.ArgumentParser()
    DetectionAndTrackingService().add_service_args(parser)

    args = parser.parse_args(["--no-per-class"])

    assert args.per_class is False


def test_sam3_recondition_on_trk_masks_accepts_explicit_false() -> None:
    parser = argparse.ArgumentParser()
    DetectionAndTrackingService().add_service_args(parser)

    args = parser.parse_args(["--sam3-recondition-on-trk-masks", "false"])

    assert args.sam3_recondition_on_trk_masks is False


def test_sam3_max_clip_duration_alias_matches_source_config_key() -> None:
    config = DetectionAndTrackingConfig.model_validate({"sam3_max_clip_duration_s": 45.0})

    assert config.sam3_max_duration_s == 45.0


def test_build_config_supports_disabled_mode() -> None:
    args = argparse.Namespace(
        tracker="rfdetr-boosttrack",
        disabled=True,
        model_cache_path=None,
        gpu_ids="all",
        classes=[],
        threshold=0.3,
        iou_threshold=0.3,
        per_class=False,
        min_hits=3,
        max_age=60,
        min_track_frames=1,
        bbox_expansion_ratio=0.1,
        save_video=False,
        save_red_id_overlay=False,
        save_rgb=False,
        copy_media=False,
        allow_model_download=False,
        sam3_prompts=[],
        sam3_version="sam3",
        sam3_runtime="auto",
        sam3_tracking_mode="chunked",
        sam3_target_fps=10.0,
        sam3_session_reset_s=10.0,
        sam3_max_duration_s=30.0,
        sam3_write_annotated_video=False,
        sam3_annotated_video_trails=False,
        sam3_annotated_video_label_style="id",
        sam3_annotated_video_mask_opacity=0,
        sam3_annotated_video_shape="contour",
        sam3_write_masks=False,
        sam3_score_threshold_detection=None,
        sam3_det_nms_thresh=None,
        sam3_new_det_thresh=None,
        sam3_fill_hole_area=None,
        sam3_recondition_every_nth_frame=None,
        sam3_recondition_on_trk_masks=None,
        sam3_high_conf_thresh=None,
        sam3_high_iou_thresh=None,
        sam3_multiplex_count=16,
        sam3_max_num_objects=16,
        sam3_compile=False,
        extract_crops=False,
        crop_classes=[],
        crops_per_track=16,
        crop_padding=0.0,
        min_crop_size=0,
        min_detection_score=0.0,
        min_track_seconds=0.0,
        crop_format="jpg",
        crop_subdir="tracks/crops",
        tracks_sidecar="detection_and_tracking/tracks.json",
    )

    config = build_config(args)

    assert config.enabled is False
    assert config.tracker == "rfdetr-boosttrack"
    assert config.extract_crops is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sam3_score_threshold_detection", -0.1),
        ("sam3_det_nms_thresh", 1.1),
        ("sam3_new_det_thresh", -0.1),
        ("sam3_high_conf_thresh", 1.1),
        ("sam3_high_iou_thresh", -0.1),
        ("sam3_fill_hole_area", -1),
        ("sam3_recondition_every_nth_frame", 0),
        ("bbox_expansion_ratio", -0.1),
        ("threshold", -0.01),
        ("min_track_frames", 0),
    ],
)
def test_detection_config_rejects_invalid_sam3_bounds(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        DetectionAndTrackingConfig.model_validate({field: value})


def test_dockerfile_builds_only_approved_media_dependencies() -> None:
    dockerfile = Path(__file__).parents[1] / "docker" / "Dockerfile"
    text = dockerfile.read_text(encoding="utf-8")

    assert "build_restricted_ffmpeg.sh" in text
    assert "media_toolchain.py ffmpeg-install --profile vp9-output" in text
    assert "--no-install-package av" in text
    assert "--no-install-package opencv-python" in text
    assert "--no-install-package opencv-python-headless" in text
    assert "--component opencv-headless" in text
    assert "--component pyav" in text
    assert text.count("media_toolchain.py verify") == 2
    assert "source=/opt/media-runtime" in text
    assert "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video" in text


def test_opencv_source_build_disables_video_backends() -> None:
    script = Path(__file__).parents[3] / "scripts" / "install_opencv_headless_from_source.sh"
    text = script.read_text(encoding="utf-8")

    assert "opencv-python-headless" in text
    assert "pip download --no-cache-dir --no-deps --no-build-isolation" in text
    assert "-DWITH_FFMPEG=OFF" in text
    assert "-DWITH_GSTREAMER=OFF" in text
    assert "-DWITH_V4L=OFF" in text


def test_shared_media_build_scripts_enforce_source_and_license_policy() -> None:
    scripts_dir = Path(__file__).parents[3] / "scripts"
    ffmpeg_text = (scripts_dir / "build_restricted_ffmpeg.sh").read_text(encoding="utf-8")
    pyav_text = (scripts_dir / "install_pyav_from_source.sh").read_text(encoding="utf-8")

    assert "--disable-gpl" in ffmpeg_text
    assert "--disable-nonfree" in ffmpeg_text
    assert "--disable-everything" in ffmpeg_text
    assert 'configure-flags "${profile}"' in ffmpeg_text
    assert "--no-binary=av" in pyav_text


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--min-detection-score", "2", "must be between 0.0 and 1.0"),
        ("--min-detection-score", "-0.1", "must be between 0.0 and 1.0"),
        ("--sam3-score-threshold-detection", "-0.1", "must be between 0.0 and 1.0"),
        ("--sam3-det-nms-thresh", "1.1", "must be between 0.0 and 1.0"),
        ("--sam3-new-det-thresh", "-0.1", "must be between 0.0 and 1.0"),
        ("--sam3-high-conf-thresh", "1.1", "must be between 0.0 and 1.0"),
        ("--sam3-high-iou-thresh", "-0.1", "must be between 0.0 and 1.0"),
        ("--sam3-fill-hole-area", "-1", "must be >= 0"),
        ("--sam3-recondition-every-nth-frame", "0", "must be >= 1"),
    ],
)
def test_sam3_cli_bounds_reject_invalid_values(
    flag: str,
    value: str,
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = argparse.ArgumentParser(prog="detection-service")
    DetectionAndTrackingService().add_service_args(parser)

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args([flag, value])

    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err


def test_crop_args_default_to_disabled() -> None:
    parser = argparse.ArgumentParser()
    DetectionAndTrackingService().add_service_args(parser)

    args = parser.parse_args([])

    assert args.extract_crops is False
    assert args.crop_classes == []
    assert args.crops_per_track == 16
    assert args.crop_padding == 0.0
    assert args.min_crop_size == 0
    assert args.crop_format == "jpg"
    assert args.crop_subdir == "tracks/crops"
    assert args.tracks_sidecar == "detection_and_tracking/tracks.json"


def test_crop_args_parse_pas_video_profile() -> None:
    parser = argparse.ArgumentParser()
    DetectionAndTrackingService().add_service_args(parser)

    args = parser.parse_args(
        [
            "--extract-crops",
            "--crop-classes",
            "person",
            "--crops-per-track",
            "8",
            "--crop-padding",
            "0.1",
            "--min-crop-size",
            "32",
            "--crop-format",
            "png",
        ]
    )

    assert args.extract_crops is True
    assert args.crop_classes == ["person"]
    assert args.crops_per_track == 8
    assert args.crop_padding == 0.1
    assert args.min_crop_size == 32
    assert args.crop_format == "png"


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--crop-padding", "-0.1", "must be >= 0.0"),
        ("--crops-per-track", "0", "must be >= 1"),
        ("--min-crop-size", "-1", "must be >= 0"),
    ],
)
def test_crop_cli_bounds_reject_invalid_values(
    flag: str,
    value: str,
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = argparse.ArgumentParser(prog="detection-service")
    DetectionAndTrackingService().add_service_args(parser)

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args([flag, value])

    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err
