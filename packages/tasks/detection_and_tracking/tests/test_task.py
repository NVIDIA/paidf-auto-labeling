# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from core import (
    DataEntry,
    EnhancedMediaState,
    SceneContext,
    ScenePaths,
    ScenePipelineState,
    read_pipeline_state,
    write_pipeline_state,
)
from detection_and_tracking import DetectionAndTrackingConfig, DetectionAndTrackingTask, StubTracker
from detection_and_tracking.artifacts import TRACKING_ARTIFACTS_KEY, TrackingArtifactsState
from detection_and_tracking.backends.runtime import instances_json_path, objects_json_path
from detection_and_tracking.factory import create_tracker, list_trackers
from detection_and_tracking.tracker import Tracker, TrackingResult
from pydantic import ValidationError


class _CaptureTracker(Tracker):
    def __init__(self) -> None:
        super().__init__(logger=logging.getLogger("test_capture_tracker"))
        self.seen_media_path: Path | None = None

    def run(
        self,
        media_path: Path,
        scene_paths: ScenePaths,
        *,
        scene_ctx: SceneContext,
    ) -> TrackingResult:
        _ = scene_ctx
        self.seen_media_path = media_path
        objects_json = objects_json_path(scene_paths)
        instances_json = instances_json_path(scene_paths)
        objects_json.write_text("{}", encoding="utf-8")
        instances_json.write_text("{}", encoding="utf-8")
        return TrackingResult(
            success=True,
            objects_json=objects_json,
            instances_json=instances_json,
        )


def test_stub_tracker_emits_daft_artifacts(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.touch()
    data = tmp_path / "scene"

    task = DetectionAndTrackingTask(
        config=DetectionAndTrackingConfig(tracker="stub"),
    )
    task.run(DataEntry(media_path=str(media), data_path=str(data)))

    objects = data / "contextual" / "objects.json"
    instances = data / "contextual" / "instances.json"
    assert objects.exists() and instances.exists()
    raw_media = data / "raw" / "clip.mp4"
    assert raw_media.is_symlink()
    assert raw_media.resolve() == media.resolve()
    objects_payload = json.loads(objects.read_text())
    assert objects_payload["version"] == "metropolis-v3.0"
    assert objects_payload["video_id"] == "clip"
    assert objects_payload["frames"] == []

    pipeline_state = read_pipeline_state(data)
    tracking_artifacts = TrackingArtifactsState.model_validate(
        pipeline_state.task_artifacts[TRACKING_ARTIFACTS_KEY]
    )
    assert tracking_artifacts.success is True
    assert tracking_artifacts.objects_json == str(objects)


def test_stub_tracker_uses_persisted_media_id_for_staged_active_media(
    tmp_path: Path,
) -> None:
    data = tmp_path / "pipeline_data_tmp"
    active = data / "sidecars" / "active.mp4"
    active.parent.mkdir(parents=True)
    active.touch()
    write_pipeline_state(data, ScenePipelineState(media_id="clip_001"))

    task = DetectionAndTrackingTask(
        config=DetectionAndTrackingConfig(tracker="stub"),
    )
    task.run(DataEntry(media_path=str(active), data_path=str(data)))

    objects_payload = json.loads((data / "contextual" / "objects.json").read_text())
    assert objects_payload["video_id"] == "clip_001"


def test_task_can_copy_raw_media(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"media")
    data = tmp_path / "scene"

    task = DetectionAndTrackingTask(
        config=DetectionAndTrackingConfig(tracker="stub", copy_media=True),
    )
    task.run(DataEntry(media_path=str(media), data_path=str(data)))

    raw_media = data / "raw" / "clip.mp4"
    assert raw_media.exists()
    assert not raw_media.is_symlink()
    assert raw_media.read_bytes() == b"media"


def test_disabled_task_is_noop(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.touch()
    data = tmp_path / "scene"
    task = DetectionAndTrackingTask(
        config=DetectionAndTrackingConfig(enabled=False),
        tracker=StubTracker(logger=logging.getLogger("test_disabled_tracker")),
    )
    task.run(DataEntry(media_path=str(media), data_path=str(data)))
    assert not (data / "contextual" / "objects.json").exists()


def test_builtin_tracker_kinds_are_registered() -> None:
    kinds = list_trackers()
    assert kinds == ["rfdetr-boosttrack", "rfdetr-bytetrack", "sam3", "stub"]


def test_deepocsort_tracker_kind_still_fails_loudly() -> None:
    with pytest.raises(NotImplementedError, match="planned but not implemented"):
        create_tracker("rfdetr-deepocsort", logger=logging.getLogger("test_deepocsort"))


def test_config_rejects_unknown_tracker_kind() -> None:
    unknown_tracker: Any = "unknown"
    with pytest.raises(ValidationError):
        DetectionAndTrackingConfig(tracker=unknown_tracker)


def test_sam3_requires_prompts() -> None:
    with pytest.raises(ValueError, match="requires sam3_prompts"):
        create_tracker("sam3", logger=logging.getLogger("test_sam3"))


def test_task_uses_enhanced_media_when_available(tmp_path: Path) -> None:
    original = tmp_path / "clip.mp4"
    original.touch()
    enhanced = tmp_path / "sr.mp4"
    enhanced.touch()
    data = tmp_path / "scene"
    write_pipeline_state(
        data,
        ScenePipelineState(
            enhanced_media=EnhancedMediaState(success=True, output_path=str(enhanced))
        ),
    )
    tracker = _CaptureTracker()
    task = DetectionAndTrackingTask(
        config=DetectionAndTrackingConfig(tracker="stub"), tracker=tracker
    )

    task.run(DataEntry(media_path=str(original), data_path=str(data)))

    assert tracker.seen_media_path == enhanced


def test_task_ignores_missing_enhanced_media(tmp_path: Path) -> None:
    original = tmp_path / "clip.mp4"
    original.touch()
    missing = tmp_path / "missing_sr.mp4"
    data = tmp_path / "scene"
    write_pipeline_state(
        data,
        ScenePipelineState(
            enhanced_media=EnhancedMediaState(success=True, output_path=str(missing))
        ),
    )
    tracker = _CaptureTracker()
    task = DetectionAndTrackingTask(
        config=DetectionAndTrackingConfig(tracker="stub"), tracker=tracker
    )

    task.run(DataEntry(media_path=str(original), data_path=str(data)))

    assert tracker.seen_media_path == original
