# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detection-and-tracking ``TaskInterface`` implementation.

Wraps a configured ``Tracker`` so it composes with the framework's
pipeline. Configuration is intentionally tiny — backend selection plus
runtime paths and artifact controls — and lives in a Pydantic model rather
than free-form keyword arguments so config-loading layers (YAML / OmegaConf)
can validate it once at the service boundary.
"""

from __future__ import annotations

from pathlib import Path
from typing import override

from core import (
    DataEntry,
    ensure_scene_skeleton,
    read_pipeline_state,
    scene_context_for_entry,
    write_pipeline_state,
)
from core.tasks import SequentialTask

from detection_and_tracking.artifacts import TRACKING_ARTIFACTS_KEY, TrackingArtifactsState
from detection_and_tracking.backends.runtime import stage_raw_media
from detection_and_tracking.config import DetectionAndTrackingConfig
from detection_and_tracking.crops import extract_track_crops
from detection_and_tracking.factory import create_tracker
from detection_and_tracking.tracker import Tracker, TrackingResult


class DetectionAndTrackingTask(SequentialTask):
    """Run detection + tracking and emit DAFT contextual artifacts.

    Tracker construction is deferred to a factory + registry so the task
    stays decoupled from any specific CV backend. The task is responsible
    for: scene skeleton creation, scene-context derivation, calling the
    tracker, and recording tracking artifact references in the scene
    manifest so downstream tasks can find the produced files.
    """

    def __init__(
        self,
        config: DetectionAndTrackingConfig | None = None,
        *,
        tracker: Tracker | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or "detection_and_tracking")
        self.config = config or DetectionAndTrackingConfig()
        self.tracker: Tracker | None = tracker

    @override
    def run(self, data_entry: DataEntry) -> DataEntry:
        if not self.config.enabled:
            self.logger.info("Detection-and-tracking disabled by config; skipping.")
            return data_entry

        if self.tracker is None:
            self.tracker = create_tracker(self.config.tracker, self.logger, self.config)

        scene_paths = ensure_scene_skeleton(data_entry.data_path)
        pipeline_state = read_pipeline_state(data_entry.data_path)
        original_media_path = Path(data_entry.media_path)
        media_path = _select_tracker_input(original_media_path, pipeline_state.enhanced_media)
        scene_ctx = scene_context_for_entry(data_entry)
        stage_raw_media(
            scene_paths=scene_paths,
            media_path=media_path,
            scene_ctx=scene_ctx,
            copy_media=self.config.copy_media,
            logger=self.logger,
        )

        result: TrackingResult = self.tracker.run(
            media_path,
            scene_paths,
            scene_ctx=scene_ctx,
        )

        if not result.success:
            self.logger.warning(
                f"Tracker {self.tracker.__class__.__name__} reported failure on "
                f"{media_path}; downstream tasks will fall back to original input."
            )

        objects_json = str(result.objects_json) if result.objects_json else None
        instances_json = str(result.instances_json) if result.instances_json else None
        detection_overlay = str(result.detection_overlay) if result.detection_overlay else None
        tracking_overlay = str(result.tracking_overlay) if result.tracking_overlay else None
        annotated_video = str(result.annotated_video) if result.annotated_video else None
        red_id_overlay = str(result.tracking_video_red_id) if result.tracking_video_red_id else None

        tracks_json: str | None = None
        if (
            result.success
            and self.config.extract_crops
            and result.objects_json is not None
            and result.instances_json is not None
        ):
            tracks_path = extract_track_crops(
                media_path,
                scene_paths,
                self.config,
                objects_json=result.objects_json,
                instances_json=result.instances_json,
                logger=self.logger,
            )
            tracks_json = str(tracks_path) if tracks_path else None

        pipeline_state.data_entry_id = pipeline_state.data_entry_id or data_entry.id
        pipeline_state.media_path = pipeline_state.media_path or data_entry.media_path
        pipeline_state.task_artifacts[TRACKING_ARTIFACTS_KEY] = TrackingArtifactsState(
            success=result.success,
            objects_json=objects_json,
            instances_json=instances_json,
            detection_overlay_path=detection_overlay,
            tracking_overlay_path=tracking_overlay,
            annotated_video_path=annotated_video,
            red_id_overlay_path=red_id_overlay,
            tracks_json=tracks_json,
            crops_root=self.config.crop_subdir if tracks_json else None,
        ).model_dump()
        write_pipeline_state(data_entry.data_path, pipeline_state)

        return data_entry


def _select_tracker_input(original_media_path: Path, enhanced_media: object | None) -> Path:
    output_path = getattr(enhanced_media, "output_path", None)
    success = bool(getattr(enhanced_media, "success", False))
    if success and output_path:
        candidate = Path(str(output_path))
        if candidate.exists() and candidate.is_file():
            return candidate
    return original_media_path


__all__ = ["DetectionAndTrackingConfig", "DetectionAndTrackingTask"]
