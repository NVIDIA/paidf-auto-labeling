# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free tracker used to exercise the pipeline shape."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core import (
    SceneContext,
    ScenePaths,
    write_json,
)
from core.formats.daft import daft_envelope

from detection_and_tracking.backends.runtime import instances_json_path, objects_json_path
from detection_and_tracking.tracker import Tracker, TrackingResult


class StubTracker(Tracker):
    """Dependency-free tracker used to exercise the full pipeline.

    Writes valid (but empty) DAFT envelopes for ``objects.json`` and
    ``instances.json`` so downstream emitters and the DAFT validator have
    something to consume during smoke tests. Replace with a real backend
    (e.g. RF-DETR + BoostTrack) once those dependencies are migrated.
    """

    def run(
        self,
        media_path: Path,
        scene_paths: ScenePaths,
        *,
        scene_ctx: SceneContext,
    ) -> TrackingResult:
        self.logger.info(f"[stub-tracker] writing empty objects/instances for {media_path}")

        objects_payload: dict[str, Any] = daft_envelope(
            "objects", scene_ctx, include_scene_id=not scene_ctx.is_image
        )
        objects_payload["frames"] = []
        objects_json = objects_json_path(scene_paths)
        write_json(objects_json, objects_payload)

        instances_payload: dict[str, Any] = daft_envelope(
            "instances", scene_ctx, include_scene_id=False
        )
        instances_payload["instances"] = []
        instances_json = instances_json_path(scene_paths)
        write_json(instances_json, instances_payload)

        return TrackingResult(
            success=True,
            objects_json=objects_json,
            instances_json=instances_json,
        )


__all__ = ["StubTracker"]
