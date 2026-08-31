# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tracker base contract.

The ``Tracker`` ABC is the seam between the framework's ``TaskInterface``
and the concrete CV models (``rfdetr`` + ``boosttrack``/``deepocsort``,
SAM3, …). Concrete backends live under ``backends/`` so adding a new
one is one new file plus a ``register_tracker`` call — the ABC stays
narrow.

Keeping the ABC narrow (one ``run`` method, one result dataclass)
keeps backend ports mechanical and avoids coupling the task interface
to any one detector or tracker implementation.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from core import (
    SceneContext,
    ScenePaths,
)


@dataclass
class TrackingResult:
    """Result from a single detection-and-tracking run.

    Attributes:
        success: True iff the tracker considers its output valid.
        instances_json: Path to ``contextual/instances.json`` when written.
        objects_json: Path to ``contextual/objects.json`` when written.
        detection_overlay: Optional sidecar overlay with detector boxes.
        tracking_overlay: Optional sidecar overlay with tracked boxes.
        annotated_video: Optional model-specific annotated video sidecar.
        tracking_video_red_id: Optional sidecar overlay with track-id
            labels burned in. Downstream VLM stages prefer this as input
            when available so the VLM can reference per-instance ids.
    """

    success: bool
    instances_json: Path | None = None
    objects_json: Path | None = None
    detection_overlay: Path | None = None
    tracking_overlay: Path | None = None
    annotated_video: Path | None = None
    tracking_video_red_id: Path | None = None


class Tracker(ABC):
    """Abstract base class for detection-and-tracking implementations.

    Implementations are constructed once before the per-scene loop
    (heavy model state is loaded in ``__init__``) and ``run`` is called
    per scene. This is a *direct Python call* contract — implementations
    must not require subprocess invocation here.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    @abstractmethod
    def run(
        self,
        media_path: Path,
        scene_paths: ScenePaths,
        *,
        scene_ctx: SceneContext,
    ) -> TrackingResult:
        """Run detection and tracking on a single media sample.

        Implementations should write
        ``contextual/objects.json`` and ``contextual/instances.json`` on
        success and return a ``TrackingResult`` with their paths.
        """


__all__ = ["Tracker", "TrackingResult"]
