# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base class and result type for detection & tracking."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TrackingResult:
    """Result from a single detection & tracking run."""

    success: bool
    instances_json: Optional[Path] = None
    objects_json: Optional[Path] = None
    # Optional overlay variant with track IDs burned in
    # (``sidecars/<stem>_tracking_red_id.<ext>`` — ``mp4`` for video input,
    # ``png`` for image input). Populated only when the tracker writes it.
    # Pipeline uses this as a preferred input to downstream VLM stages so the
    # VLM sees per-instance labels.
    tracking_video_red_id: Optional[Path] = None


class BaseTracker(ABC):
    """Abstract base class for all detection & tracking strategies.

    - Constructed ONCE before the sample loop (model loaded in __init__).
    - ``run()`` is called per-sample — always a direct Python call (no subprocess).
    """

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    @abstractmethod
    def run(self, video_path: Path, output_dir: Path) -> TrackingResult:
        """Run detection & tracking on a single media sample.

        Args:
            video_path: Path to the input media (SR output or original).
            output_dir: Per-sample output (scene) directory. JSON outputs land
                under ``contextual/``; debug/overlay artifacts under
                ``sidecars/``.

        Returns:
            TrackingResult with paths to written JSON artefacts.
        """
