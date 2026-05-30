# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base class and result type for VLM JSON generation."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class VlmJsonResult:
    """Result from a single VLM JSON generation run.

    Image scenes populate ``image_json``. Video scenes populate ``video_json``
    and ``events_json``.
    """

    success: bool
    events_json: Optional[Path] = None
    video_json: Optional[Path] = None
    image_json: Optional[Path] = None


class BaseVlmJsonGenerator(ABC):
    """Abstract base class for all VLM JSON generation strategies.

    - Constructed ONCE before the sample loop (model/client loaded in __init__).
    - ``generate()`` is called per-sample — always a direct Python call (no subprocess).
    """

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    @abstractmethod
    def generate(self, video_path: Path, output_dir: Path) -> VlmJsonResult:
        """Run VLM JSON generation on a single media sample.

        Args:
            video_path: Path to the input media.
            output_dir: Per-sample output (scene) directory. JSON outputs land
                under ``contextual/``; intermediate artefacts under
                ``sidecars/``.

        Returns:
            VlmJsonResult with paths to written JSON artefacts.
        """
