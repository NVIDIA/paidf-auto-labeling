# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolver contract for super-resolution backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from core import ScenePaths


@dataclass(frozen=True)
class SrResult:
    """Outcome of one super-resolution attempt."""

    success: bool
    output_path: Path | None = None
    message: str | None = None


class Resolver(ABC):
    """Abstract base class for super-resolution backends."""

    @abstractmethod
    def run(self, media_path: Path, scene_paths: ScenePaths) -> SrResult:
        """Run super-resolution for one media sample."""


__all__ = ["Resolver", "SrResult"]
