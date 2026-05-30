# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base class for super-resolution runners."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class BaseSuperResolver(ABC):
    """Abstract base class for super-resolution strategies.

    SR is the only stage that remains a subprocess (``torchrun`` for
    multi-GPU diffusion).  The resolver is still constructed ONCE before
    the sample loop; ``run()`` is called per-sample.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    @abstractmethod
    def run(
        self,
        input_video: Path,
        output_video: Path,
        *,
        log_dir: Optional[Path] = None,
        pipeline_log: Optional[Path] = None,
    ) -> Path:
        """Run super-resolution on a single media sample.

        Args:
            input_video:   Path to the input media.
            output_video:  Desired path for the SR output media.
            log_dir:       Optional directory for per-stage log files.
            pipeline_log:  Optional pipeline log file to append stage events to.

        Returns:
            Path to the SR output media (same as ``output_video`` on success).

        Raises:
            SystemExit: If SR fails and ``empty_output_policy="fail"``.
        """
