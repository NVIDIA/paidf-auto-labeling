# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base class and result type for MCQ generation."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from al_utils.schema.data import SampleConfig


@dataclass
class MCQResult:
    """Result from a single MCQ generation run.

    Task paths are explicit by DAFT task type. Use ``fallback_json`` only when a
    caller needs the first available task file regardless of task type.
    """

    success: bool
    mcq_json: Optional[Path] = None
    bcq_json: Optional[Path] = None
    open_qa_json: Optional[Path] = None
    metadata_json: Optional[Path] = None

    @property
    def fallback_json(self) -> Optional[Path]:
        """First written task file, preserving mcq -> bcq -> open_qa precedence."""
        return self.mcq_json or self.bcq_json or self.open_qa_json


class BaseMCQGenerator(ABC):
    """Abstract base class for all MCQ generation strategies.

    - Constructed ONCE before the sample loop (factory initialises all state).
    - ``generate()`` is called per-sample; always a direct Python call.
    - ``run_pre_step()`` is a no-op by default, overridden by
      ``QuestionDrivenVlmLlmGenerator`` for the LLM prompt-gen phase that must
      run before super-resolution.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def run_pre_step(self, out_dir: Path, sample: "SampleConfig") -> None:
        """Optional pre-SR phase.

        Default: no-op.
        Override in ``QuestionDrivenVlmLlmGenerator`` for LLM prompt generation.

        Args:
            out_dir: Per-sample output directory (local, already resolved).
            sample:  Per-sample config (inputs / output paths).
        """

    @abstractmethod
    def generate(
        self,
        video_path: Path,
        output_dir: Path,
        *,
        events_json: Optional[Path] = None,
        video_json: Optional[Path] = None,
        metadata_json: Optional[Path] = None,
    ) -> MCQResult:
        """Generate MCQ JSON for a single media sample.

        Args:
            video_path:    Path to the input media (SR output or original).
            output_dir:    Per-sample output directory.
            events_json:   Optional path to VLM ``events.json`` produced by the
                           VLM JSON stage; ignored by current MCQ modes.
            video_json:    Optional path to VLM ``video.json`` produced by the
                           VLM JSON stage; ignored by current MCQ modes.
            metadata_json: Path to a metadata JSON sidecar (used by
                           ``metadata-llm`` mode; ignored by other modes).

        Returns:
            MCQResult with paths to written artefacts.
        """
