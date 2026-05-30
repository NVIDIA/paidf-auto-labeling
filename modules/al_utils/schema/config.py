# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from typing import List, Optional

from al_utils.schema.data import SampleConfig
from al_utils.schema.endpoints import EndpointsConfig
from al_utils.schema.mcq import McqGenerationConfig
from al_utils.schema.pipeline_settings import PipelineSettings
from al_utils.schema.sr import SuperResolutionConfig
from al_utils.schema.tracking import DetectionAndTrackingConfig
from al_utils.schema.vlm_json import VlmJsonConfig
from pydantic import BaseModel, ConfigDict, model_validator


class PipelineConfig(BaseModel):
    """Root pipeline config.

    ``ConfigSchema`` alias is kept in ``config/schema.py`` for stable existing
    tests and imports.
    """

    model_config = ConfigDict(extra="forbid")

    pipeline: PipelineSettings
    data: List[SampleConfig]
    endpoints: Optional[EndpointsConfig] = None
    super_resolution: Optional[SuperResolutionConfig] = None
    detection_and_tracking: Optional[DetectionAndTrackingConfig] = None
    vlm_json: Optional[VlmJsonConfig] = None
    mcq_generation: Optional[McqGenerationConfig] = None

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> "PipelineConfig":
        if not isinstance(self.data, list) or len(self.data) == 0:
            raise ValueError("data must be a non-empty list of samples")

        vlm_enabled = bool(self.vlm_json and self.vlm_json.enabled)
        mcq_enabled = bool(self.mcq_generation and self.mcq_generation.enabled)
        if vlm_enabled:
            has_ep = bool(self.endpoints and self.endpoints.vlm and self.endpoints.vlm.url and self.endpoints.vlm.model)
            has_env = bool(os.getenv("VLM_BASE_URL") and os.getenv("VLM_MODEL"))
            if not (has_ep or has_env):
                raise ValueError(
                    "vlm_json.enabled=true requires endpoints.vlm.{url,model} "
                    "(or NVCF env vars VLM_BASE_URL and VLM_MODEL)"
                )

        mcq_mode = self.mcq_generation.mode if self.mcq_generation is not None else "question-driven-vlm-llm"

        if mcq_enabled:
            if not (self.mcq_generation and self.mcq_generation.window_metadata_extraction):
                raise ValueError(
                    "mcq_generation.mode is a window/metadata mode, but "
                    "mcq_generation.window_metadata_extraction is missing."
                )
            has_vlm = bool(
                self.endpoints and self.endpoints.vlm and self.endpoints.vlm.url and self.endpoints.vlm.model
            ) or bool(os.getenv("VLM_BASE_URL") and os.getenv("VLM_MODEL"))
            has_llm = bool(
                self.endpoints and self.endpoints.llm and self.endpoints.llm.url and self.endpoints.llm.model
            ) or bool(os.getenv("LLM_BASE_URL") and os.getenv("LLM_MODEL"))
            if (
                mcq_mode
                in {
                    "window-vlm-llm",
                    "window-direct-vlm",
                    "question-driven-vlm-llm",
                }
                and not has_vlm
            ):
                raise ValueError(
                    "mcq_generation.mode in {window-vlm-llm, window-direct-vlm, "
                    "question-driven-vlm-llm} "
                    "requires endpoints.vlm.{url,model} (or env vars VLM_BASE_URL and VLM_MODEL)"
                )
            if (
                mcq_mode
                in {
                    "window-vlm-llm",
                    "question-driven-vlm-llm",
                    "metadata-llm",
                }
                and not has_llm
            ):
                raise ValueError(
                    "mcq_generation.mode in {window-vlm-llm, question-driven-vlm-llm, "
                    "metadata-llm} "
                    "requires endpoints.llm.{url,model} (or env vars LLM_BASE_URL and LLM_MODEL)"
                )
            if mcq_mode == "metadata-llm":
                if not all(bool(s.inputs and s.inputs.metadata_json_path) for s in (self.data or [])):
                    raise ValueError(
                        "mcq_generation.enabled=true (mode=metadata-llm) requires "
                        "data[*].inputs.metadata_json_path for all samples"
                    )

        return self
