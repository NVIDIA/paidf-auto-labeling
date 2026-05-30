# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# This module re-exports from al_utils.schema for stable imports.
# New code should import directly from al_utils.schema.

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

# Re-export all public schema types from al_utils.schema.
from al_utils.schema.base import (
    _REQUIRED_SENTINEL,
    _clean_optional_str,
    _clean_required_str,
    _clean_str,
)
from al_utils.schema.config import PipelineConfig
from al_utils.schema.data import SampleConfig, SampleInputsConfig, SampleOutputConfig
from al_utils.schema.endpoints import EndpointsConfig, LlmEndpointConfig, VlmEndpointConfig
from al_utils.schema.mcq import McqGenerationConfig, WindowMetadataExtractionConfig
from al_utils.schema.pipeline_settings import PipelineSettings
from al_utils.schema.sr import SuperResolutionConfig
from al_utils.schema.tracking import DetectionAndTrackingConfig
from al_utils.schema.vlm_json import VlmJsonConfig
from pydantic import ValidationError

# Stable alias: existing code imports ConfigSchema from this module.
ConfigSchema = PipelineConfig

__all__ = [
    # New canonical name
    "PipelineConfig",
    # Stable alias
    "ConfigSchema",
    # Sub-models
    "PipelineSettings",
    "SampleConfig",
    "SampleInputsConfig",
    "SampleOutputConfig",
    "EndpointsConfig",
    "VlmEndpointConfig",
    "LlmEndpointConfig",
    "SuperResolutionConfig",
    "DetectionAndTrackingConfig",
    "VlmJsonConfig",
    "McqGenerationConfig",
    "WindowMetadataExtractionConfig",
    # Helpers
    "_REQUIRED_SENTINEL",
    "_clean_str",
    "_clean_optional_str",
    "_clean_required_str",
    # Validation function
    "validate_schema",
]


def validate_schema(config: Dict[str, Any], *, logger: Optional[logging.Logger] = None) -> Optional[PipelineConfig]:
    """
    Validate user-facing config and return the validated Pydantic model.

    Returns a PipelineConfig instance on success, or None on validation failure.
    The returned model can be passed directly to pipeline stages — no normalize_config() needed.
    """
    try:
        cfg: Dict[str, Any] = dict(config or {})
        pipeline = dict(cfg.get("pipeline", {}) or {})
        if "input_dir" in pipeline:
            raise ValueError("pipeline.input_dir is not allowed. Use data[*].inputs.video_path.")
        if "out_dir" in pipeline or "log_dir" in pipeline:
            raise ValueError("pipeline.{out_dir,log_dir} are not supported. Use data[*].output.out_dir.")

        return PipelineConfig.model_validate(cfg)
    except (ValidationError, ValueError) as e:
        if logger is not None:
            logger.error("Config schema validation failed.")
            logger.error(str(e))
        return None
