# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import config.loader
import config.schema
import pytest
from al_utils.schema.config import PipelineConfig
from al_utils.schema.data import SampleConfig, SampleInputsConfig, SampleOutputConfig
from al_utils.schema.endpoints import EndpointsConfig, LlmEndpointConfig, VlmEndpointConfig
from al_utils.schema.mcq import McqGenerationConfig, WindowMetadataExtractionConfig
from al_utils.schema.pipeline_settings import PipelineSettings
from pydantic import ValidationError


def test_required_sentinel_fails_schema_validation() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "configs" / "pipeline_example.yaml"
    cfg_obj, _config_dir = config.loader.load_config_with_overrides(str(config_path), overrides=[], logger=None)

    # validate_schema() is a safe wrapper (logs + returns None on error)
    assert config.schema.validate_schema(cfg_obj, logger=None) is None

    # Direct validation should surface a clear error mentioning __REQUIRED__.
    with pytest.raises(ValidationError) as e:
        config.schema.ConfigSchema.model_validate(cfg_obj)
    msg = str(e.value)
    assert "__REQUIRED__" in msg
    assert "must be set" in msg or "is required and cannot be empty" in msg


def test_metadata_llm_auto_disables_vlm_verify() -> None:
    """metadata-llm auto-sets vlm_verify_enabled=False so no VLM endpoint is needed."""
    w = WindowMetadataExtractionConfig(
        window_seconds=4.0,
        vlm_verify_enabled=True,  # blueprint default
    )
    cfg = McqGenerationConfig(
        enabled=True,
        mode="metadata-llm",
        window_metadata_extraction=w,
    )
    assert cfg.window_metadata_extraction is not None
    assert cfg.window_metadata_extraction.vlm_verify_enabled is False


def test_metadata_llm_requires_metadata_json_path() -> None:
    w = WindowMetadataExtractionConfig(window_seconds=4.0)
    mcq = McqGenerationConfig(enabled=True, mode="metadata-llm", window_metadata_extraction=w)

    with pytest.raises(ValidationError) as exc:
        PipelineConfig(
            pipeline=PipelineSettings(),
            data=[
                SampleConfig(
                    inputs=SampleInputsConfig(video_path="video.mp4"),
                    output=SampleOutputConfig(out_dir="out"),
                )
            ],
            endpoints=EndpointsConfig(
                vlm=VlmEndpointConfig(url="", model=""),
                llm=LlmEndpointConfig(url="http://llm/v1", model="llm"),
            ),
            mcq_generation=mcq,
        )

    assert "metadata_json_path" in str(exc.value)


def test_other_modes_preserve_vlm_verify() -> None:
    """vlm_verify_enabled is not modified for non-metadata-llm modes."""
    w = WindowMetadataExtractionConfig(window_seconds=4.0, vlm_verify_enabled=True)
    for mode in ("window-vlm-llm", "window-direct-vlm", "question-driven-vlm-llm"):
        cfg = McqGenerationConfig(enabled=True, mode=mode, window_metadata_extraction=w)  # type: ignore[arg-type]
        assert cfg.window_metadata_extraction is not None
        assert cfg.window_metadata_extraction.vlm_verify_enabled is True, f"failed for mode={mode}"
