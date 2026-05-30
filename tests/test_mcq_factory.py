# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for mcq_generation/factory.py — dispatch to correct generator class."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
from al_utils.endpoint_resolver import EndpointResolver
from al_utils.schema.config import PipelineConfig
from al_utils.schema.data import SampleConfig, SampleInputsConfig, SampleOutputConfig
from al_utils.schema.endpoints import EndpointsConfig, LlmEndpointConfig, VlmEndpointConfig
from al_utils.schema.mcq import McqGenerationConfig, WindowMetadataExtractionConfig
from al_utils.schema.pipeline_settings import PipelineSettings
from mcq_generation.factory import create_mcq_generator


def _resolver() -> MagicMock:
    r = MagicMock(spec=EndpointResolver)
    r.resolve_vlm.return_value = ("http://vlm/v1", "fake-vlm")
    r.resolve_llm.return_value = ("http://llm/v1", "fake-llm")
    r.vlm_retries = r.llm_retries = 3
    r.vlm_retry_backoff_s = r.llm_retry_backoff_s = 1.0
    return r


def _config(mode: str, *, with_window: bool = False) -> PipelineConfig:
    wme = WindowMetadataExtractionConfig(single_window=True) if with_window else None
    meta_path = "fake_meta.json" if mode == "metadata-llm" else None
    return PipelineConfig(
        pipeline=PipelineSettings(),
        data=[
            SampleConfig(
                inputs=SampleInputsConfig(
                    video_path="v.mp4",
                    metadata_json_path=meta_path,
                ),
                output=SampleOutputConfig(out_dir="out"),
            )
        ],
        endpoints=EndpointsConfig(
            vlm=VlmEndpointConfig(url="http://vlm/v1", model="fake-vlm"),
            llm=LlmEndpointConfig(url="http://llm/v1", model="fake-llm"),
        ),
        mcq_generation=McqGenerationConfig(
            enabled=True,
            mode=mode,
            window_metadata_extraction=wme,
        ),
    )


def _logger() -> logging.Logger:
    return logging.getLogger("test_factory")


def test_factory_window_vlm_llm() -> None:
    # Patch where the factory uses the symbol (imported into mcq_generation.factory).
    with patch("mcq_generation.factory.WindowVlmLlmGenerator") as MockGen:
        gen = create_mcq_generator(_config("window-vlm-llm", with_window=True), _resolver(), _logger())
    MockGen.assert_called_once()
    assert gen is MockGen.return_value


def test_factory_window_direct_vlm() -> None:
    with patch("mcq_generation.factory.WindowDirectVlmGenerator") as MockGen:
        gen = create_mcq_generator(_config("window-direct-vlm", with_window=True), _resolver(), _logger())
    MockGen.assert_called_once()
    assert gen is MockGen.return_value


def test_factory_question_driven_vlm_llm() -> None:
    with patch("mcq_generation.factory.QuestionDrivenVlmLlmGenerator") as MockGen:
        gen = create_mcq_generator(_config("question-driven-vlm-llm", with_window=True), _resolver(), _logger())
    MockGen.assert_called_once()
    assert gen is MockGen.return_value


def test_factory_metadata_llm() -> None:
    with patch("mcq_generation.factory.MetadataLlmGenerator") as MockGen:
        gen = create_mcq_generator(_config("metadata-llm", with_window=True), _resolver(), _logger())
    MockGen.assert_called_once()
    assert gen is MockGen.return_value


def test_factory_unknown_mode_raises() -> None:
    cfg = _config("question-driven-vlm-llm", with_window=True)
    bad_mcq = McqGenerationConfig.model_construct(mode="not-a-real-mode", enabled=True)
    cfg = cfg.model_copy(update={"mcq_generation": bad_mcq})
    with pytest.raises(ValueError, match="Unknown mcq_generation.mode"):
        create_mcq_generator(cfg, _resolver(), _logger())


def test_factory_window_mode_without_window_cfg_raises() -> None:
    """window-vlm-llm without window_metadata_extraction must raise in _require_window_cfg."""
    # Pydantic's cross-field validator catches this at config creation time, so we bypass
    # Pydantic to test the factory's own guard independently.
    cfg = _config("window-vlm-llm", with_window=True)
    bad_mcq = McqGenerationConfig.model_construct(mode="window-vlm-llm", enabled=True, window_metadata_extraction=None)
    cfg = cfg.model_copy(update={"mcq_generation": bad_mcq})
    with pytest.raises(ValueError, match="window_metadata_extraction"):
        create_mcq_generator(cfg, _resolver(), _logger())


def test_factory_config_dir_forwarded() -> None:
    """config_dir is forwarded to the generator constructor."""
    with patch("mcq_generation.factory.WindowVlmLlmGenerator") as MockGen:
        create_mcq_generator(
            _config("window-vlm-llm", with_window=True),
            _resolver(),
            _logger(),
            config_dir="/some/dir",
        )
    _, kwargs = MockGen.call_args
    assert kwargs.get("config_dir") == "/some/dir"
