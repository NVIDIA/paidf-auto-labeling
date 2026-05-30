# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the three thin MCQ generator wrapper classes.

Tests cover:
  - _load_text() path resolution / fallback
  - WindowVlmLlmGenerator.generate(): paths and skip_existing forwarded to runner
  - WindowDirectVlmGenerator.generate(): paths and skip_existing forwarded to runner
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

from al_utils.endpoint_resolver import EndpointResolver
from al_utils.schema.config import PipelineConfig
from al_utils.schema.data import SampleConfig, SampleInputsConfig, SampleOutputConfig
from al_utils.schema.endpoints import EndpointsConfig, LlmEndpointConfig, VlmEndpointConfig
from al_utils.schema.mcq import McqGenerationConfig, WindowMetadataExtractionConfig
from al_utils.schema.pipeline_settings import PipelineSettings
from mcq_generation.mcq.utils.prompt_io import load_text as _load_text
from mcq_generation.window_direct_vlm import WindowDirectVlmGenerator
from mcq_generation.window_vlm_llm import WindowVlmLlmGenerator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolver() -> MagicMock:
    r = MagicMock(spec=EndpointResolver)
    r.resolve_vlm.return_value = ("http://vlm/v1", "fake-vlm")
    r.resolve_llm.return_value = ("http://llm/v1", "fake-llm")
    r.vlm_retries = r.llm_retries = 3
    r.vlm_retry_backoff_s = r.llm_retry_backoff_s = 1.0
    return r


def _window_config(*, skip_existing: bool = False) -> PipelineConfig:
    return PipelineConfig(
        pipeline=PipelineSettings(),
        data=[
            SampleConfig(
                inputs=SampleInputsConfig(video_path="v.mp4"),
                output=SampleOutputConfig(out_dir="out"),
            )
        ],
        endpoints=EndpointsConfig(
            vlm=VlmEndpointConfig(url="http://vlm/v1", model="fake-vlm"),
            llm=LlmEndpointConfig(url="http://llm/v1", model="fake-llm"),
        ),
        mcq_generation=McqGenerationConfig(
            enabled=True,
            mode="window-vlm-llm",
            window_metadata_extraction=WindowMetadataExtractionConfig(
                single_window=True,
                skip_existing=skip_existing,
            ),
        ),
    )


def _logger() -> logging.Logger:
    return logging.getLogger("test_wrappers")


# ---------------------------------------------------------------------------
# _load_text
# ---------------------------------------------------------------------------


def test_load_text_none_returns_empty() -> None:
    assert _load_text(None, None) == ""


def test_load_text_existing_file(tmp_path: Path) -> None:
    f = tmp_path / "prompt.txt"
    f.write_text("hello prompt")
    assert _load_text(str(f), None) == "hello prompt"


def test_load_text_missing_returns_empty() -> None:
    assert _load_text("/nonexistent/file.txt", None) == ""


def test_load_text_relative_resolves_via_config_dir(tmp_path: Path) -> None:
    f = tmp_path / "scene.md"
    f.write_text("scene content")
    result = _load_text("scene.md", str(tmp_path))
    assert result == "scene content"


def test_load_text_relative_fallback_to_cwd_when_not_found(tmp_path: Path) -> None:
    # config_dir doesn't contain the file — falls back to path as-is → missing → ""
    result = _load_text("no_such_file.md", str(tmp_path))
    assert result == ""


# ---------------------------------------------------------------------------
# WindowVlmLlmGenerator.generate()
# ---------------------------------------------------------------------------


def test_window_vlm_llm_generate_calls_runner(tmp_path: Path) -> None:
    cfg = _window_config(skip_existing=False)
    with patch("mcq_generation.window_vlm_llm.WindowVlmLlmRunner") as MockRunner:
        gen = WindowVlmLlmGenerator(config=cfg, resolver=_resolver(), logger=_logger())
        mock_runner_instance = MockRunner.return_value
        gen.generate(tmp_path / "video.mp4", tmp_path)

    mock_runner_instance.run_single.assert_called_once()
    _, kwargs = mock_runner_instance.run_single.call_args
    assert kwargs["clip_path"] == tmp_path / "video.mp4"
    assert kwargs["output_dir"] == tmp_path
    assert kwargs["skip_existing"] is False


def test_window_vlm_llm_skip_existing_forwarded(tmp_path: Path) -> None:
    cfg = _window_config(skip_existing=True)
    with patch("mcq_generation.window_vlm_llm.WindowVlmLlmRunner") as MockRunner:
        gen = WindowVlmLlmGenerator(config=cfg, resolver=_resolver(), logger=_logger())
        mock_runner_instance = MockRunner.return_value
        gen.generate(tmp_path / "video.mp4", tmp_path)

    _, kwargs = mock_runner_instance.run_single.call_args
    assert kwargs["skip_existing"] is True


def test_window_vlm_llm_returns_bcq_path_when_bcq_only(tmp_path: Path) -> None:
    cfg = _window_config(skip_existing=False)
    with patch("mcq_generation.window_vlm_llm.WindowVlmLlmRunner") as MockRunner:
        gen = WindowVlmLlmGenerator(config=cfg, resolver=_resolver(), logger=_logger())
        mock_runner_instance = MockRunner.return_value

        def _write_bcq_only(**kwargs) -> None:
            task_dir = Path(kwargs["output_dir"]) / "task"
            task_dir.mkdir(parents=True, exist_ok=True)
            (task_dir / "bcq.json").write_text('{"items": [{"answer": "Yes"}]}', encoding="utf-8")

        mock_runner_instance.run_single.side_effect = _write_bcq_only
        result = gen.generate(tmp_path / "video.mp4", tmp_path)

    mock_runner_instance.run_single.assert_called_once()
    _, kwargs = mock_runner_instance.run_single.call_args
    assert kwargs["clip_path"] == tmp_path / "video.mp4"
    assert kwargs["output_dir"] == tmp_path
    assert kwargs["skip_existing"] is False
    assert not (tmp_path / "task" / "mcq.json").exists()
    assert result.mcq_json is None
    assert result.bcq_json == tmp_path / "task" / "bcq.json"
    assert result.fallback_json == tmp_path / "task" / "bcq.json"


# ---------------------------------------------------------------------------
# WindowDirectVlmGenerator.generate()
# ---------------------------------------------------------------------------


def test_window_direct_vlm_generate_calls_runner(tmp_path: Path) -> None:
    cfg = _window_config(skip_existing=False)
    cfg = cfg.model_copy(update={"mcq_generation": cfg.mcq_generation.model_copy(update={"mode": "window-direct-vlm"})})
    with patch("mcq_generation.window_direct_vlm.WindowDirectVlmRunner") as MockRunner:
        gen = WindowDirectVlmGenerator(config=cfg, resolver=_resolver(), logger=_logger())
        mock_runner_instance = MockRunner.return_value
        gen.generate(tmp_path / "video.mp4", tmp_path)

    mock_runner_instance.run_single.assert_called_once()
    _, kwargs = mock_runner_instance.run_single.call_args
    assert kwargs["clip_path"] == tmp_path / "video.mp4"
    assert kwargs["output_dir"] == tmp_path
    assert kwargs["skip_existing"] is False


def test_window_direct_vlm_skip_existing_forwarded(tmp_path: Path) -> None:
    cfg = _window_config(skip_existing=True)
    cfg = cfg.model_copy(update={"mcq_generation": cfg.mcq_generation.model_copy(update={"mode": "window-direct-vlm"})})
    with patch("mcq_generation.window_direct_vlm.WindowDirectVlmRunner") as MockRunner:
        gen = WindowDirectVlmGenerator(config=cfg, resolver=_resolver(), logger=_logger())
        mock_runner_instance = MockRunner.return_value
        gen.generate(tmp_path / "video.mp4", tmp_path)

    _, kwargs = mock_runner_instance.run_single.call_args
    assert kwargs["skip_existing"] is True
