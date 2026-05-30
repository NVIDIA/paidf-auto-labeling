# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests that the window MCQ generators clean up the captions/ working dir after generate()."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from al_utils.endpoint_resolver import EndpointResolver
from al_utils.schema.config import PipelineConfig
from al_utils.schema.data import SampleConfig, SampleInputsConfig, SampleOutputConfig
from al_utils.schema.endpoints import EndpointsConfig, LlmEndpointConfig, VlmEndpointConfig
from al_utils.schema.mcq import McqGenerationConfig, WindowMetadataExtractionConfig
from al_utils.schema.pipeline_settings import PipelineSettings
from mcq_generation.question_driven_vlm_llm import QuestionDrivenVlmLlmGenerator
from mcq_generation.window_direct_vlm import WindowDirectVlmGenerator
from mcq_generation.window_vlm_llm import WindowVlmLlmGenerator

REPO_ROOT = Path(__file__).parents[2]
QUESTION_BANK = REPO_ROOT / "cookbooks/traffic/question_bank.json"


def _make_config(mode: str = "window-vlm-llm") -> PipelineConfig:
    return PipelineConfig(
        pipeline=PipelineSettings(),
        data=[
            SampleConfig(
                inputs=SampleInputsConfig(video_path="video.mp4"),
                output=SampleOutputConfig(out_dir="out"),
            )
        ],
        endpoints=EndpointsConfig(
            vlm=VlmEndpointConfig(url="http://fake-vlm/v1", model="fake-vlm"),
            llm=LlmEndpointConfig(url="http://fake-llm/v1", model="fake-llm"),
        ),
        mcq_generation=McqGenerationConfig(
            enabled=True,
            mode=mode,
            window_metadata_extraction=WindowMetadataExtractionConfig(
                single_window=True,
                question_bank_file=str(QUESTION_BANK),
            ),
        ),
    )


def _make_resolver() -> MagicMock:
    r = MagicMock(spec=EndpointResolver)
    r.resolve_vlm.return_value = ("http://fake-vlm/v1", "fake-vlm")
    r.resolve_llm.return_value = ("http://fake-llm/v1", "fake-llm")
    r.llm_retries = 3
    r.llm_retry_backoff_s = 1.0
    r.vlm_retries = 3
    r.vlm_retry_backoff_s = 1.0
    return r


def _fake_run_single_creates_clip_dir(*, output_root, **kw):
    """Simulate runner creating a per-clip working directory (as the real runner does)."""
    clip_dir = output_root / "clip"
    clip_dir.mkdir(parents=True, exist_ok=True)


def _fake_run_single_creates_nonempty_clip_dir(*, output_root, **kw):
    """Simulate runner leaving files behind in the working dir."""
    clip_dir = output_root / "clip"
    clip_dir.mkdir(parents=True, exist_ok=True)
    (clip_dir / "leftover.json").write_text("{}")


def _assert_no_captions_dir(out_dir: Path) -> None:
    captions = out_dir / "captions"
    assert not captions.exists(), "captions/ dir should have been removed after generate()"


# ---------------------------------------------------------------------------
# WindowVlmLlmGenerator
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not QUESTION_BANK.exists(), reason="question bank file not found")
def test_window_vlm_llm_captions_cleaned_up(tmp_path: Path) -> None:
    # Patch _load_text so constructor doesn't need real prompt files
    with patch("mcq_generation.window_vlm_llm._load_text", return_value="fake prompt"):
        gen = WindowVlmLlmGenerator(_make_config("window-vlm-llm"), _make_resolver(), logging.getLogger("test"))

    gen._runner.run_single = MagicMock(side_effect=_fake_run_single_creates_clip_dir)

    (tmp_path / "video.mp4").write_bytes(b"fake")
    gen.generate(video_path=str(tmp_path / "video.mp4"), output_dir=tmp_path)

    _assert_no_captions_dir(tmp_path)


@pytest.mark.skipif(not QUESTION_BANK.exists(), reason="question bank file not found")
def test_window_vlm_llm_captions_cleaned_up_nonempty(tmp_path: Path) -> None:
    with patch("mcq_generation.window_vlm_llm._load_text", return_value="fake prompt"):
        gen = WindowVlmLlmGenerator(_make_config("window-vlm-llm"), _make_resolver(), logging.getLogger("test"))

    gen._runner.run_single = MagicMock(side_effect=_fake_run_single_creates_nonempty_clip_dir)

    (tmp_path / "video.mp4").write_bytes(b"fake")
    gen.generate(video_path=str(tmp_path / "video.mp4"), output_dir=tmp_path)

    _assert_no_captions_dir(tmp_path)


# ---------------------------------------------------------------------------
# WindowDirectVlmGenerator
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not QUESTION_BANK.exists(), reason="question bank file not found")
def test_window_direct_vlm_captions_cleaned_up(tmp_path: Path) -> None:
    with patch("mcq_generation.window_direct_vlm._load_text", return_value="fake prompt"):
        gen = WindowDirectVlmGenerator(_make_config("window-direct-vlm"), _make_resolver(), logging.getLogger("test"))

    gen._runner.run_single = MagicMock(side_effect=_fake_run_single_creates_clip_dir)

    (tmp_path / "video.mp4").write_bytes(b"fake")
    gen.generate(video_path=str(tmp_path / "video.mp4"), output_dir=tmp_path)

    _assert_no_captions_dir(tmp_path)


# ---------------------------------------------------------------------------
# QuestionDrivenVlmLlmGenerator
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not QUESTION_BANK.exists(), reason="question bank file not found")
def test_question_driven_captions_cleaned_up(tmp_path: Path) -> None:
    gen = QuestionDrivenVlmLlmGenerator(
        _make_config("question-driven-vlm-llm"),
        _make_resolver(),
        logging.getLogger("test"),
    )

    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "scene_prompt.used.md").write_text("describe the scene")
    (prompts_dir / "mcq_prompt.used.md").write_text("answer these questions")

    (tmp_path / "video.mp4").write_bytes(b"fake")

    with patch("mcq_generation.question_driven_vlm_llm.WindowVlmLlmRunner") as MockRunner:
        mock_instance = MagicMock()
        mock_instance.run_single.side_effect = _fake_run_single_creates_clip_dir
        MockRunner.return_value = mock_instance

        gen.generate(video_path=str(tmp_path / "video.mp4"), output_dir=tmp_path)

    _assert_no_captions_dir(tmp_path)


@pytest.mark.skipif(not QUESTION_BANK.exists(), reason="question bank file not found")
def test_question_driven_captions_cleaned_up_nonempty(tmp_path: Path) -> None:
    """captions/ is removed even when the runner leaves files inside."""
    gen = QuestionDrivenVlmLlmGenerator(
        _make_config("question-driven-vlm-llm"),
        _make_resolver(),
        logging.getLogger("test"),
    )

    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "scene_prompt.used.md").write_text("scene")
    (prompts_dir / "mcq_prompt.used.md").write_text("mcq")

    (tmp_path / "video.mp4").write_bytes(b"fake")

    with patch("mcq_generation.question_driven_vlm_llm.WindowVlmLlmRunner") as MockRunner:
        mock_instance = MagicMock()
        mock_instance.run_single.side_effect = _fake_run_single_creates_nonempty_clip_dir
        MockRunner.return_value = mock_instance

        gen.generate(video_path=str(tmp_path / "video.mp4"), output_dir=tmp_path)

    _assert_no_captions_dir(tmp_path)
