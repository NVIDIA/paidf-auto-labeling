# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct unit tests for QuestionDrivenVlmLlmGenerator.run_pre_step().

run_pre_step() runs an LLM-only prompt-generation phase before SR/tracking.
It writes scene_prompt.used.md and mcq_prompt.used.md into
<out_dir>/prompts/.
"""

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

REPO_ROOT = Path(__file__).parents[2]
QUESTION_BANK = REPO_ROOT / "cookbooks/traffic/question_bank.json"


def _make_config() -> PipelineConfig:
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
            mode="question-driven-vlm-llm",
            window_metadata_extraction=WindowMetadataExtractionConfig(
                single_window=True,
                question_bank_file=str(QUESTION_BANK),
            ),
        ),
    )


def _make_resolver() -> MagicMock:
    resolver = MagicMock(spec=EndpointResolver)
    resolver.resolve_vlm.return_value = ("http://fake-vlm/v1", "fake-vlm")
    resolver.resolve_llm.return_value = ("http://fake-llm/v1", "fake-llm")
    resolver.llm_retries = 3
    resolver.llm_retry_backoff_s = 1.0
    resolver.vlm_retries = 3
    resolver.vlm_retry_backoff_s = 1.0
    return resolver


@pytest.mark.skipif(not QUESTION_BANK.exists(), reason="question bank file not found")
def test_run_pre_step_writes_prompt_files(tmp_path: Path) -> None:
    """run_pre_step() must write scene_prompt.used.md and mcq_prompt.used.md."""
    config = _make_config()
    resolver = _make_resolver()
    logger = logging.getLogger("test_qd_pre_step")

    gen = QuestionDrivenVlmLlmGenerator(config, resolver, logger)

    fake_scene_response = {"prompt_text": "Describe the vehicles in each window."}

    with patch(
        "mcq_generation.question_driven_vlm_llm.generate_vlm_scene_prompt",
        return_value=fake_scene_response,
    ) as mock_llm:
        gen.run_pre_step(tmp_path, MagicMock())

    # LLM was called exactly once (prompt generation phase)
    mock_llm.assert_called_once()

    prompts_dir = tmp_path / "prompts"
    assert prompts_dir.is_dir(), "prompts directory not created"

    scene_used = prompts_dir / "scene_prompt.used.md"
    mcq_used = prompts_dir / "mcq_prompt.used.md"
    assert scene_used.exists(), "scene_prompt.used.md not written"
    assert mcq_used.exists(), "mcq_prompt.used.md not written"
    assert (prompts_dir / "prompts.used.json").exists(), "prompts.used.json not written"
    assert not (prompts_dir / "scene_prompt.generated_by_llm.md").exists()
    assert not (prompts_dir / "mcq_prompt.bank_injected.md").exists()
    assert not (prompts_dir / "mcq_prompt.augmented.md").exists()

    # Content matches the mocked LLM response
    assert fake_scene_response["prompt_text"] in scene_used.read_text()


@pytest.mark.skipif(not QUESTION_BANK.exists(), reason="question bank file not found")
def test_run_pre_step_writes_only_non_duplicate_qd_artifacts(tmp_path: Path) -> None:
    config = _make_config()
    config = config.model_copy(
        update={
            "mcq_generation": config.mcq_generation.model_copy(
                update={
                    "window_metadata_extraction": config.mcq_generation.window_metadata_extraction.model_copy(
                        update={"append_mapper_rules": True}
                    )
                }
            )
        }
    )
    resolver = _make_resolver()
    logger = logging.getLogger("test_qd_pre_step_artifacts")

    gen = QuestionDrivenVlmLlmGenerator(config, resolver, logger)

    with (
        patch(
            "mcq_generation.question_driven_vlm_llm.generate_vlm_scene_prompt",
            return_value={"prompt_text": "Describe the vehicles in each window."},
        ),
        patch(
            "mcq_generation.question_driven_vlm_llm.generate_mapper_rules",
            return_value={"rules_text": "Only answer from visible evidence."},
        ),
    ):
        gen.run_pre_step(tmp_path, MagicMock())

    prompts_dir = tmp_path / "prompts"
    assert (prompts_dir / "scene_prompt.used.md").exists()
    assert (prompts_dir / "mcq_prompt.used.md").exists()
    assert (prompts_dir / "mcq_prompt.bank_injected.md").exists()
    assert (prompts_dir / "mcq_prompt.mapper_rules.generated_by_llm.md").exists()
    assert not (prompts_dir / "scene_prompt.generated_by_llm.md").exists()
    assert not (prompts_dir / "mcq_prompt.augmented.md").exists()


@pytest.mark.skipif(not QUESTION_BANK.exists(), reason="question bank file not found")
def test_run_pre_step_skip_existing(tmp_path: Path) -> None:
    """When skip_existing=True and outputs already exist, LLM is not called again."""
    config = _make_config()
    # Override to enable skip_existing
    config = config.model_copy(
        update={
            "mcq_generation": config.mcq_generation.model_copy(
                update={
                    "window_metadata_extraction": config.mcq_generation.window_metadata_extraction.model_copy(
                        update={"skip_existing": True}
                    )
                }
            )
        }
    )
    resolver = _make_resolver()
    logger = logging.getLogger("test_qd_pre_step_skip")

    gen = QuestionDrivenVlmLlmGenerator(config, resolver, logger)

    # Pre-create the expected output files
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "scene_prompt.used.md").write_text("existing scene prompt")
    (prompts_dir / "mcq_prompt.used.md").write_text("existing mcq prompt")

    with patch("mcq_generation.question_driven_vlm_llm.generate_vlm_scene_prompt") as mock_llm:
        gen.run_pre_step(tmp_path, MagicMock())
        mock_llm.assert_not_called()  # skip_existing=True prevents LLM call
