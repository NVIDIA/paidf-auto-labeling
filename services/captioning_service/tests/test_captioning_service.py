# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import re
from pathlib import Path
from unittest.mock import patch

import pytest
from captioning_service.main import CaptioningService, build_config
from core import DataEntry
from core.policy import EmptyOutputPolicy


def test_captioning_service_empty_input_exits_cleanly() -> None:
    service = CaptioningService()

    with pytest.raises(SystemExit) as exc_info:
        service.execute(argparse.Namespace(), [])

    assert exc_info.value.code == "Pass --input or --input-file with at least one DataEntry."


def test_build_config_maps_cli_args() -> None:
    args = argparse.Namespace(
        disabled=False,
        input_source="tracking",
        image_group_dir="/data/identity",
        max_group_images=5,
        vlm_provider="gemini",
        vlm_endpoint_url="https://example.invalid",
        vlm_model="gemini-test",
        enable_llm_summary=True,
        llm_provider="openai-compatible",
        llm_endpoint_url="http://localhost:8000/v1",
        llm_model="llm-test",
        prompt_text="video prompt",
        prompt_file=None,
        image_prompt_text="image prompt",
        image_prompt_file=None,
        summary_prompt_text="summary prompt",
        summary_prompt_file=None,
        system_prompt="system",
        window_seconds=5.0,
        window_frames=0,
        remainder_threshold=2,
        single_window=True,
        sampling_fps=2.0,
        max_frames=4,
        resolution=512,
        media_mode="auto",
        max_tokens=256,
        summary_input_token_budget=512,
        temperature=0.1,
        top_p=0.8,
        timeout_s=30.0,
        retries=1,
        retry_backoff_s=0.5,
        preserve_raw_model_output=True,
        no_contextual=True,
        sidecar_filename="custom.json",
    )

    config = build_config(args)

    assert config.enabled is True
    assert config.input_source == "tracking"
    assert config.image_group_dir == "/data/identity"
    assert config.max_group_images == 5
    assert config.vlm_provider == "gemini"
    assert config.vlm_endpoint_url == "https://example.invalid"
    assert config.vlm_model == "gemini-test"
    assert config.enable_llm_summary is True
    assert config.llm_provider == "openai-compatible"
    assert config.llm_endpoint_url == "http://localhost:8000/v1"
    assert config.llm_model == "llm-test"
    assert config.prompt_text == "video prompt"
    assert config.image_prompt_text == "image prompt"
    assert config.summary_prompt_text == "summary prompt"
    assert config.system_prompt == "system"
    assert config.window_seconds == 5.0
    assert config.window_frames == 0
    assert config.remainder_threshold == 2
    assert config.single_window is True
    assert config.sampling_fps == 2.0
    assert config.max_frames == 4
    assert config.resolution == 512
    assert config.media_mode == "auto"
    assert config.max_tokens == 256
    assert config.summary_input_token_budget == 512
    assert config.temperature == 0.1
    assert config.top_p == 0.8
    assert config.timeout_s == 30.0
    assert config.retries == 1
    assert config.retry_backoff_s == 0.5
    assert config.preserve_raw_model_output is True
    assert config.write_contextual is False
    assert config.sidecar_filename == "custom.json"
    assert not hasattr(config, "write_visual_qa_sidecar")
    assert not hasattr(config, "visual_qa_generation_mode")


def test_parser_defaults_enable_auto_captioning() -> None:
    parser = argparse.ArgumentParser()
    CaptioningService().add_service_args(parser)

    args = parser.parse_args([])
    config = build_config(args)

    assert config.enabled is True
    assert config.input_source == "auto"
    assert config.image_group_dir is None
    assert config.max_group_images == 0
    assert config.media_mode == "auto"
    assert config.preserve_raw_model_output is False
    assert not hasattr(config, "write_visual_qa_sidecar")
    assert not hasattr(args, "enable_visual_qa")


def test_parser_rejects_visual_qa_options() -> None:
    parser = argparse.ArgumentParser()
    CaptioningService().add_service_args(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--enable-visual-qa"])


def test_execute_invokes_linear_pipeline_run() -> None:
    service = CaptioningService()
    parser = argparse.ArgumentParser()
    service.add_service_args(parser)
    args = parser.parse_args([])
    entries = [
        DataEntry(
            id="entry-1",
            media_path="/data/source/clip.mp4",
            data_path="/data/scene",
        )
    ]

    with (
        patch("captioning_service.main.CaptioningTask") as task_cls,
        patch("captioning_service.main.DaftValidationTask") as validation_task_cls,
        patch("captioning_service.main.LinearPipeline") as pipeline_cls,
    ):
        pipeline = pipeline_cls.return_value
        pipeline.run.return_value = entries

        service.execute(args, entries)

    task_cls.assert_called_once()
    pipeline_cls.assert_called_once()
    _, kwargs = pipeline_cls.call_args
    assert kwargs["name"] == "captioning_pipeline"
    assert kwargs["policy"] is EmptyOutputPolicy.FAIL
    assert kwargs["tasks"] == [task_cls.return_value, validation_task_cls.return_value]
    pipeline.run.assert_called_once_with(entries)


def test_execute_disabled_short_circuits_before_input_validation() -> None:
    service = CaptioningService()

    with (
        patch("captioning_service.main.CaptioningTask") as task_cls,
        patch("captioning_service.main.DaftValidationTask") as validation_task_cls,
        patch("captioning_service.main.LinearPipeline") as pipeline_cls,
    ):
        service.execute(argparse.Namespace(disabled=True), [])

    task_cls.assert_not_called()
    validation_task_cls.assert_not_called()
    pipeline_cls.assert_not_called()


def test_container_uses_controlled_media_toolchain() -> None:
    dockerfile = Path(__file__).parents[1] / "docker" / "Dockerfile"
    text = dockerfile.read_text(encoding="utf-8")
    excluded_packages = {
        name.lower() for name in re.findall(r"--no-install-package\s+([^\s\\]+)", text)
    }

    # FFmpeg and the OpenCV/PyAV wheels come from the shared media base, so the
    # service no longer compiles them; it installs the prebuilt wheels instead.
    assert "FROM ${BUILDER_BASE_IMAGE} AS builder" in text
    assert "ffmpeg-install" not in text
    assert "media_toolchain.py python-install" not in text
    assert "--no-install-package av" in text
    assert "opencv-python" in excluded_packages
    assert "opencv-python-headless" in excluded_packages
    assert '--find-links "${UPA_MEDIA_WHEEL_DIR}"' in text
    assert "opencv-python-headless av" in text
    # The builder and the final runtime image both still verify the codec policy.
    assert text.count("media_toolchain.py verify") == 2
    assert "apt-get install -y --no-install-recommends ffmpeg" not in text
