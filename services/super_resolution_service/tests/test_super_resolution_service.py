# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import re
from pathlib import Path
from unittest.mock import patch

import pytest
from core import DataEntry
from core.policy import EmptyOutputPolicy
from super_resolution_service.main import SuperResolutionService, build_config


def test_super_resolution_service_empty_input_exits_cleanly() -> None:
    service = SuperResolutionService()

    with pytest.raises(SystemExit) as exc_info:
        service.execute(argparse.Namespace(), [])

    assert exc_info.value.code == "Pass --input or --input-file with at least one DataEntry."


def test_build_config_maps_cli_args() -> None:
    args = argparse.Namespace(
        resolver="seedvr2",
        disabled=False,
        variant="seedvr2_7b",
        seed=123,
        res_h=720,
        res_w=1280,
        resolution_policy="auto",
        min_input_short_side=540,
        min_input_long_side=960,
        window_frames=64,
        overlap_frames=16,
        out_fps=30.0,
        gpu_ids="0,1",
        use_multi_gpu=True,
        model_cache_path="/models",
        seedvr_root="/opt/seedvr",
        allow_checkpoint_download=True,
        keep_intermediates=True,
        empty_output_policy="fail",
        command_timeout_s=3600.0,
    )

    config = build_config(args)

    assert config.enabled is True
    assert config.resolver == "seedvr2"
    assert config.resolution_policy == "auto"
    assert config.min_input_short_side == 540
    assert config.min_input_long_side == 960
    assert config.seedvr2.variant == "seedvr2_7b"
    assert config.seedvr2.seed == 123
    assert config.seedvr2.window_frames == 64
    assert config.seedvr2.overlap_frames == 16
    assert config.seedvr2.out_fps == 30.0
    assert config.seedvr2.gpu_ids == "0,1"
    assert config.seedvr2.use_multi_gpu is True
    assert config.seedvr2.model_cache_path == "/models"
    assert config.seedvr2.seedvr_root == "/opt/seedvr"
    assert config.seedvr2.allow_checkpoint_download is True
    assert config.seedvr2.keep_intermediates is True
    assert config.seedvr2.empty_output_policy == "fail"
    assert config.seedvr2.command_timeout_s == 3600.0


def test_build_config_supports_disabled_mode() -> None:
    args = argparse.Namespace(
        resolver="seedvr2",
        disabled=True,
        variant="seedvr2_3b",
        seed=42,
        res_h=720,
        res_w=1280,
        resolution_policy="always",
        min_input_short_side=720,
        min_input_long_side=1280,
        window_frames=128,
        overlap_frames=64,
        out_fps=None,
        gpu_ids="all",
        use_multi_gpu=False,
        model_cache_path=None,
        seedvr_root=None,
        allow_checkpoint_download=False,
        keep_intermediates=False,
        empty_output_policy="warn",
        command_timeout_s=None,
    )

    config = build_config(args)

    assert config.enabled is False
    assert config.seedvr2.gpu_ids == "all"


def test_execute_invokes_linear_pipeline_run() -> None:
    service = SuperResolutionService()
    entries = [DataEntry(id="entry-1", media_path="/data/source/clip.mp4", data_path="/data/scene")]
    args = argparse.Namespace(
        resolver="seedvr2",
        disabled=False,
        variant="seedvr2_3b",
        seed=42,
        res_h=720,
        res_w=1280,
        resolution_policy="always",
        min_input_short_side=720,
        min_input_long_side=1280,
        window_frames=128,
        overlap_frames=64,
        out_fps=None,
        gpu_ids="all",
        use_multi_gpu=False,
        model_cache_path=None,
        seedvr_root=None,
        allow_checkpoint_download=False,
        keep_intermediates=False,
        empty_output_policy="fail",
        command_timeout_s=None,
    )

    with (
        patch("super_resolution_service.main.SuperResolutionTask") as task_cls,
        patch("super_resolution_service.main.DaftValidationTask") as validation_task_cls,
        patch("super_resolution_service.main.LinearPipeline") as pipeline_cls,
    ):
        pipeline = pipeline_cls.return_value
        pipeline.run.return_value = entries

        service.execute(args, entries)

    task_cls.assert_called_once()
    pipeline_cls.assert_called_once()
    _, kwargs = pipeline_cls.call_args
    assert kwargs["name"] == "super_resolution_pipeline"
    assert kwargs["policy"] is EmptyOutputPolicy.FAIL
    assert kwargs["tasks"] == [task_cls.return_value, validation_task_cls.return_value]
    pipeline.run.assert_called_once_with(entries)


def test_execute_disabled_short_circuits_before_input_validation() -> None:
    service = SuperResolutionService()

    with (
        patch("super_resolution_service.main.SuperResolutionTask") as task_cls,
        patch("super_resolution_service.main.DaftValidationTask") as validation_task_cls,
        patch("super_resolution_service.main.LinearPipeline") as pipeline_cls,
    ):
        service.execute(argparse.Namespace(disabled=True), [])

    task_cls.assert_not_called()
    validation_task_cls.assert_not_called()
    pipeline_cls.assert_not_called()


def test_dockerfile_bakes_seedvr_source() -> None:
    dockerfile = Path(__file__).parents[1] / "docker" / "Dockerfile"
    text = dockerfile.read_text(encoding="utf-8")

    assert re.search(r"\bARG\s+SEEDVR_GIT_SHA\s*=", text)
    assert re.search(r"\bgit\s+fetch\b.*\borigin\b", text)
    assert "color_fix.py" in text
    assert "rm -rf /opt/seedvr/.git" in text
    assert re.search(r"\bCOPY\s+--from=builder\b.*?/opt/seedvr\b", text)


def test_dockerfile_uses_approved_vp9_codec_profile() -> None:
    dockerfile = Path(__file__).parents[1] / "docker" / "Dockerfile"
    text = dockerfile.read_text(encoding="utf-8")

    assert "build_restricted_ffmpeg.sh" in text
    assert "media_toolchain.py ffmpeg-install --profile vp9-output" in text
    assert "--component pyav" in text
    assert text.count("media_toolchain.py verify") == 2
    assert "source=/opt/media-runtime" in text
