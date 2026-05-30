#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# SR multi-GPU tests:
# - test_sr_multigpu_config_is_parsed_correctly: verify SeedVR2Resolver has correct sp_size/gpu_list.
#

from __future__ import annotations

import logging
from pathlib import Path

import config.loader
import config.schema
import pytest
import torch
from al_utils.common import resolve_gpu_list
from sr_runner.seedvr2 import SeedVR2Resolver

_CUDA_DEVICE_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
PIPELINE_EXAMPLE_CONFIG = CONFIGS_DIR / "pipeline_example.yaml"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _base_overrides(tmp_path: Path, video_path: Path, gpu_ids: str = "0,1") -> list[str]:
    return [
        f"data.0.inputs.video_path={video_path}",
        f"data.0.output.out_dir={tmp_path / 'out'}",
        f"data.0.output.log_dir={tmp_path / 'out' / 'logs'}",
        "endpoints.vlm.url=http://example.invalid/v1",
        "endpoints.vlm.model=dummy-vlm",
        "endpoints.llm.url=http://example.invalid/v1",
        "endpoints.llm.model=dummy-llm",
        "detection_and_tracking.enabled=false",
        "vlm_json.enabled=false",
        "mcq_generation.enabled=false",
        "pipeline.use_multi_gpu=true",
        f"pipeline.gpu_ids={gpu_ids}",
    ]


def test_sr_multigpu_config_is_parsed_correctly(tmp_path: Path) -> None:
    """With use_multi_gpu=true and gpu_ids=0,1, SeedVR2Resolver must be initialized
    with sp_size=2 and 2 GPUs in its gpu_list. Does not require real hardware."""
    if not PIPELINE_EXAMPLE_CONFIG.exists():
        pytest.skip(f"Config not found: {PIPELINE_EXAMPLE_CONFIG}")

    video = tmp_path / "sr_input.mp4"
    video.write_bytes(b"")
    logger = logging.getLogger(__name__)
    overrides = _base_overrides(tmp_path, video, gpu_ids="0,1")

    cfg, _ = config.loader.load_config_with_overrides(str(PIPELINE_EXAMPLE_CONFIG), overrides, logger=logger)
    assert cfg is not None
    validated = config.schema.validate_schema(cfg, logger=logger)
    assert validated is not None

    # Verify schema parsed the multi-GPU knobs correctly.
    assert validated.super_resolution is not None
    assert validated.pipeline.use_multi_gpu is True
    assert str(validated.pipeline.gpu_ids) == "0,1"

    # Verify SeedVR2Resolver derives sp_size=2 and gpu_list=[0,1] from config.
    gpu_list = resolve_gpu_list(validated.pipeline.gpu_ids)
    resolver = SeedVR2Resolver(config=validated, logger=logger, gpu_list=gpu_list)
    assert resolver._sp_size == 2
    assert resolver._gpu_list == [0, 1], f"Expected [0, 1], got {resolver._gpu_list}"
