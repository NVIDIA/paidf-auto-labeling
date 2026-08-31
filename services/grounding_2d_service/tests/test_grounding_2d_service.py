# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse

import pytest
from grounding_2d_service.main import Grounding2DService, build_config


def test_service_requires_canonical_data_entry_input() -> None:
    service = Grounding2DService()

    with pytest.raises(SystemExit) as exc_info:
        service.execute(argparse.Namespace(disabled=False), [])

    assert exc_info.value.code == "Pass --input or --input-file with at least one DataEntry."


def test_build_config_uses_vlm_and_sam3_args() -> None:
    args = argparse.Namespace(
        disabled=False,
        force_reprocess=True,
        caption=None,
        input_metadata_filename="input.json",
        vlm_endpoint_url="http://localhost:8000/v1",
        vlm_model="mock-vlm",
        system_prompt=None,
        max_tokens=1024,
        temperature=0.1,
        top_p=0.8,
        timeout_s=30.0,
        retries=1,
        retry_backoff_s=0.5,
        max_instances_per_expression=5,
        filter_ungroundable_expressions=True,
        expression_filter_policy_path=None,
        min_instance_score=0.6,
        min_bbox_area=100,
        sam3_model_cache_path="/models",
        sam3_gpu_ids="0",
        sam3_version="sam3.1",
        sam3_runtime="native",
        sam3_target_fps=10.0,
        sam3_session_reset_s=10.0,
        sam3_max_duration_s=30.0,
        sam3_write_annotated_media=False,
        sam3_annotated_media_label_style="name",
        sam3_annotated_media_mask_opacity=0,
    )

    config = build_config(args)

    assert config.force_reprocess is True
    assert config.vlm_endpoint_url == "http://localhost:8000/v1"
    assert config.vlm_model == "mock-vlm"
    assert config.filter_ungroundable_expressions is True
    assert config.min_instance_score == 0.6
    assert config.min_bbox_area == 100
    assert config.sam3_model_cache_path == "/models"
    assert config.sam3_gpu_ids == "0"
    assert config.sam3_version == "sam3.1"
    assert config.sam3_runtime == "native"
