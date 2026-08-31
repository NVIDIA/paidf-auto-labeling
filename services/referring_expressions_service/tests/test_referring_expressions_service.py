# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse

import pytest
from referring_expressions_service.main import ReferringExpressionsService, build_config


def test_service_requires_data_entries() -> None:
    service = ReferringExpressionsService()
    with pytest.raises(SystemExit):
        service.execute(argparse.Namespace(disabled=False), [])


def test_build_config_maps_cli_args() -> None:
    args = argparse.Namespace(
        disabled=False,
        force_reprocess=True,
        vlm_endpoint_url="http://localhost:8001/v1",
        vlm_model="mock-vlm",
        system_prompt=None,
        max_tokens=1024,
        temperature=0.1,
        top_p=0.8,
        timeout_s=30.0,
        retries=1,
        retry_backoff_s=0.5,
        frame_number=0,
        draw_box_overlay=True,
        min_match_iou=0.3,
        enable_grouped_expressions=False,
        enable_double_check=False,
    )
    config = build_config(args)
    assert config.force_reprocess is True
    assert config.vlm_endpoint_url == "http://localhost:8001/v1"
    assert config.vlm_model == "mock-vlm"
    assert config.draw_box_overlay is True
    assert config.min_match_iou == 0.3
