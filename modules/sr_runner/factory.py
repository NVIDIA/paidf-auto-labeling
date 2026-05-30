# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Super-resolution runner factory.

Dispatches on ``config.super_resolution.model`` to the appropriate runner class.
"""

from __future__ import annotations

import logging

from al_utils.common import resolve_gpu_list
from al_utils.schema.config import PipelineConfig
from sr_runner.base import BaseSuperResolver
from sr_runner.seedvr2 import SeedVR2Resolver


def create_sr_runner(
    config: PipelineConfig,
    logger: logging.Logger,
) -> BaseSuperResolver:
    """Create the appropriate SR runner from configuration.

    Args:
        config: Full pipeline config (Pydantic model).
        logger: Logger instance.

    Returns:
        Configured ``BaseSuperResolver`` instance.

    Raises:
        ValueError: If ``super_resolution`` section is absent or model is unknown.
    """
    sr = config.super_resolution
    if sr is None:
        raise ValueError("'super_resolution' section is required to create an SR runner")

    gpu_list = resolve_gpu_list(config.pipeline.gpu_ids)
    model = str(sr.model)
    if model == "seedvr2":
        return SeedVR2Resolver(config=config, logger=logger, gpu_list=gpu_list)

    raise ValueError(f"Unknown super_resolution.model: {model!r}")
