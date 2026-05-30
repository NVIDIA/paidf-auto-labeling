# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Detection & tracking factory.

Dispatches on ``config.detection_and_tracking`` to the appropriate tracker class.
Raises ``ValueError`` on invalid config; never returns ``None``
(the call site initialises ``det_tracker = None`` and only calls this when
the section is enabled).
"""

from __future__ import annotations

import logging

from al_utils.common import resolve_gpu_list
from al_utils.schema.config import PipelineConfig
from detection_and_tracking.base import BaseTracker
from detection_and_tracking.rfdetr_tracking import RFDetrTracker


def create_tracker(
    config: PipelineConfig,
    logger: logging.Logger,
) -> BaseTracker:
    """Create the appropriate tracker from configuration.

    Currently the only supported tracker backend is RF-DETR.

    Args:
        config: Full pipeline config (Pydantic model).
        logger: Logger instance.

    Returns:
        Configured ``BaseTracker`` instance with model loaded.

    Raises:
        ValueError: If ``detection_and_tracking`` section is absent or config
                    is otherwise invalid (e.g. ReID weights not found).
    """
    dt = config.detection_and_tracking
    if dt is None:
        raise ValueError("'detection_and_tracking' section is required to create a tracker")

    gpu_list = resolve_gpu_list(config.pipeline.gpu_ids)
    model = str(dt.model)
    if model == "rfdetr":
        return RFDetrTracker(config=config, logger=logger, gpu_list=gpu_list)

    raise ValueError(f"Unknown detection_and_tracking.model: {model!r}")
