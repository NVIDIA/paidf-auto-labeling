# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from al_utils.common import get


def validate_environment(logger: logging.Logger, config: Dict[str, Any]) -> None:
    """
    Lightweight env validation: warn about missing env vars that are commonly needed.
    """
    warning_only: List[str] = ["LOG_LEVEL"]
    optional: List[str] = ["HF_TOKEN"]

    # VLM endpoints may or may not require auth.
    # Full priority: VLM_API_KEY > NVIDIA_API_KEY > OPENAI_API_KEY > "EMPTY".
    # Warn (debug-level) only when none of the keys are set.
    vlm_enabled = bool(get(config, "vlm_json.enabled", False))
    if vlm_enabled:
        if not any(os.getenv(k) for k in ("VLM_API_KEY", "NVIDIA_API_KEY", "OPENAI_API_KEY")):
            optional.append("VLM_API_KEY")

    for var in warning_only:
        if not os.getenv(var):
            logger.warning(f"Environment variable {var} is not set")
    for var in optional:
        if not os.getenv(var):
            logger.debug(f"Environment variable {var} is not set")
