# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from unittest.mock import MagicMock

from config.validate import validate_environment


def test_validate_environment(monkeypatch):
    logger = MagicMock(spec=logging.Logger)
    # Missing LOG_LEVEL should warn
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    validate_environment(logger, {})
    logger.warning.assert_called_with("Environment variable LOG_LEVEL is not set")

    # If VLM enabled and no API keys set, debug-warn about VLM_API_KEY (first in priority chain)
    monkeypatch.delenv("VLM_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    validate_environment(logger, {"vlm_json": {"enabled": True}})
    logger.debug.assert_called_with("Environment variable VLM_API_KEY is not set")
