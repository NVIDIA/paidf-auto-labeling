# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging


def setup_runner_logger(module_name: str, verbose: bool) -> logging.Logger:
    """
    Create or retrieve a module-level logger with a StreamHandler.

    Idempotent: returns the existing logger if it already has handlers.
    Sets propagate=False to avoid duplicate messages when the root logger
    is also configured by the caller (e.g. cli.py's basicConfig).
    """
    logger = logging.getLogger(module_name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    h = logging.StreamHandler()
    h.setLevel(logging.DEBUG if verbose else logging.INFO)
    h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(h)
    return logger
