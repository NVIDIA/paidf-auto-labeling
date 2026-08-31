# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logging utilities for the Auto-Labeling Framework.

All framework loggers are children of a single root logger named
``pseudo_annotation``. Components retrieve their logger via ``get_logger``
and inherit their effective level from the root logger.
"""

import logging
import sys
from typing import Literal

LOGGER_NAMESPACE = "pseudo_annotation"
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def get_logger(
    name: str,
    kind: Literal["task", "pipeline", "service"] | None = None,
) -> logging.Logger:
    """
    Return a child logger under the framework's root logger.

    The returned logger has no handlers or level of its own; records
    propagate up to the root logger, which owns the formatter and the
    effective level set by ``apply_log_level``.

    Args:
        name: Logger name (typically the component instance name).
        kind: Optional component kind (``"task"``, ``"pipeline"``,
            ``"service"``) used to namespace the logger.

    Returns:
        A child logger of ``pseudo_annotation``.
    """
    _configure_root()
    full_name = f"{LOGGER_NAMESPACE}.{kind}.{name}" if kind else f"{LOGGER_NAMESPACE}.{name}"
    return logging.getLogger(full_name)


def apply_log_level(level: str) -> None:
    """
    Set the effective log level for every logger created by ``get_logger``.

    Args:
        level: One of DEBUG, INFO, WARNING, ERROR, CRITICAL.

    Raises:
        ValueError: If ``level`` is not a recognized logging level.
    """
    if level not in LOG_LEVELS:
        raise ValueError(f"Invalid logging level: {level}")
    _configure_root()
    logging.getLogger(LOGGER_NAMESPACE).setLevel(getattr(logging, level))


def _configure_root() -> None:
    """Idempotently install the framework's handler and default level on the root logger."""
    root = logging.getLogger(LOGGER_NAMESPACE)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        root.addHandler(handler)
        root.setLevel(logging.INFO)
