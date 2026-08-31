# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest
from core.utils.logging import LOGGER_NAMESPACE, apply_log_level, get_logger


def test_apply_log_level_debug() -> None:
    apply_log_level(level="DEBUG")
    assert logging.getLogger(LOGGER_NAMESPACE).level == logging.DEBUG


def test_apply_log_level_info() -> None:
    apply_log_level(level="DEBUG")
    apply_log_level(level="INFO")
    assert logging.getLogger(LOGGER_NAMESPACE).level == logging.INFO


def test_apply_log_level_invalid_raises() -> None:
    with pytest.raises(ValueError, match="Invalid logging level"):
        apply_log_level(level="LOUD")


def test_get_logger_inherits_root_level() -> None:
    apply_log_level(level="WARNING")
    child = get_logger("ChildA", kind="task")
    assert child.getEffectiveLevel() == logging.WARNING


def test_apply_log_level_propagates_to_existing_children() -> None:
    child = get_logger("ChildB", kind="pipeline")
    apply_log_level(level="DEBUG")
    assert child.getEffectiveLevel() == logging.DEBUG
    apply_log_level(level="ERROR")
    assert child.getEffectiveLevel() == logging.ERROR
