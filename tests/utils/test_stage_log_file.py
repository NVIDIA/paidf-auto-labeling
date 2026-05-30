# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for al_utils.common.stage_log_file."""

from __future__ import annotations

import logging
from pathlib import Path

from al_utils.common import stage_log_file


def test_noop_when_log_dir_is_none() -> None:
    """stage_log_file(None) is a no-op — no files created, no errors."""
    logger = logging.getLogger("test_noop")
    with stage_log_file("tracking", None):
        logger.info("should not be captured anywhere")


def test_creates_named_log_file(tmp_path: Path) -> None:
    with stage_log_file("tracking", tmp_path):
        pass
    assert (tmp_path / "tracking.log").exists()


def test_log_file_contains_start_end_header(tmp_path: Path) -> None:
    with stage_log_file("vlm_json", tmp_path):
        pass
    content = (tmp_path / "vlm_json.log").read_text()
    assert "[vlm_json] START" in content
    assert "[vlm_json] END" in content


def test_logger_messages_captured_in_log_file(tmp_path: Path) -> None:
    logger = logging.getLogger("test_capture")
    logger.setLevel(logging.DEBUG)
    with stage_log_file("mcq", tmp_path):
        logger.info("hello from mcq stage")
    content = (tmp_path / "mcq.log").read_text()
    assert "hello from mcq stage" in content


def test_handler_removed_after_context_exits(tmp_path: Path) -> None:
    root = logging.getLogger()
    before = len(root.handlers)
    with stage_log_file("tracking", tmp_path):
        assert len(root.handlers) == before + 1
    assert len(root.handlers) == before


def test_handler_removed_on_exception(tmp_path: Path) -> None:
    root = logging.getLogger()
    before = len(root.handlers)
    try:
        with stage_log_file("tracking", tmp_path):
            raise ValueError("boom")
    except ValueError:
        pass
    assert len(root.handlers) == before
