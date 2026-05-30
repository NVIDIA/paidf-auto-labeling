# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for loading prompt files with config-relative path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def load_text(path_str: Optional[str], config_dir: Optional[str]) -> str:
    """Resolve and read a text file; returns empty string if path is absent or unreadable."""
    if not path_str:
        return ""
    p = Path(path_str)
    if not p.is_absolute() and config_dir:
        candidate = Path(config_dir) / p
        p = candidate if candidate.exists() else p
    if p.exists():
        return p.read_text(encoding="utf-8")
    return ""


def resolve_path(path_str: Optional[str], config_dir: Optional[str]) -> Optional[Path]:
    """Resolve a path string relative to config_dir; returns None if path_str is empty."""
    if not path_str:
        return None
    p = Path(path_str)
    if not p.is_absolute() and config_dir:
        candidate = Path(config_dir) / p
        return candidate if candidate.exists() else p
    return p
