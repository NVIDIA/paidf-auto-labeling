# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import Union

from al_utils.path_sanitize import expand_artifact_root_token


def resolve_path(value: Union[str, Path], *, config_dir: Path, repo_root: Path) -> str:
    v = expand_artifact_root_token(str(value), repo_root=repo_root)
    if v.startswith(("msc://", "s3://", "gs://", "ais://", "http://", "https://")):
        return v
    p = Path(v).expanduser()
    if p.is_absolute():
        return str(p)
    if v.startswith(("./", "../")):
        return str((config_dir / p).resolve())
    return str((repo_root / p).resolve())


def resolve_input_path(value: Union[str, Path], *, config_dir: Path, repo_root: Path) -> str:
    """Resolve an *input* path by checking repo_root first, then config_dir.

    This is intentionally existence-aware to avoid surprising behavior when a runnable
    config is saved under `output/.../config.yaml`:
    - Try interpreting relative paths against repo_root first.
    - If the resulting path does not exist, fall back to config_dir.
    - Remote paths are returned unchanged.
    """
    v = expand_artifact_root_token(str(value), repo_root=repo_root)
    if v.startswith(("msc://", "s3://", "gs://", "ais://", "http://", "https://")):
        return v

    p = Path(v).expanduser()
    if p.is_absolute():
        return str(p)

    # Explicit config-relative paths always win.
    if v.startswith(("./", "../")):
        return str((config_dir / p).resolve())

    cand_root = (repo_root / p).resolve()
    if cand_root.exists():
        return str(cand_root)

    cand_cfg = (config_dir / p).resolve()
    if cand_cfg.exists():
        return str(cand_cfg)

    # Deterministic fallback for error messages: prefer repo_root.
    return str(cand_root)
