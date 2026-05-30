# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical on-disk layout of a DAFT scene."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ScenePaths:
    """Canonical DAFT file paths for a single scene.

    A DAFT scene has a fixed on-disk layout::

        <scene>/
        ├── raw/<media_id>.<ext>               # analyzed media (local symlink or remote-staged copy)
        ├── contextual/
        │   ├── video.json | image.json        # scene-level metadata (one or the other)
        │   ├── events.json                    # temporal events (video scenes only)
        │   ├── instances.json                 # tracked-object catalogue
        │   └── objects.json                   # per-frame detections
        ├── task/
        │   ├── mcq.json                       # multi-choice questions
        │   ├── bcq.json                       # binary (Yes/No) questions
        │   └── open_qa.json                   # open-ended questions
        └── sidecars/                          # non-DAFT diagnostic files

    ``<media_id>`` is the stem of the original input filename.

    Image scenes use ``image.json`` (no fps/duration) instead of
    ``video.json`` and do not write ``events.json``.
    """

    scene_dir: Path

    raw_dir: Path

    contextual_dir: Path
    contextual_video: Path
    contextual_image: Path
    contextual_events: Path
    contextual_instances: Path
    contextual_objects: Path

    task_dir: Path
    task_mcq: Path
    task_bcq: Path
    task_open_qa: Path

    sidecars_dir: Path


def scene_paths(scene_dir: Path | str) -> ScenePaths:
    """Return canonical DAFT path handles rooted at ``scene_dir``.

    Pure path construction — no filesystem side effects. Call
    ``ensure_scene_skeleton`` to materialize the empty directory tree.
    """
    root = Path(scene_dir)
    return ScenePaths(
        scene_dir=root,
        raw_dir=root / "raw",
        contextual_dir=root / "contextual",
        contextual_video=root / "contextual" / "video.json",
        contextual_image=root / "contextual" / "image.json",
        contextual_events=root / "contextual" / "events.json",
        contextual_instances=root / "contextual" / "instances.json",
        contextual_objects=root / "contextual" / "objects.json",
        task_dir=root / "task",
        task_mcq=root / "task" / "mcq.json",
        task_bcq=root / "task" / "bcq.json",
        task_open_qa=root / "task" / "open_qa.json",
        sidecars_dir=root / "sidecars",
    )


def resolve_raw_media(scene_dir: Path | str) -> Optional[Path]:
    """Return the media file under ``<scene>/raw/``, or ``None`` if absent.

    The raw directory contains the analyzed media, either as a symlink for
    normal local inputs or as a copied file for remote-staged inputs. This
    helper returns the first file found, regardless of name — callers should
    not assume a specific stem.
    """
    raw = Path(scene_dir) / "raw"
    if not raw.is_dir():
        return None
    for candidate in sorted(raw.iterdir()):
        if candidate.is_file() or candidate.is_symlink():
            return candidate
    return None


def ensure_scene_skeleton(scene_dir: Path | str) -> ScenePaths:
    """Create ``raw/``, ``contextual/``, ``task/``, ``sidecars/`` under ``scene_dir``.

    Idempotent. Called once per sample at the start of ``pipeline.py``. Returns
    the same ``ScenePaths`` as ``scene_paths(scene_dir)`` for convenience.
    """
    paths = scene_paths(scene_dir)
    for d in (paths.raw_dir, paths.contextual_dir, paths.task_dir, paths.sidecars_dir):
        d.mkdir(parents=True, exist_ok=True)
    return paths


__all__ = ["ScenePaths", "ensure_scene_skeleton", "resolve_raw_media", "scene_paths"]
