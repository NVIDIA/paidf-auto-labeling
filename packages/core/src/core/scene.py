# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Per-scene context and canonical on-disk DAFT v3 layout.

A scene is the unit of pseudo-annotation work. Every task in a pipeline
operates on a single scene at a time and reads/writes through the same
canonical directory layout described by ``ScenePaths``.

Two pieces are exposed:

- ``SceneContext``: per-scene constants (``media_id``, image-vs-video flag,
  ISO date) threaded by keyword into every task call. There is no
  module-level mutable state so converters and tasks remain test-friendly
  and parallel-safe.
- ``ScenePaths`` + ``ensure_scene_skeleton``: the on-disk contract every task
  agrees on. Tasks derive their target paths from these helpers rather than
  threading per-file paths through configuration.
- Active/raw media helpers: canonical sidecar names used by pipelines to stage
  the original input and hand off the current media between tasks.

Image-extension detection is local (no extra dependencies); extend
``IMAGE_EXTENSIONS`` if a new format needs to be recognized.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from datetime import date as _date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from core.models import DataEntry, read_pipeline_state

ACTIVE_MEDIA_GLOB: str = "active.*"
"""Glob used to discover existing active image or video sidecars."""

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})
"""Known image suffixes."""

VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".avi", ".mkv", ".mov", ".webm"})
"""Known video suffixes."""

RAW_DIRNAME: str = "raw"
"""DAFT analyzed-media directory name."""

CONTEXTUAL_DIRNAME: str = "contextual"
"""DAFT validated scene metadata directory name."""

TASK_DIRNAME: str = "task"
"""DAFT validated task items directory name."""

SIDECARS_DIRNAME: str = "sidecars"
"""DAFT sidecars directory name."""

RAW_MEDIA_BASENAME: str = "raw"
"""Canonical sidecar basename for the original input media copy. (e.g. raw.mp4)"""


def is_image_path(path: Path | str) -> bool:
    """
    Check whether ``path`` points at a supported image format.

    Args:
        path: Filesystem path to inspect.
    Returns:
        True if the path's extension is in ``IMAGE_EXTENSIONS`` (case-insensitive).
    """
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def _is_media_path(path: Path | str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS


def _sidecar_media_filename_for_input(
    input_path: Path | str,
    *,
    basename: str,
) -> str:
    source = Path(input_path)
    if not source.suffix:
        msg = f"Input media path {input_path!s} must include a file suffix."
        raise ValueError(msg)
    return f"{basename}{source.suffix}"


def active_media_path_for_input(scene_dir: Path | str, input_path: Path | str) -> Path:
    """
    Return the canonical active-media path for an input media file.

    The cross-container handoff contract uses ``sidecars/active<source suffix>``
    so image and video formats are preserved.

    Args:
        scene_dir: Root DAFT scene directory.
        input_path: Original input media path used to infer image vs video.
    Returns:
        The active media path under ``scene_dir/sidecars``.
    """
    filename = _sidecar_media_filename_for_input(
        input_path,
        basename="active",
    )
    return Path(scene_dir) / SIDECARS_DIRNAME / filename


def raw_media_path(scene_dir: Path | str, input_path: Path | str) -> Path:
    """
    Return the canonical raw-media bookkeeping path for a DAFT scene.

    Args:
        scene_dir: Root DAFT scene directory.
        input_path: Original input media path used to infer image vs video.
    Returns:
        The raw media path under ``scene_dir/sidecars``.
    """
    filename = _sidecar_media_filename_for_input(
        input_path,
        basename=RAW_MEDIA_BASENAME,
    )
    return Path(scene_dir) / SIDECARS_DIRNAME / filename


def find_active_media_path(scene_dir: Path | str) -> Path | None:
    """
    Find the first existing active media file in a DAFT scene's sidecars directory.

    Args:
        scene_dir: Root DAFT scene directory.
    Returns:
        The active media path, or None when no active media file exists.
    """
    root = Path(scene_dir) / SIDECARS_DIRNAME
    for candidate in sorted(root.glob(ACTIVE_MEDIA_GLOB)):
        if candidate.is_file() and _is_media_path(candidate):
            return candidate
    return None


def scene_media_id_from_path(scene_dir: Path | str) -> str:
    """
    Derive the default DAFT media identifier from a scene directory path.

    Remote data paths may be staged into framework-owned temp directories before
    tasks run, so callers that still have the original scene path should persist
    this value before rewriting ``DataEntry.data_path``.
    """
    return Path(str(scene_dir).rstrip("/")).name


class SceneContext(BaseModel):
    """
    Per-scene constants threaded through every task call.

    Built once per ``DataEntry`` (typically by the pipeline before invoking the
    first task) and passed to every task via the ``scene_ctx`` keyword. Tasks
    read ``media_id`` (the DAFT scene anchor — ``video_id`` for video scenes,
    ``image_id`` for image scenes), ``is_image`` to pick the right schema
    flavor, and ``iso_date`` so every artifact in the scene shares one
    deterministic date stamp regardless of pipeline runtime.

    Attributes:
        media_id: DAFT scene anchor identifier (typically the input file stem).
        is_image: True for image inputs, False for video inputs.
        iso_date: ISO-formatted date stamped into every DAFT envelope this
            scene produces. Defaults to today.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    media_id: str
    is_image: bool = False
    iso_date: str = Field(default_factory=lambda: _date.today().isoformat())
    instances_source: str | None = None
    license_str: str | None = None
    tags: tuple[str, ...] = ()

    @classmethod
    def from_input(cls, input_path: Path | str, **overrides: Any) -> SceneContext:
        """
        Build a ``SceneContext`` from an input file path.

        ``media_id`` defaults to the file stem; ``is_image`` is auto-detected
        from the extension. Any constructor field can be overridden via keyword
        arguments.

        Args:
            input_path: Path to the input media file.
            **overrides: Keyword overrides for any ``SceneContext`` field.
        Returns:
            A new ``SceneContext`` derived from ``input_path`` with overrides
            applied.
        """
        p = Path(input_path)
        defaults: dict[str, Any] = {"media_id": p.stem, "is_image": is_image_path(p)}
        defaults.update(overrides)
        return cls(**defaults)

    @property
    def scene_id_field(self) -> str:
        """DAFT scene-anchor field name (``"image_id"`` or ``"video_id"``)."""
        return "image_id" if self.is_image else "video_id"


def scene_context_for_entry(data_entry: DataEntry) -> SceneContext:
    """
    Build scene context from a pipeline data entry.

    During pipeline execution, ``data_entry.media_path`` may point at a
    framework-managed media sidecar, including transient task outputs such as
    enhanced media. In that case, the media anchor should be stable scene
    identity rather than the generic sidecar basename. Direct task calls
    against ordinary media keep the input file stem as the media ID.

    Args:
        data_entry: Pipeline entry containing media and scene directory paths.
    Returns:
        Scene context for the entry's current media and DAFT scene.
    """
    media_path = Path(data_entry.media_path)
    scene_media_id = scene_media_id_from_path(data_entry.data_path)
    if _is_scene_managed_media(data_entry.data_path, media_path):
        return SceneContext.from_input(
            media_path,
            media_id=_stable_scene_media_id(data_entry.data_path)
            or scene_media_id
            or media_path.stem,
        )
    return SceneContext.from_input(media_path)


def _stable_scene_media_id(scene_dir: Path | str) -> str | None:
    return read_pipeline_state(scene_dir).media_id


def _is_scene_managed_media(scene_dir: Path | str, media_path: Path) -> bool:
    sidecars_dir = Path(scene_dir) / SIDECARS_DIRNAME
    if _is_media_path(media_path) and _is_under_directory(media_path, sidecars_dir):
        return True

    try:
        active_path = active_media_path_for_input(scene_dir, media_path)
        raw_path = raw_media_path(scene_dir, media_path)
    except ValueError:
        return False
    return _same_path(media_path, active_path) or _same_path(media_path, raw_path)


def _is_under_directory(path: Path, directory: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(directory.resolve(strict=False))
    except ValueError:
        return False
    return True


def _same_path(left: Path, right: Path) -> bool:
    return left.resolve(strict=False) == right.resolve(strict=False)


def resolve_sidecar(sidecars_root: Path, relative_name: str) -> Path:
    """Resolve an operator-supplied sidecar name under ``sidecars_root`` safely.

    Joins ``relative_name`` onto ``sidecars_root`` and verifies the resolved
    path stays inside that root, rejecting absolute paths and ``..`` traversal.
    Use this for any sidecar/crop path that can come from config or CLI input so
    a malicious or mistaken value cannot read or write outside the scene.

    Args:
        sidecars_root: The scene ``sidecars/`` directory to contain the path.
        relative_name: A relative sidecar path (e.g. ``"visual_qa/items.json"``).

    Returns:
        The resolved, contained absolute path.

    Raises:
        ValueError: If the resolved path escapes ``sidecars_root``.
    """
    root = sidecars_root.resolve(strict=False)
    candidate = (root / Path(relative_name)).resolve(strict=False)
    if not candidate.is_relative_to(root):
        raise ValueError(f"sidecar path escapes scene sidecars root: {relative_name!r}")
    return candidate


@dataclass(frozen=True)
class ScenePaths:
    """
    Canonical directory handles for a single DAFT v3 scene.

    A scene directory has a fixed top-level layout::

        <scene>/
        ├── raw/                          # analyzed media (link or copy)
        ├── contextual/                   # DAFT-validated scene metadata
        ├── task/                         # DAFT-validated task items
        └── sidecars/                     # non-DAFT diagnostic files

    Tasks and emitters compose their own filenames against these directory
    handles (e.g. ``scene_paths.task_dir / "mcq.json"``). Adding a new
    annotation type never requires editing core.
    """

    scene_dir: Path
    raw_dir: Path
    contextual_dir: Path
    task_dir: Path
    sidecars_dir: Path


def ensure_scene_skeleton(scene_dir: Path | str) -> ScenePaths:
    """
    Create the ``raw/``, ``contextual/``, ``task/``, ``sidecars/`` subdirectories
    under ``scene_dir`` if missing. Idempotent.

    Args:
        scene_dir: Root directory for the scene.
    Returns:
        A ``ScenePaths`` exposing handles to the four subdirectories.
    """
    root = Path(scene_dir)
    paths = ScenePaths(
        scene_dir=root,
        raw_dir=root / RAW_DIRNAME,
        contextual_dir=root / CONTEXTUAL_DIRNAME,
        task_dir=root / TASK_DIRNAME,
        sidecars_dir=root / SIDECARS_DIRNAME,
    )
    for d in (paths.raw_dir, paths.contextual_dir, paths.task_dir, paths.sidecars_dir):
        d.mkdir(parents=True, exist_ok=True)
    return paths


def raw_analyzed_media_name(media_id: str, suffix: str) -> str:
    """Return the ``raw/<name>`` filename for a scene's analyzed media.

    Scene media identifiers are the scene directory name, which for the
    file-per-scene layout already carries the input extension (e.g.
    ``"clip.mp4"``). Appending the source suffix unconditionally would produce a
    doubled extension (``"clip.mp4.mp4"``); when ``media_id`` already ends with
    ``suffix`` it is used verbatim so the analyzed media follows the documented
    ``raw/<media_id>.<ext>`` contract with a single extension.

    Args:
        media_id: DAFT scene media identifier (may or may not include the suffix).
        suffix: Source media suffix including the leading dot (e.g. ``".mp4"``).
    Returns:
        The bare filename to use under ``raw/``.
    """
    normalized = suffix.lower()
    if normalized and media_id.lower().endswith(normalized):
        return media_id
    return f"{media_id}{normalized}"


def stage_raw_media(
    *,
    raw_dir: Path,
    media_path: Path,
    media_id: str,
    copy_media: bool,
    logger: logging.Logger,
) -> Path | None:
    """Materialize ``raw/<media_id>.<ext>`` for a scene's analyzed media.

    Local product runs normally use a relative symlink to avoid duplicating
    large videos. Set ``copy_media`` when the source is temporary or the output
    scene must be self-contained. Idempotent: an existing entry is replaced.

    Args:
        raw_dir: The scene's ``raw/`` directory (created if missing).
        media_path: The analyzed media file to stage (link target or copy source).
        media_id: DAFT scene media identifier used to name the staged file.
        copy_media: Copy the bytes when True, otherwise create a relative symlink.
        logger: Logger for the missing-input warning path.
    Returns:
        The staged ``raw/`` path, or ``None`` when the source media is missing.
    """
    source = media_path.expanduser()
    if not source.exists() or not source.is_file():
        logger.warning("Raw media was not staged because input is missing: %s", media_path)
        return None

    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / raw_analyzed_media_name(media_id, source.suffix)
    try:
        if raw_path.samefile(source):
            return raw_path
    except FileNotFoundError:
        if raw_path.resolve() == source.resolve():
            return raw_path
    if raw_path.exists() or raw_path.is_symlink():
        raw_path.unlink()

    if copy_media:
        shutil.copy2(source, raw_path)
        return raw_path

    target = os.path.relpath(source.resolve(), raw_dir.resolve())
    raw_path.symlink_to(target)
    return raw_path


__all__ = [
    "ACTIVE_MEDIA_GLOB",
    "CONTEXTUAL_DIRNAME",
    "IMAGE_EXTENSIONS",
    "RAW_DIRNAME",
    "RAW_MEDIA_BASENAME",
    "SIDECARS_DIRNAME",
    "TASK_DIRNAME",
    "SceneContext",
    "ScenePaths",
    "VIDEO_EXTENSIONS",
    "active_media_path_for_input",
    "ensure_scene_skeleton",
    "find_active_media_path",
    "is_image_path",
    "raw_analyzed_media_name",
    "raw_media_path",
    "resolve_sidecar",
    "scene_context_for_entry",
    "scene_media_id_from_path",
    "stage_raw_media",
]
