# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared runtime helpers for production detection/tracking backends."""

from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
from typing import Any

from core import SceneContext, ScenePaths, is_image_path, write_json
from core import stage_raw_media as _stage_raw_media
from core.formats.daft import daft_envelope


class OptionalDependencyError(RuntimeError):
    """Raised when a selected backend is missing optional runtime dependencies."""


def import_optional(module_name: str, *, backend: str) -> Any:
    """Import an optional backend dependency with a backend-specific error."""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        extra = "sam3" if backend == "sam3" else "rfdetr"
        raise OptionalDependencyError(
            f"Backend {backend!r} requires optional dependency {module_name!r}. "
            f"Install detection-and-tracking[{extra}] or use the matching service container."
        ) from exc


def resolve_model_cache_root(model_cache_path: str | None) -> Path:
    """Resolve the shared model cache root."""
    env_cache = _clean_optional_path(os.getenv("MODEL_CACHE_PATH"))
    cfg_cache = _clean_optional_path(model_cache_path)
    if env_cache is not None:
        return env_cache
    if cfg_cache is not None:
        return cfg_cache
    container_default = Path("/models")
    if container_default.exists():
        return container_default.resolve()
    return (Path.cwd() / "ckpts").resolve()


def media_output_suffix(media_path: Path) -> str:
    """Return the sidecar media suffix for an input path."""
    return "png" if is_image_path(media_path) else "mp4"


def stage_raw_media(
    *,
    scene_paths: ScenePaths,
    media_path: Path,
    scene_ctx: SceneContext,
    copy_media: bool,
    logger: logging.Logger,
) -> Path | None:
    """Materialize ``raw/<media_id>.<ext>`` for the analyzed media.

    Thin wrapper over :func:`core.stage_raw_media` that adapts the
    ``ScenePaths``/``SceneContext`` call convention used by detection backends.
    Local product runs normally use a relative symlink to avoid duplicating
    large videos; set ``copy_media`` when the output scene must be self-contained.
    """
    return _stage_raw_media(
        raw_dir=scene_paths.raw_dir,
        media_path=media_path,
        media_id=scene_ctx.media_id,
        copy_media=copy_media,
        logger=logger,
    )


def objects_json_path(scene_paths: ScenePaths) -> Path:
    """Return the canonical contextual objects artifact path."""
    return scene_paths.contextual_dir / "objects.json"


def instances_json_path(scene_paths: ScenePaths) -> Path:
    """Return the canonical contextual instances artifact path."""
    return scene_paths.contextual_dir / "instances.json"


def write_tracking_artifacts(
    *,
    scene_paths: ScenePaths,
    scene_ctx: SceneContext,
    frames: list[dict[str, Any]],
    instances: list[dict[str, Any]],
) -> None:
    """Write the canonical contextual objects/instances artifacts."""
    objects_payload = daft_envelope("objects", scene_ctx, include_scene_id=not scene_ctx.is_image)
    objects_payload["frames"] = frames
    write_json(objects_json_path(scene_paths), objects_payload)

    instances_payload = daft_envelope("instances", scene_ctx, include_scene_id=False)
    instances_payload["instances"] = instances
    write_json(instances_json_path(scene_paths), instances_payload)


def first_configured_gpu(gpu_ids: str | int | None) -> int | None:
    """Return the first configured GPU id, or ``None`` when CUDA should not be forced."""
    if gpu_ids is None:
        return None
    if isinstance(gpu_ids, int):
        return gpu_ids if gpu_ids >= 0 else None
    raw = str(gpu_ids).strip().lower()
    if raw in {"", "all"}:
        return None
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        try:
            gpu_id = int(token)
        except ValueError:
            continue
        return gpu_id if gpu_id >= 0 else None
    return None


def _clean_optional_path(value: str | None) -> Path | None:
    raw = str(value or "").strip()
    if not raw or raw.lower() in {"none", "null"}:
        return None
    return Path(raw).expanduser().resolve()


def as_int(value: Any, default: int) -> int:
    """Best-effort conversion of optional numeric runtime values to ``int``."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_float(value: Any, default: float) -> float:
    """Best-effort conversion of optional numeric runtime values to ``float``."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def list_from_runtime(value: Any) -> list[Any]:
    """Convert array-like runtime values into a Python list."""
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
        return [converted]
    try:
        return list(value)
    except TypeError:
        return [value]


__all__ = [
    "OptionalDependencyError",
    "as_float",
    "as_int",
    "first_configured_gpu",
    "import_optional",
    "instances_json_path",
    "list_from_runtime",
    "media_output_suffix",
    "objects_json_path",
    "resolve_model_cache_root",
    "stage_raw_media",
    "write_tracking_artifacts",
]
