# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load task-owned sidecar artifacts into DAFT export stage inputs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core import read_pipeline_state

from reasoning.paths import ScenePaths
from reasoning.stage_inputs import StageInputs, safe_read_json

CAPTION_ARTIFACTS_KEY = "captioning"


def load_stage_inputs(paths: ScenePaths, *, logger: logging.Logger) -> StageInputs:
    """Load DAFT stage inputs from canonical files and task artifact sidecars."""
    inputs = StageInputs.load(paths, logger=logger)
    artifact_paths = ArtifactPathResolver(paths.scene_dir)
    caption_artifacts = task_artifacts(
        paths,
        key=CAPTION_ARTIFACTS_KEY,
        logger=logger,
    )
    sidecar_metadata = inputs.sidecar_metadata or first_json_dict(
        [
            paths.sidecars_dir / "captioning" / "metadata.json",
        ],
        logger=logger,
        tag="caption-metadata",
    )
    sidecar_metadata_chunk = inputs.sidecar_metadata_chunk or first_json_dict(
        [
            paths.sidecars_dir / "captioning" / "metadata_chunk.json",
            artifact_paths.existing_path(artifact_str(caption_artifacts, "metadata_chunk_json")),
            paths.sidecars_dir / "captioning" / "video_captions.json",
            artifact_paths.existing_path(artifact_str(caption_artifacts, "video_json")),
        ],
        logger=logger,
        tag="caption-metadata-chunk",
    )
    image_caption = first_json_dict(
        [
            paths.sidecars_dir / "captioning" / "image_caption.json",
            paths.sidecars_dir / "captioning" / "image_captions.json",
            artifact_paths.existing_path(artifact_str(caption_artifacts, "image_json")),
            artifact_paths.existing_path(artifact_str(caption_artifacts, "metadata_chunk_json")),
        ],
        logger=logger,
        tag="caption-image",
    )
    scene_video = inputs.scene_video or normalize_video_sidecar(
        sidecar_metadata_chunk or sidecar_metadata
    )
    scene_image = inputs.scene_image or normalize_image_sidecar(image_caption or sidecar_metadata)
    return StageInputs(
        sidecar_metadata=sidecar_metadata,
        sidecar_metadata_chunk=sidecar_metadata_chunk,
        scene_video=scene_video,
        scene_image=scene_image,
        scene_events=inputs.scene_events,
        scene_instances=inputs.scene_instances,
        scene_msted=inputs.scene_msted,
        scene_anomaly=inputs.scene_anomaly,
        scene_pas=inputs.scene_pas,
    )


def task_artifacts(
    paths: ScenePaths,
    *,
    key: str,
    logger: logging.Logger,
) -> dict[str, object] | None:
    """Read one task artifact block from scene pipeline state."""
    try:
        artifacts = read_pipeline_state(paths.scene_dir).task_artifacts.get(key)
    except (OSError, ValueError) as exc:
        logger.warning("[%s-artifacts] failed to read pipeline state: %s", key, exc)
        return None
    return artifacts


def artifact_str(artifacts: dict[str, object] | None, field: str) -> str | None:
    """Return artifact field value when it is a string."""
    if artifacts is None:
        return None
    value = artifacts.get(field)
    return value if isinstance(value, str) else None


def first_json_dict(
    candidates: list[Path | None],
    *,
    logger: logging.Logger,
    tag: str,
) -> dict[str, Any] | None:
    """Return the first candidate path that reads as a JSON object."""
    for candidate in candidates:
        if candidate is None:
            continue
        payload = safe_read_json(candidate, logger=logger, tag=tag)
        if isinstance(payload, dict):
            return payload
    return None


@dataclass(frozen=True)
class ArtifactPathResolver:
    """Resolve persisted task artifact paths for one scene directory."""

    scene_dir: Path

    def existing_path(self, value: str | None) -> Path | None:
        """Return an existing artifact path, resolving relatives against the scene first."""
        if not value:
            return None
        raw = Path(value).expanduser()
        candidates = [raw] if raw.is_absolute() else [self.scene_dir / raw, raw]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None


def existing_path(value: str | None) -> Path | None:
    """Return ``value`` as a path only when it exists."""
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.exists() else None


def normalize_video_sidecar(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Adapt captioning video sidecars to the DAFT stage-input video shape."""
    if payload is None:
        return None
    normalized = dict(payload)
    if clean_str(normalized.get("scene_description")) is None:
        scene_description = first_window_parsed_string(normalized, "scene_description")
        scene_description = scene_description or first_window_string(
            normalized,
            ("scene_description", "description", "caption"),
        )
        if scene_description is not None:
            normalized["scene_description"] = scene_description
    if clean_str(normalized.get("event_summary")) is None:
        event_summary = clean_str(normalized.get("event_summary"))
        event_summary = event_summary or clean_str(normalized.get("summary"))
        event_summary = event_summary or first_window_parsed_string(normalized, "event_summary")
        if event_summary is not None:
            normalized["event_summary"] = event_summary
    if not isinstance(normalized.get("duration"), (int, float)):
        duration = duration_from_caption_payload(normalized)
        if duration is not None:
            normalized["duration"] = duration
    return normalized


def normalize_image_sidecar(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Adapt captioning image sidecars to the DAFT stage-input image shape."""
    if payload is None:
        return None
    caption = clean_str(payload.get("caption"))
    if caption is None:
        parsed = payload.get("parsed")
        if isinstance(parsed, dict):
            caption = clean_str(parsed.get("caption"))
    if caption is None:
        return None
    normalized = dict(payload)
    normalized["caption"] = caption
    return normalized


def first_window_parsed_string(payload: dict[str, Any], key: str) -> str | None:
    """Return the first non-empty ``windows[].parsed[key]`` string."""
    windows = payload.get("windows")
    if not isinstance(windows, list):
        return None
    for window in windows:
        if not isinstance(window, dict):
            continue
        parsed = window.get("parsed")
        if not isinstance(parsed, dict):
            continue
        value = clean_str(parsed.get(key))
        if value is not None:
            return value
    return None


def first_window_string(payload: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    """Return the first non-empty string in any window for ``keys``."""
    windows = payload.get("windows")
    if not isinstance(windows, list):
        return None
    for window in windows:
        if not isinstance(window, dict):
            continue
        for key in keys:
            value = clean_str(window.get(key))
            if value is not None:
                return value
    return None


def duration_from_caption_payload(payload: dict[str, Any]) -> float | None:
    """Infer duration from captioning sidecar metadata."""
    media = payload.get("media")
    if isinstance(media, dict):
        duration = media.get("duration_s")
        if isinstance(duration, (int, float)) and duration > 0:
            return float(duration)
    duration_span = payload.get("duration_span")
    if isinstance(duration_span, list) and len(duration_span) == 2:
        start, end = duration_span
        if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end >= start:
            return float(end - start)
    return None


def clean_str(value: Any) -> str | None:
    """Return a stripped non-empty string."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


__all__ = [
    "CAPTION_ARTIFACTS_KEY",
    "ArtifactPathResolver",
    "artifact_str",
    "clean_str",
    "duration_from_caption_payload",
    "existing_path",
    "first_json_dict",
    "first_window_parsed_string",
    "first_window_string",
    "load_stage_inputs",
    "normalize_image_sidecar",
    "normalize_video_sidecar",
    "task_artifacts",
]
