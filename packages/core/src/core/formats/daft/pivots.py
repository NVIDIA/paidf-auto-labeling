# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service-owned DAFT pivot helpers for captioning and visual QA outputs."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from core.formats.daft.converters import (
    to_daft_chunks,
    to_daft_scene_description,
    to_daft_tasks,
    to_daft_temporal_description,
    to_daft_video_summarization,
)
from core.formats.daft.errors import DaftConvertError
from core.formats.daft.writer import DaftSceneWriter
from core.scene import SceneContext, ensure_scene_skeleton

CAPTION_ARTIFACTS_KEY = "captioning"
VISUAL_QA_ARTIFACTS_KEY = "visual_qa"

# Auxiliary (non-DAFT-type) per-person QA artifact written under
# ``sidecars/visual_qa/``. Unlike the flat mcq/bcq/open_qa files (which collapse
# every window into one video-keyed bank), this one preserves per-tracked-person
# Q&A. It is a task-internal artifact, not a registered DAFT type, so it lives in
# sidecars/ rather than task/.
PERSONA_QA_FILENAME = "persona_qa.json"

_DESCRIPTION_KEYS: tuple[str, ...] = (
    "enhanced_caption",
    "description",
    "caption",
    "summary",
)


def emit_captioning_daft_outputs(
    scene_dir: Path | str,
    ctx: SceneContext,
    *,
    caption_artifacts: Mapping[str, object] | None = None,
    logger: logging.Logger | None = None,
) -> tuple[Path, ...]:
    """Write DAFT captioning pivots from captioning sidecars for one scene."""
    log = logger or logging.getLogger(__name__)
    paths = ensure_scene_skeleton(scene_dir)
    writer = DaftSceneWriter(paths.scene_dir)

    metadata = _first_json_dict(
        [
            paths.sidecars_dir / "captioning" / "metadata.json",
            paths.sidecars_dir / "metadata.json",
        ],
        logger=log,
        tag="captioning-pivot",
    )
    dense_metadata = _first_json_dict(
        [
            _artifact_path(paths.scene_dir, caption_artifacts, "metadata_chunk_json"),
            paths.sidecars_dir / "captioning" / "metadata_chunk.json",
            paths.sidecars_dir / "metadata_chunk.json",
            _artifact_path(paths.scene_dir, caption_artifacts, "video_json"),
            paths.sidecars_dir / "captioning" / "video_captions.json",
        ],
        logger=log,
        tag="captioning-pivot",
    )
    image_metadata = _first_json_dict(
        [
            _artifact_path(paths.scene_dir, caption_artifacts, "image_json"),
            paths.sidecars_dir / "captioning" / "image_caption.json",
            paths.sidecars_dir / "captioning" / "image_captions.json",
            _artifact_path(paths.scene_dir, caption_artifacts, "metadata_chunk_json"),
        ],
        logger=log,
        tag="captioning-pivot",
    )
    _validate_caption_media_id(metadata, ctx=ctx, source="captioning metadata sidecar")
    _validate_caption_media_id(
        dense_metadata,
        ctx=ctx,
        source="captioning dense metadata sidecar",
    )
    _validate_caption_media_id(image_metadata, ctx=ctx, source="captioning image sidecar")

    written: list[Path] = []
    if not ctx.is_image:
        windows = _windows(dense_metadata) or _windows(metadata)
        written.extend(_emit_caption_windows(writer, ctx, windows=windows, logger=log))

    contextual = (
        _normalize_image_sidecar(image_metadata or metadata)
        if ctx.is_image
        else _normalize_video_sidecar(dense_metadata or metadata)
    )
    written.extend(_emit_caption_prose(writer, ctx, contextual=contextual, logger=log))
    return tuple(written)


def emit_visual_qa_daft_outputs(
    scene_dir: Path | str,
    ctx: SceneContext,
    *,
    visual_qa_artifacts: Mapping[str, object] | None = None,
    logger: logging.Logger | None = None,
) -> tuple[Path, ...]:
    """Write DAFT QA task files from a normalized visual-QA items sidecar."""
    log = logger or logging.getLogger(__name__)
    paths = ensure_scene_skeleton(scene_dir)
    payload = _first_json_dict(
        [
            _artifact_path(paths.scene_dir, visual_qa_artifacts, "items_json"),
            paths.sidecars_dir / "visual_qa" / "items.json",
        ],
        logger=log,
        tag="visual-qa-pivot",
    )
    if payload is None:
        log.debug("[visual_qa] no normalized items sidecar; skipping DAFT task pivot")
        return ()

    _validate_media_id(payload, expected=ctx.media_id, source="visual QA items sidecar")
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        log.warning("[visual_qa] items sidecar missing list field 'items'; skipping task pivot")
        return ()

    items = [item for item in raw_items if isinstance(item, dict)]
    if len(items) != len(raw_items):
        log.warning("[visual_qa] dropped non-object item(s) from visual QA sidecar")
    if not items:
        log.debug("[visual_qa] visual QA items sidecar is empty; skipping DAFT task pivot")
        return ()

    try:
        mcq_payload, bcq_payload, open_qa_payload = to_daft_tasks(items, ctx=ctx)
    except DaftConvertError as exc:
        log.warning("[visual_qa] task conversion failed: %s (skipping)", exc)
        return ()

    writer = DaftSceneWriter(paths.scene_dir)
    outputs = (
        ("mcq", mcq_payload),
        ("bcq", bcq_payload),
        ("open_qa", open_qa_payload),
    )
    written: list[Path] = []
    for name, task_payload in outputs:
        if task_payload is None:
            continue
        result = writer.write_payload(task_payload, expected_type=name)
        written.append(result.path)
        log.info("[visual_qa] wrote DAFT %s from normalized items to %s", name, result.path)
    return tuple(written)


def emit_persona_qa_daft_outputs(
    scene_dir: Path | str,
    ctx: SceneContext,
    *,
    visual_qa_artifacts: Mapping[str, object] | None = None,
    logger: logging.Logger | None = None,
) -> Path | None:
    """Write an auxiliary per-person QA artifact (``sidecars/visual_qa/persona_qa.json``).

    The flat DAFT task files (``mcq``/``bcq``/``open_qa``) aggregate every
    window into a single video-keyed bank, which collapses a multi-person clip
    to one pseudo-subject. The per-track visual-QA windows in
    ``windows.normalized.json`` instead retain one Q&A set per tracked person,
    so this emitter promotes them into a grouped, per-persona artifact the flat
    files cannot express. Each persona's items are routed through the same
    ``to_daft_tasks`` contract as the flat files, so they stay DAFT-shaped.

    Only windows with a non-null ``track_id`` produce personas. A whole-clip
    pass (single null-track window, e.g. the anomaly pass) therefore emits
    nothing and never clobbers a per-track pass's ``persona_qa.json``. Returns
    the written path, or ``None`` when there is nothing per-person to emit.
    """
    log = logger or logging.getLogger(__name__)
    paths = ensure_scene_skeleton(scene_dir)
    payload = _first_json_dict(
        [
            _artifact_path(paths.scene_dir, visual_qa_artifacts, "windows_json"),
            paths.sidecars_dir / "visual_qa" / "windows.normalized.json",
        ],
        logger=log,
        tag="persona-qa-pivot",
    )
    if payload is None:
        log.debug("[visual_qa] no normalized windows sidecar; skipping persona_qa")
        return None

    _validate_media_id(payload, expected=ctx.media_id, source="visual QA windows sidecar")

    personas: list[dict[str, Any]] = []
    for window in _windows(payload):
        track_id = window.get("track_id")
        if track_id is None:
            continue
        raw_items = window.get("items")
        if not isinstance(raw_items, list):
            continue
        items = [item for item in raw_items if isinstance(item, dict)]
        if not items:
            continue
        try:
            mcq_payload, bcq_payload, open_qa_payload = to_daft_tasks(items, ctx=ctx)
        except DaftConvertError as exc:
            log.warning("[visual_qa] persona_qa conversion failed for track %s: %s", track_id, exc)
            continue
        mcq_items = mcq_payload["items"] if mcq_payload else []
        bcq_items = bcq_payload["items"] if bcq_payload else []
        open_qa_items = open_qa_payload["items"] if open_qa_payload else []
        if not (mcq_items or bcq_items or open_qa_items):
            continue
        persona: dict[str, Any] = {"track_id": track_id}
        for key in ("window_index", "n_crops_in_chunk", "crop_dir"):
            if key in window:
                persona[key] = window[key]
        persona["mcq"] = mcq_items
        persona["bcq"] = bcq_items
        persona["open_qa"] = open_qa_items
        personas.append(persona)

    if not personas:
        log.debug("[visual_qa] no per-track windows; skipping persona_qa")
        return None

    personas.sort(key=lambda entry: str(entry["track_id"]))
    out_payload = {
        "schema_version": "1",
        "media_id": ctx.media_id,
        "type": "persona_qa",
        "n_personas": len(personas),
        "personas": personas,
    }
    out_path = paths.sidecars_dir / "visual_qa" / PERSONA_QA_FILENAME
    _write_json_atomic(out_path, out_payload)
    log.info("[visual_qa] wrote per-person QA (%d persona(s)) to %s", len(personas), out_path)
    return out_path


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def _emit_caption_windows(
    writer: DaftSceneWriter,
    ctx: SceneContext,
    *,
    windows: list[dict[str, Any]],
    logger: logging.Logger,
) -> list[Path]:
    if not windows:
        logger.debug("[captioning] no video windows; skipping chunks and temporal_description")
        return []
    written: list[Path] = []
    try:
        chunks_payload = to_daft_chunks(windows, ctx=ctx, description_keys=_DESCRIPTION_KEYS)
    except DaftConvertError as exc:
        logger.warning("[captioning] chunks conversion failed: %s", exc)
    else:
        if chunks_payload is not None:
            result = writer.write_contextual("chunks", chunks_payload)
            written.append(result.path)
            logger.info("[captioning] wrote DAFT chunks to %s", result.path)

    try:
        temporal_payload = to_daft_temporal_description(
            windows,
            ctx=ctx,
            description_keys=_DESCRIPTION_KEYS,
        )
    except DaftConvertError as exc:
        logger.warning("[captioning] temporal_description conversion failed: %s", exc)
    else:
        if temporal_payload is not None:
            result = writer.write_task("temporal_description", temporal_payload)
            written.append(result.path)
            logger.info("[captioning] wrote DAFT temporal_description to %s", result.path)
    return written


def _emit_caption_prose(
    writer: DaftSceneWriter,
    ctx: SceneContext,
    *,
    contextual: dict[str, Any] | None,
    logger: logging.Logger,
) -> list[Path]:
    if contextual is None:
        logger.debug("[captioning] no contextual prose sidecar; skipping prose task pivots")
        return []

    written: list[Path] = []
    if ctx.is_image:
        answer = _clean_str(contextual.get("caption"))
        question = "Describe the image."
    else:
        answer = _clean_str(contextual.get("scene_description")) or _clean_str(
            contextual.get("caption")
        )
        question = "Describe the scene."
    if answer is not None:
        try:
            payload = to_daft_scene_description(answer, ctx=ctx, question=question)
        except DaftConvertError as exc:
            logger.warning("[captioning] scene_description conversion failed: %s", exc)
        else:
            if payload is not None:
                result = writer.write_task("scene_description", payload)
                written.append(result.path)
                logger.info("[captioning] wrote DAFT scene_description to %s", result.path)

    if ctx.is_image:
        return written

    summary = _clean_str(contextual.get("event_summary"))
    if summary is None:
        return written
    timestamp: tuple[float, float] | None = None
    duration = contextual.get("duration")
    if isinstance(duration, (int, float)) and duration > 0:
        timestamp = (0.0, float(duration))
    try:
        payload = to_daft_video_summarization(summary, ctx=ctx, timestamp=timestamp)
    except DaftConvertError as exc:
        logger.warning("[captioning] video_summarization conversion failed: %s", exc)
    else:
        if payload is not None:
            result = writer.write_task("video_summarization", payload)
            written.append(result.path)
            logger.info("[captioning] wrote DAFT video_summarization to %s", result.path)
    return written


def _safe_read_json(path: Path, *, logger: logging.Logger, tag: str) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("[%s] failed to read %s: %s", tag, path, exc)
        return None


def _first_json_dict(
    candidates: list[Path | None],
    *,
    logger: logging.Logger,
    tag: str,
) -> dict[str, Any] | None:
    for candidate in candidates:
        if candidate is None:
            continue
        payload = _safe_read_json(candidate, logger=logger, tag=tag)
        if isinstance(payload, dict):
            return payload
    return None


def _artifact_path(
    scene_dir: Path,
    artifacts: Mapping[str, object] | None,
    field: str,
) -> Path | None:
    if artifacts is None:
        return None
    value = artifacts.get(field)
    if not isinstance(value, str) or not value:
        return None
    raw = Path(value).expanduser()
    candidates = [raw] if raw.is_absolute() else [scene_dir / raw, raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _windows(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if payload is None:
        return []
    windows = payload.get("windows")
    if not isinstance(windows, list):
        return []
    return [window for window in windows if isinstance(window, dict)]


def _normalize_video_sidecar(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    normalized = dict(payload)
    if _clean_str(normalized.get("scene_description")) is None:
        scene_description = _first_window_parsed_string(normalized, "scene_description")
        scene_description = scene_description or _first_window_string(
            normalized,
            ("scene_description", "description", "caption"),
        )
        if scene_description is not None:
            normalized["scene_description"] = scene_description
    if _clean_str(normalized.get("event_summary")) is None:
        event_summary = _clean_str(normalized.get("summary"))
        event_summary = event_summary or _first_window_parsed_string(normalized, "event_summary")
        if event_summary is not None:
            normalized["event_summary"] = event_summary
    if not isinstance(normalized.get("duration"), (int, float)):
        duration = _duration_from_caption_payload(normalized)
        if duration is not None:
            normalized["duration"] = duration
    return normalized


def _normalize_image_sidecar(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    caption = _clean_str(payload.get("caption"))
    if caption is None:
        parsed = payload.get("parsed")
        if isinstance(parsed, dict):
            caption = _clean_str(parsed.get("caption"))
    if caption is None:
        return None
    normalized = dict(payload)
    normalized["caption"] = caption
    return normalized


def _first_window_parsed_string(payload: dict[str, Any], key: str) -> str | None:
    windows = payload.get("windows")
    if not isinstance(windows, list):
        return None
    for window in windows:
        if not isinstance(window, dict):
            continue
        parsed = window.get("parsed")
        if not isinstance(parsed, dict):
            continue
        value = _clean_str(parsed.get(key))
        if value is not None:
            return value
    return None


def _first_window_string(payload: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    windows = payload.get("windows")
    if not isinstance(windows, list):
        return None
    for window in windows:
        if not isinstance(window, dict):
            continue
        for key in keys:
            value = _clean_str(window.get(key))
            if value is not None:
                return value
    return None


def _duration_from_caption_payload(payload: dict[str, Any]) -> float | None:
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


def _clean_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _validate_media_id(payload: dict[str, Any], *, expected: str, source: str) -> None:
    value = payload.get("media_id")
    if isinstance(value, str) and value.strip() and value.strip() != expected:
        raise ValueError(f"{source} media_id={value!r} does not match expected {expected!r}")


def _validate_caption_media_id(
    payload: dict[str, Any] | None,
    *,
    ctx: SceneContext,
    source: str,
) -> None:
    if payload is None:
        return
    for field in (ctx.scene_id_field, "media_id"):
        value = payload.get(field)
        if isinstance(value, str) and value.strip() and value.strip() != ctx.media_id:
            raise ValueError(f"{source} {field}={value!r} does not match expected {ctx.media_id!r}")


__all__ = [
    "CAPTION_ARTIFACTS_KEY",
    "PERSONA_QA_FILENAME",
    "VISUAL_QA_ARTIFACTS_KEY",
    "emit_captioning_daft_outputs",
    "emit_persona_qa_daft_outputs",
    "emit_visual_qa_daft_outputs",
]
