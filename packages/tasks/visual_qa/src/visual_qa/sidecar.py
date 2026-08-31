# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visual QA sidecar readers and writers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from core import resolve_sidecar


def read_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON object from ``path``."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def write_json_object(path: Path, payload: dict[str, Any]) -> Path:
    """Write ``payload`` as pretty JSON, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return path


def find_first_existing_sidecar(
    scene_sidecars: Path, relative_names: tuple[str, ...]
) -> Path | None:
    """Return the first existing sidecar candidate."""
    for name in relative_names:
        try:
            candidate = resolve_sidecar(scene_sidecars, name)
        except ValueError:
            continue
        if candidate.is_file():
            return candidate
    return None


def extract_window_item_groups(
    payload: dict[str, Any],
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    """Extract per-window raw item groups from supported sidecar shapes.

    Supported inputs:

    - ``{"items": [...]}`` — already normalized or pre-aggregated.
    - ``{"mcq": [...]}`` — source MCQ object.
    - ``{"windows": [...]}`` — each window may contain ``items`` / ``mcq`` /
      ``mcq_json`` directly, or a legacy JSON string in
      ``llm_enhanced_caption`` / ``enhanced_caption``.
    """
    if isinstance(payload.get("items"), list):
        return ([list(_dict_items(payload["items"]))], [])
    if isinstance(payload.get("mcq"), list):
        return ([list(_dict_items(payload["mcq"]))], [])

    windows = payload.get("windows")
    if not isinstance(windows, list):
        return ([], [])

    groups: list[list[dict[str, Any]]] = []
    normalized_windows: list[dict[str, Any]] = []
    for idx, raw_window in enumerate(windows):
        if not isinstance(raw_window, dict):
            continue
        items = _items_from_window(raw_window)
        normalized_window = _window_metadata(raw_window, index=idx)
        normalized_window["items"] = items
        normalized_windows.append(normalized_window)
        groups.append(items)
    return groups, normalized_windows


def _items_from_window(window: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("items", "mcq"):
        value = window.get(key)
        if isinstance(value, list):
            return list(_dict_items(value))

    for key in ("mcq_json", "parsed_mcq"):
        value = window.get(key)
        obj = _coerce_mcq_object(value)
        if obj is not None:
            return _items_from_object(obj)

    for key in ("llm_enhanced_caption", "enhanced_caption"):
        obj = _coerce_mcq_object(window.get(key))
        if obj is not None:
            return _items_from_object(obj)
    return []


def _items_from_object(obj: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("items", "mcq"):
        value = obj.get(key)
        if isinstance(value, list):
            return list(_dict_items(value))
    return []


def _coerce_mcq_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    for candidate in _json_candidates(value):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _json_candidates(text: str) -> list[str]:
    stripped = text.strip()
    candidates = [stripped]
    if "```" in stripped:
        parts = stripped.split("```")
        for part in parts:
            cleaned = part.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            if cleaned:
                candidates.append(cleaned)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if 0 <= start < end:
        candidates.append(stripped[start : end + 1])
    return candidates


def _dict_items(items: list[Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in items if isinstance(item, dict)]


def _window_metadata(window: dict[str, Any], *, index: int) -> dict[str, Any]:
    source_index = _numeric_value(window.get("window_index"))
    if source_index is None:
        source_index = _numeric_value(window.get("index"))
    out: dict[str, Any] = {"window_index": int(source_index) if source_index is not None else index}
    for key in (
        "start_s",
        "end_s",
        "start_frame",
        "end_frame",
        # Per-track (PAS video flow) bookkeeping, preserved so downstream
        # consumers can key normalized windows by identity.
        "track_id",
        "n_crops_in_chunk",
        "num_views",
        "first_frame_in_chunk",
        "last_frame_in_chunk",
    ):
        value = window.get(key)
        if not isinstance(value, bool) and isinstance(value, (int, float)):
            out[key] = value
    for key in ("description", "vlm_caption"):
        value = window.get(key)
        if isinstance(value, str) and value.strip():
            out[key] = value.strip()
    source_images = window.get("source_images")
    if isinstance(source_images, list) and all(isinstance(item, str) for item in source_images):
        out["source_images"] = list(source_images)
    return out


def _numeric_value(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


__all__ = [
    "extract_window_item_groups",
    "find_first_existing_sidecar",
    "read_json_object",
    "resolve_sidecar",
    "write_json_object",
]
