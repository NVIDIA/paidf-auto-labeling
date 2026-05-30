# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the DAFT converter.

Centralizes the schema version (``DAFT_VERSION``) and the on-disk write surface
(``write_daft_json``); everything else in this module is a small utility other
converter files compose.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Iterable

from al_utils.io import write_json

# nvidia-tao-daft metropolis-v3.0 schemas pin every contextual and task file's
# version field to this exact format identity.
DAFT_VERSION: str = "metropolis-v3.0"

# Input filename stem — the DAFT media identifier for the scene.
# Set once per sample by pipeline.py; read via get_scene_media_id().
_scene_media_id: str = "main"


def set_scene_media_id(input_path: "Path | str") -> str:
    """Derive and store the scene media identifier from the input filename.

    Returns the computed id (``Path(input_path).stem``) for convenience.
    """
    global _scene_media_id
    _scene_media_id = Path(input_path).stem
    return _scene_media_id


def get_scene_media_id() -> str:
    """Return the current scene media identifier."""
    return _scene_media_id


class DaftConvertError(ValueError):
    """Raised when auto-labeling-internal data cannot be converted to a DAFT-compliant form.

    Used by the per-file converters when an upstream invariant is violated
    (e.g. an MCQ item whose answer is not in its options list, an MCQ with
    more options than DAFT's single-letter answer space allows, etc.).
    """


def metadata_block(
    type_str: str,
    *,
    iso_date: str | None = None,
    description: str | None = None,
    license_str: str | None = None,
    tags: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build a DAFT ``metadata`` block.

    Only ``type`` is required by the spec. ``date`` defaults to today; other
    fields are emitted only when provided. Key ordering is stable
    (type, date, description, license, tags) for readable on-disk diffs.
    """
    block: dict[str, Any] = {"type": type_str, "date": iso_date or date.today().isoformat()}
    if description is not None:
        block["description"] = description
    if license_str is not None:
        block["license"] = license_str
    if tags is not None:
        block["tags"] = list(tags)
    return block


def write_daft_json(path: Path | str, payload: dict[str, Any]) -> None:
    """Write a DAFT-compliant payload to ``path``.

    This is the single on-disk write surface for the converter. It performs a
    minimal shape check (``version`` + ``metadata`` keys present) so converter
    bugs surface here rather than as cryptic ``tao-daft validate`` errors.
    Full schema validation is delegated to the optional ``tao-daft`` CLI hook
    (see ``modules/cli.py``).

    Creates the parent directory if it doesn't exist. Uses the auto-labeling standard JSON
    formatting (UTF-8, 2-space indent, trailing newline) via ``al_utils.io``.
    """
    if not isinstance(payload, dict):
        raise TypeError(f"write_daft_json expects a dict, got {type(payload).__name__}")
    if "version" not in payload:
        raise ValueError(f"DAFT payload missing 'version' key (path={path})")
    if "metadata" not in payload:
        raise ValueError(f"DAFT payload missing 'metadata' key (path={path})")
    write_json(Path(path), payload)


__all__ = [
    "DAFT_VERSION",
    "DaftConvertError",
    "get_scene_media_id",
    "metadata_block",
    "set_scene_media_id",
    "write_daft_json",
]
