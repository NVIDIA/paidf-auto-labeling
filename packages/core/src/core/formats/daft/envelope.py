# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
DAFT v3 file envelope helpers.

Every contextual/task JSON file produced by a task or emitter shares the same
header shape (``version`` + scene-anchor id + ``metadata`` block). These
helpers centralize that shape so individual emitters do not reconstruct it
each time, and so a future schema bump can be made in one place.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date as _date
from typing import Any

from core.scene import SceneContext

DAFT_VERSION: str = "metropolis-v3.0"
"""DAFT v3 schema version literal pinned by every contextual/task schema."""


def metadata_block(
    type_str: str,
    *,
    iso_date: str | None = None,
    description: str | None = None,
    license_str: str | None = None,
    tags: Iterable[str] | None = None,
) -> dict[str, Any]:
    """
    Build a DAFT ``metadata`` block.

    Only ``type`` is required by the spec. ``date`` defaults to today; other
    fields are emitted only when provided. Key ordering is stable
    (type, date, description, license, tags) for readable on-disk diffs.

    Args:
        type_str: DAFT metadata type literal (e.g. ``"video"``, ``"events"``).
        iso_date: ISO-formatted date. Defaults to today.
        description: Optional human-readable description.
        license_str: Optional license string.
        tags: Optional iterable of tags.
    Returns:
        A DAFT-compliant metadata mapping.
    """
    block: dict[str, Any] = {"type": type_str, "date": iso_date or _date.today().isoformat()}
    if description is not None:
        block["description"] = description
    if license_str is not None:
        block["license"] = license_str
    if tags is not None:
        block["tags"] = list(tags)
    return block


def daft_envelope(
    type_: str,
    ctx: SceneContext,
    *,
    description: str | None = None,
    include_scene_id: bool = True,
) -> dict[str, Any]:
    """
    Build a DAFT envelope (``version`` + scene id + ``metadata``).

    Args:
        type_: DAFT metadata type literal (e.g. ``"video"``, ``"events"``).
        ctx: Scene context used to derive scene id and ISO date.
        description: Optional human-readable description for the metadata block.
        include_scene_id: When False, omit the scene-anchor field. Use for
            scene-level files like ``instances.json`` that reject the anchor.
    Returns:
        A DAFT-compliant envelope mapping.
    """
    out: dict[str, Any] = {"version": DAFT_VERSION}
    if include_scene_id:
        out[ctx.scene_id_field] = ctx.media_id
    out["metadata"] = metadata_block(
        type_,
        iso_date=ctx.iso_date,
        description=description,
        license_str=ctx.license_str,
        tags=ctx.tags if ctx.tags else None,
    )
    return out


__all__ = [
    "DAFT_VERSION",
    "daft_envelope",
    "metadata_block",
]
