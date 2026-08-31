# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PL window outputs -> DAFT ``contextual/chunks.json``.

DAFT v3 ``chunks.json`` is a dense temporal segmentation of a video: each
chunk is ``{chunk_id, start, end, description}`` plus optional ``tags``.
This is essentially a re-pivot of PL's window-based VLM data — the
windowed MCQ runners (``window_vlm_llm``, ``window_direct_vlm``) already
emit per-window entries with ``start_s``, ``end_s`` and a per-window
caption into ``sidecars/metadata.json``. Pipeline.py reads that sidecar
back and drives this converter; no extra VLM call is needed.

The converter is permissive about input keys (``start_s`` or ``start``,
``caption`` or ``description``, etc.) so callers don't have to massage PL's
on-disk shape before calling. It is strict about output: timecodes are
formatted via ``seconds_to_timecode`` (matches the schema regex), chunk
ordering is preserved, and chunk ids are auto-generated when absent.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from core.formats.daft.converter_utils import (
    clamp_nonnegative_seconds,
    validate_nonnegative_duration,
)
from core.formats.daft.envelope import daft_envelope
from core.formats.daft.errors import DaftConvertError
from core.formats.daft.timecodes import seconds_to_timecode
from core.scene import SceneContext

# DAFT v3 ``chunks.json`` requires ``video_id`` at the top level — like
# ``video.json`` and unlike ``instances.json``. Image scenes have no
# temporal axis, so chunks.json is video-only by construction.

_START_KEYS: tuple[str, ...] = ("start_s", "start", "start_seconds", "start_time")
_END_KEYS: tuple[str, ...] = ("end_s", "end", "end_seconds", "end_time")
_DESC_KEYS: tuple[str, ...] = ("description", "caption", "enhanced_caption", "summary")


def to_daft_chunks(
    windows: Iterable[dict[str, Any]],
    *,
    ctx: SceneContext,
    duration: float | None = None,
    description_keys: tuple[str, ...] | None = None,
) -> dict[str, Any] | None:
    """Convert a list of PL window dicts into a DAFT ``chunks.json`` payload.

    Each input window must carry a start timestamp (one of ``start_s`` /
    ``start`` / ``start_seconds`` / ``start_time``), an end timestamp (same
    key family), and a description (one of ``description`` / ``caption`` /
    ``enhanced_caption`` / ``summary``). The first key found wins, in the
    listed order — pass ``description_keys`` to override the lookup order
    (e.g. ``("enhanced_caption", "caption")`` to prefer the LM-rewritten
    caption when present).

    ``chunk_id`` is auto-generated as ``chunk_001``, ``chunk_002``, ... when
    the source window lacks one. Timecodes are clamped to ``[0, duration]``
    when ``duration`` is provided so a probe-vs-metadata mismatch can't
    push a chunk past the video end.

    Returns ``None`` when ``windows`` is empty (DAFT requires
    ``minItems: 1`` on ``chunks``); callers should skip writing in that
    case.

    Raises ``DaftConvertError`` for image ``ctx`` (no temporal axis), for
    any window missing all start / end / description keys, and for
    timestamps that are not numeric or are non-monotonic
    (``end < start``).
    """
    if ctx.is_image:
        raise DaftConvertError("chunks.json is not produced for image scenes")

    duration = _validate_duration(duration)
    win_list = list(windows)
    if not win_list:
        return None

    desc_keys = description_keys or _DESC_KEYS

    chunks_out: list[dict[str, Any]] = []
    for idx, win in enumerate(win_list):
        if not isinstance(win, dict):
            raise DaftConvertError(
                f"chunks.json window[{idx}] must be a dict, got {type(win).__name__}"
            )
        chunks_out.append(_to_daft_chunk(win, idx, duration=duration, desc_keys=desc_keys))

    out = daft_envelope("chunks", ctx)
    out["chunks"] = chunks_out
    metadata = out.pop("metadata")
    out["metadata"] = metadata
    return out


def _to_daft_chunk(
    win: dict[str, Any],
    idx: int,
    *,
    duration: float | None,
    desc_keys: tuple[str, ...],
) -> dict[str, Any]:
    chunk_id = win.get("chunk_id")
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        chunk_id = f"chunk_{idx + 1:03d}"

    start_s = _pick_seconds(win, _START_KEYS, idx, "start")
    end_s = _pick_seconds(win, _END_KEYS, idx, "end")
    if end_s < start_s:
        raise DaftConvertError(
            f"chunks.json chunk[{idx}] {chunk_id!r} end {end_s} < start {start_s}"
        )
    start_s = _clamp(start_s, duration)
    end_s = _clamp(end_s, duration)

    description = _pick_description(win, desc_keys, idx, chunk_id)

    out: dict[str, Any] = {
        "chunk_id": chunk_id,
        "start": seconds_to_timecode(start_s),
        "end": seconds_to_timecode(end_s),
        "description": description,
    }

    tags = win.get("tags")
    if isinstance(tags, list) and tags:
        clean_tags = [str(t) for t in tags if isinstance(t, str) and t.strip()]
        if clean_tags:
            out["tags"] = clean_tags
    return out


def _pick_seconds(win: dict[str, Any], keys: tuple[str, ...], idx: int, label: str) -> float:
    for k in keys:
        if k in win:
            v = win[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise DaftConvertError(
                    f"chunks.json window[{idx}] {label} field {k!r} must be a number, got {v!r}"
                )
            secs = float(v)
            if not math.isfinite(secs):
                raise DaftConvertError(
                    f"chunks.json window[{idx}] {label} field {k!r} must be finite, got {secs}"
                )
            return secs
    raise DaftConvertError(
        f"chunks.json window[{idx}] missing {label} timestamp; expected one of {list(keys)}"
    )


def _pick_description(win: dict[str, Any], keys: tuple[str, ...], idx: int, chunk_id: str) -> str:
    for k in keys:
        v = win.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    raise DaftConvertError(
        f"chunks.json chunk[{idx}] {chunk_id!r} missing description; expected one of {list(keys)}"
    )


def _clamp(s: float, duration: float | None) -> float:
    return clamp_nonnegative_seconds(s, duration)


def _validate_duration(duration: float | None) -> float | None:
    return validate_nonnegative_duration(duration, field="chunks.json duration")


__all__ = ["to_daft_chunks"]
