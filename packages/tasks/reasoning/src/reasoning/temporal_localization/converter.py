# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured query results -> DAFT ``task/temporal_localization.json``.

DAFT v3 ``temporal_localization.json`` is the inverse of
``temporal_description``: instead of describing a known segment, the
task is to find the segment that matches a natural-language query.
Each item carries:

- the search window (``t1``/``t2`` — usually the full video span)
- the query (``question``)
- the localized event boundaries (``answer = {start, end}``)
- optional ``reasoning`` (the LLM's justification)
- optional ``video_type`` enum ``{anomaly, normal}``

This module is a *pure transformation* — it takes a list of structured
items (typically produced by an LLM grounding pass over per-window
captions, but the source is irrelevant) and returns a DAFT-compliant
payload. No LLM, no network, no domain assumptions.

Use-case agnosticism: the converter never invents queries, never
defaults a ``video_type``, never enforces a specific event taxonomy.
The query bank is entirely the caller's responsibility — this module
only validates the *shape* the caller produces.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from reasoning.common import DaftConvertError, SceneContext, daft_envelope
from reasoning.converter_utils import coerce_timecode, require_nonempty_string
from reasoning.timecodes import seconds_to_timecode, timecode_to_seconds

VideoType = Literal["anomaly", "normal"]

_VIDEO_TYPE_VALUES: tuple[str, ...] = ("anomaly", "normal")


def to_daft_temporal_localization(
    items: Iterable[dict[str, Any]],
    *,
    ctx: SceneContext,
    duration: float | None = None,
) -> dict[str, Any] | None:
    """Convert structured grounding items into a DAFT temporal_localization payload.

    Each input item must be a mapping with:

    - ``question`` (str): the natural-language query
    - ``answer`` (dict with ``start`` and ``end``): the localized
      boundaries; both can be numeric seconds or DAFT timecode strings
    - ``t1`` (numeric seconds or timecode str): start of the search
      window the LLM was given. Defaults to ``0`` when omitted.
    - ``t2`` (numeric seconds or timecode str): end of the search
      window. Defaults to ``duration`` when omitted; raises if neither
      is supplied.

    Optional per-item:

    - ``video_type``: ``"anomaly"`` or ``"normal"``
    - ``reasoning``: the LLM's justification trace

    ``duration``, when provided, clamps every timecode to ``[0, duration]``.

    Returns ``None`` when ``items`` is empty (DAFT requires
    ``minItems: 1`` on ``items``); callers should skip writing in that
    case.

    Raises :class:`DaftConvertError` for image contexts (no temporal
    axis), missing/empty required fields, unknown ``video_type`` values,
    answers where ``end < start``, or any timecode that fails the
    schema regex.
    """
    if ctx.is_image:
        raise DaftConvertError(
            "temporal_localization.json is not produced for image scenes; "
            "the schema requires a temporal axis"
        )

    item_list = list(items)
    if not item_list:
        return None

    items_out = [
        _to_item(item, idx, ctx=ctx, duration=duration) for idx, item in enumerate(item_list)
    ]

    out = daft_envelope("temporal_localization", ctx, include_scene_id=False)
    out["items"] = items_out
    return out


def _to_item(
    item: Any,
    idx: int,
    *,
    ctx: SceneContext,
    duration: float | None,
) -> dict[str, Any]:
    """Validate and normalize one localization item."""
    if not isinstance(item, dict):
        raise DaftConvertError(
            f"temporal_localization item[{idx}] must be a dict, got {type(item).__name__}"
        )

    question = _require_nonempty_string(
        item.get("question"),
        field=f"item[{idx}].question",
    )

    answer_raw = item.get("answer")
    if not isinstance(answer_raw, dict):
        raise DaftConvertError(
            f"temporal_localization item[{idx}].answer must be a dict with start/end, "
            f"got {type(answer_raw).__name__}"
        )
    ans_start = _coerce_timecode(answer_raw.get("start"), idx, "answer.start", duration=duration)
    ans_end = _coerce_timecode(answer_raw.get("end"), idx, "answer.end", duration=duration)
    if _to_secs(ans_end) < _to_secs(ans_start):
        raise DaftConvertError(
            f"temporal_localization item[{idx}].answer.end ({ans_end}) < answer.start ({ans_start})"
        )

    # Defaulting rules for the search-window bounds:
    # - t1 defaults to 0 (callers usually search the full clip)
    # - t2 defaults to ``duration`` when supplied; otherwise required
    t1_raw = item.get("t1")
    if t1_raw is None:
        t1 = seconds_to_timecode(0.0)
    else:
        t1 = _coerce_timecode(t1_raw, idx, "t1", duration=duration)

    t2_raw = item.get("t2")
    if t2_raw is None:
        if duration is None:
            raise DaftConvertError(
                f"temporal_localization item[{idx}] missing t2 and no fallback duration "
                "supplied to the converter"
            )
        t2 = seconds_to_timecode(float(duration))
    else:
        t2 = _coerce_timecode(t2_raw, idx, "t2", duration=duration)
    if _to_secs(t2) < _to_secs(t1):
        raise DaftConvertError(f"temporal_localization item[{idx}].t2 ({t2}) < t1 ({t1})")

    out: dict[str, Any] = {
        "video_id": ctx.media_id,
        "t1": t1,
        "t2": t2,
        "question": question,
        "answer": {"start": ans_start, "end": ans_end},
    }

    video_type = item.get("video_type")
    if video_type is not None:
        if not isinstance(video_type, str) or video_type not in _VIDEO_TYPE_VALUES:
            raise DaftConvertError(
                f"temporal_localization item[{idx}].video_type must be one of "
                f"{list(_VIDEO_TYPE_VALUES)}, got {video_type!r}"
            )
        out["video_type"] = video_type

    reasoning = item.get("reasoning")
    if reasoning is not None:
        if not isinstance(reasoning, str):
            raise DaftConvertError(
                f"temporal_localization item[{idx}].reasoning must be a string, "
                f"got {type(reasoning).__name__}"
            )
        r = reasoning.strip()
        if r:
            out["reasoning"] = r

    return out


def _coerce_timecode(
    value: Any,
    idx: int,
    label: str,
    *,
    duration: float | None,
) -> str:
    return coerce_timecode(
        value,
        field=f"temporal_localization item[{idx}].{label}",
        duration=duration,
        duration_field="temporal_localization duration",
    )


def _to_secs(tc: str) -> float:
    """Convenience for monotonicity checks. Trusts the caller-provided
    timecode string (always produced by ``_coerce_timecode``, so safe)."""
    return timecode_to_seconds(tc)


def _require_nonempty_string(value: Any, *, field: str) -> str:
    return require_nonempty_string(
        value,
        field=field,
        prefix="temporal_localization",
        quote_field=True,
    )


__all__ = ["VideoType", "to_daft_temporal_localization"]
