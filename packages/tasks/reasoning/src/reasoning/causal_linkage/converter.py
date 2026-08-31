# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured causal-pair items -> DAFT ``task/causal_linkage.json``.

DAFT v3 ``causal_linkage.json`` asks: "given two timestamps t1 and t2,
explain the causal relationship between the event at t1 and the
situation at t2." Each item carries:

- ``video_id`` (no ``image_id`` flavor — causal linkage is inherently
  temporal).
- ``t1`` / ``t2`` timecodes (regex ``^(\\d{2}:)?\\d{2}:\\d{2}(\\.\\d+)?$``).
- ``question`` (str, non-empty) — the natural-language prompt.
- ``answer`` (str, non-empty) — the free-text causal explanation.
- optional ``video_type`` enum ``{"anomaly", "normal"}``.
- optional ``reasoning`` trace.

This module is a *pure transformation* — it takes structured items
(typically produced by an LLM grounding pass over per-window captions,
but the source is irrelevant) and returns a DAFT-compliant payload.
The matching LLM adapter lives in :mod:`reasoning.causal_linkage.llm`.

Use-case agnosticism: the converter never invents pairs, never
defaults a ``video_type``, never enforces a domain-specific event
taxonomy. The pair bank is entirely the caller's responsibility — this
module only validates the *shape* the caller produces.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from reasoning.common import DaftConvertError, SceneContext, daft_envelope
from reasoning.converter_utils import coerce_timecode, require_nonempty_string
from reasoning.timecodes import timecode_to_seconds

VideoType = Literal["anomaly", "normal"]

_VIDEO_TYPE_VALUES: tuple[str, ...] = ("anomaly", "normal")


def to_daft_causal_linkage(
    items: Iterable[dict[str, Any]],
    *,
    ctx: SceneContext,
    duration: float | None = None,
) -> dict[str, Any] | None:
    """Convert structured causal-pair items into a DAFT payload.

    Each input item must be a mapping with:

    - ``t1`` / ``t2`` (numeric seconds *or* DAFT timecode string): the
      cause / effect timestamps. ``t2 >= t1`` is enforced.
    - ``question`` (str, non-empty): the natural-language causal prompt.
    - ``answer`` (str, non-empty): the LLM (or hand-authored) answer.

    Optional per-item:

    - ``video_type``: ``"anomaly"`` or ``"normal"``.
    - ``reasoning``: step-by-step rationale.

    ``duration``, when provided, clamps every timecode to ``[0, duration]``.

    Returns a complete DAFT payload, or ``None`` when ``items`` is
    empty (schema requires ``minItems: 1``); callers should skip
    writing in that case.

    Raises :class:`DaftConvertError` for image contexts (no temporal
    axis), missing/empty required fields, malformed timecodes,
    ``t2 < t1``, or unknown ``video_type`` values.
    """
    if ctx.is_image:
        raise DaftConvertError(
            "causal_linkage.json is not produced for image scenes; the schema "
            "requires a temporal axis"
        )

    item_list = list(items)
    if not item_list:
        return None

    items_out = [_to_item(it, idx, ctx=ctx, duration=duration) for idx, it in enumerate(item_list)]

    out = daft_envelope("causal_linkage", ctx, include_scene_id=False)
    out["items"] = items_out
    return out


def _to_item(
    item: Any,
    idx: int,
    *,
    ctx: SceneContext,
    duration: float | None,
) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise DaftConvertError(
            f"causal_linkage item[{idx}] must be a dict, got {type(item).__name__}"
        )

    t1 = _coerce_timecode(item.get("t1"), idx, "t1", duration=duration)
    t2 = _coerce_timecode(item.get("t2"), idx, "t2", duration=duration)
    if _to_secs(t2) < _to_secs(t1):
        raise DaftConvertError(f"causal_linkage item[{idx}].t2 ({t2}) < t1 ({t1})")

    question = _require_nonempty_string(
        item.get("question"), field=f"causal_linkage item[{idx}].question"
    )
    answer = _require_nonempty_string(
        item.get("answer"), field=f"causal_linkage item[{idx}].answer"
    )

    out: dict[str, Any] = {
        "video_id": ctx.media_id,
        "t1": t1,
        "t2": t2,
        "question": question,
        "answer": answer,
    }

    video_type = item.get("video_type")
    if video_type is not None:
        if not isinstance(video_type, str) or video_type not in _VIDEO_TYPE_VALUES:
            raise DaftConvertError(
                f"causal_linkage item[{idx}].video_type must be one of "
                f"{list(_VIDEO_TYPE_VALUES)}, got {video_type!r}"
            )
        # video_type is positioned after video_id and before t1/t2 in
        # the schema example; respect that order so on-disk diffs are
        # readable.
        new_out: dict[str, Any] = {"video_id": out["video_id"], "video_type": video_type}
        for k in ("t1", "t2", "question", "answer"):
            new_out[k] = out[k]
        out = new_out

    reasoning = item.get("reasoning")
    if reasoning is not None:
        if not isinstance(reasoning, str):
            raise DaftConvertError(
                f"causal_linkage item[{idx}].reasoning must be a string, "
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
        field=f"causal_linkage item[{idx}].{label}",
        duration=duration,
        duration_field="causal_linkage duration",
    )


def _to_secs(tc: str) -> float:
    """Convenience for monotonicity checks. Trusts the caller-provided
    timecode string (always produced by ``_coerce_timecode``, so safe)."""
    return timecode_to_seconds(tc)


def _require_nonempty_string(value: Any, *, field: str) -> str:
    return require_nonempty_string(value, field=field)


__all__ = ["VideoType", "to_daft_causal_linkage"]
