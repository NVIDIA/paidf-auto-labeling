# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PL window outputs -> DAFT ``task/temporal_description.json``.

DAFT v3 ``temporal_description.json`` is a dense video-captioning task: each
item is a question-answer pair anchored to a temporal segment ``[t1, t2]``.
The semantics mirror DAFT ``contextual/chunks.json`` (per-window prose) but
in the *task* namespace, so downstream consumers can train Q&A heads off
the same segmentation.

This converter is a pure re-pivot of PL's window data — the same input
shape that ``to_daft_chunks`` accepts. No LLM call is made: each window
becomes one item with the window caption as the answer.

Use-case agnosticism (the deliberate design constraints):

- ``question_template`` is an ordinary ``str.format`` template that the
  caller controls. The default (``"What happened in the video between
  {t1} and {t2}?"``) is the schema's own example; pass anything else if
  your domain wants a different framing (e.g. ``"What are the key safety
  observations between {t1} and {t2}?"``). The template receives ``t1``
  and ``t2`` as DAFT timecode strings.

- ``video_type`` is opt-in. The schema's enum is ``{"anomaly", "normal"}``
  — useful for ITS / surveillance datasets, meaningless for many others.
  Pass it when your config classifies the clip; omit it (default) when
  no such classification exists. The converter never invents a value.

- ``description_keys`` lets the caller flip between raw VLM caption and
  the LM-rewritten ``enhanced_caption`` (same knob as ``to_daft_chunks``)
  without the converter having to know which one is "right" for the use
  case.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from core.formats.daft.converter_utils import clamp_nonnegative_seconds
from core.formats.daft.envelope import daft_envelope
from core.formats.daft.errors import DaftConvertError
from core.formats.daft.timecodes import seconds_to_timecode
from core.scene import SceneContext

VideoType = Literal["anomaly", "normal"]

# Same lookup families as ``to_daft_chunks`` so callers can hand the same
# window list to both converters without massaging key names.
_START_KEYS: tuple[str, ...] = ("start_s", "start", "start_seconds", "start_time")
_END_KEYS: tuple[str, ...] = ("end_s", "end", "end_seconds", "end_time")
_DESC_KEYS: tuple[str, ...] = ("description", "caption", "enhanced_caption", "summary")

# Schema's own example is the most use-case-neutral starting point. The
# caller can override; the placeholders ``{t1}`` and ``{t2}`` are guaranteed
# to receive DAFT-formatted timecode strings.
DEFAULT_QUESTION_TEMPLATE: str = "What happened in the video between {t1} and {t2}?"


def to_daft_temporal_description(
    windows: Iterable[dict[str, Any]],
    *,
    ctx: SceneContext,
    question_template: str = DEFAULT_QUESTION_TEMPLATE,
    description_keys: tuple[str, ...] | None = None,
    duration: float | None = None,
    video_type: VideoType | None = None,
    reasoning_key: str | None = None,
) -> dict[str, Any] | None:
    """Convert per-window PL data into a DAFT ``temporal_description.json`` payload.

    Each input window must carry a start timestamp (one of ``start_s`` /
    ``start`` / ``start_seconds`` / ``start_time``), an end timestamp
    (same key family), and an answer text (one of ``description`` /
    ``caption`` / ``enhanced_caption`` / ``summary``). Override
    ``description_keys`` to flip the lookup order — pass
    ``("enhanced_caption", "caption")`` to prefer the LM-rewritten caption.

    ``question_template`` is formatted with ``t1`` and ``t2`` (already as
    DAFT timecode strings) so the caller can phrase the question however
    fits the use case without the converter knowing the domain.

    ``video_type`` is emitted only when supplied — the schema enum is
    ``{"anomaly", "normal"}`` and is meaningful only for some datasets.

    ``reasoning_key`` lets callers thread a per-window reasoning trace
    (e.g. populated by an upstream LLM-enrichment step) into each item's
    ``reasoning`` field. When ``None`` (default) no reasoning is emitted,
    even if the window has a key by that name; this keeps the output
    deterministic and the caller in control of what goes through.

    Returns ``None`` when ``windows`` is empty (DAFT requires
    ``minItems: 1`` on ``items``); callers should skip writing in that
    case.

    Raises ``DaftConvertError`` for image ``ctx`` (no temporal axis) and
    for any window missing a start / end / description key, or with
    non-numeric or non-monotonic timestamps.
    """
    if ctx.is_image:
        raise DaftConvertError("temporal_description.json is not produced for image scenes")

    win_list = list(windows)
    if not win_list:
        return None

    desc_keys = description_keys or _DESC_KEYS

    items_out: list[dict[str, Any]] = []
    for idx, win in enumerate(win_list):
        if not isinstance(win, dict):
            raise DaftConvertError(
                f"temporal_description.json window[{idx}] must be a dict, got {type(win).__name__}"
            )
        items_out.append(
            _to_daft_item(
                win,
                idx,
                ctx=ctx,
                question_template=question_template,
                desc_keys=desc_keys,
                duration=duration,
                video_type=video_type,
                reasoning_key=reasoning_key,
            )
        )

    out = daft_envelope("temporal_description", ctx, include_scene_id=False)
    out["items"] = items_out
    return out


def _to_daft_item(
    win: dict[str, Any],
    idx: int,
    *,
    ctx: SceneContext,
    question_template: str,
    desc_keys: tuple[str, ...],
    duration: float | None,
    video_type: VideoType | None,
    reasoning_key: str | None,
) -> dict[str, Any]:
    start_s = _pick_seconds(win, _START_KEYS, idx, "start")
    end_s = _pick_seconds(win, _END_KEYS, idx, "end")
    if end_s < start_s:
        raise DaftConvertError(
            f"temporal_description.json item[{idx}] end {end_s} < start {start_s}"
        )
    start_s = _clamp(start_s, duration)
    end_s = _clamp(end_s, duration)

    t1 = seconds_to_timecode(start_s)
    t2 = seconds_to_timecode(end_s)

    answer = _pick_description(win, desc_keys, idx)

    try:
        question = question_template.format(t1=t1, t2=t2)
    except (KeyError, IndexError) as exc:
        raise DaftConvertError(
            f"temporal_description.json question_template {question_template!r} "
            f"references unknown placeholder: {exc}"
        ) from exc
    if not question.strip():
        raise DaftConvertError(
            f"temporal_description.json item[{idx}] question is empty after formatting"
        )

    out: dict[str, Any] = {
        "video_id": ctx.media_id,
        "t1": t1,
        "t2": t2,
        "question": question,
        "answer": answer,
    }
    if video_type is not None:
        if video_type not in ("anomaly", "normal"):
            raise DaftConvertError(
                "temporal_description.json video_type must be 'anomaly' or 'normal', "
                f"got {video_type!r}"
            )
        out["video_type"] = video_type
    if reasoning_key is not None:
        rv = win.get(reasoning_key)
        if isinstance(rv, str) and rv.strip():
            out["reasoning"] = rv.strip()
    return out


def _pick_seconds(win: dict[str, Any], keys: tuple[str, ...], idx: int, label: str) -> float:
    for k in keys:
        if k in win:
            v = win[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise DaftConvertError(
                    f"temporal_description.json window[{idx}] {label} field {k!r} "
                    f"must be a number, got {v!r}"
                )
            return float(v)
    raise DaftConvertError(
        f"temporal_description.json window[{idx}] missing {label} timestamp; "
        f"expected one of {list(keys)}"
    )


def _pick_description(win: dict[str, Any], keys: tuple[str, ...], idx: int) -> str:
    for k in keys:
        v = win.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    raise DaftConvertError(
        f"temporal_description.json item[{idx}] missing answer text; expected one of {list(keys)}"
    )


def _clamp(s: float, duration: float | None) -> float:
    duration_s = float(duration) if duration is not None else None
    return clamp_nonnegative_seconds(s, duration_s)


__all__ = ["DEFAULT_QUESTION_TEMPLATE", "VideoType", "to_daft_temporal_description"]
