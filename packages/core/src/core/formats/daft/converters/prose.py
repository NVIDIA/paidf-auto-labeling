# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PL prose outputs -> DAFT ``task/scene_description.json`` + ``task/video_summarization.json``.

These two task types are pure captioning: each item is a free-form
``(question, answer)`` pair, optionally with a ``reasoning`` trace. Unlike
mcq/bcq there are no options to route, no answer regex to satisfy, and no
upstream filtering invariants — they exist mainly to expose the prose the VLM
already produces (``video.json["scene_description"]``,
``video.json["event_summary"]``, ``image.json["caption"]``) in DAFT's task
namespace so downstream consumers can train captioning heads off the same
scene without re-parsing ``contextual/``.

The schemas are nearly twins:

- ``scene_description``: items have ``video_id`` xor ``image_id``,
  ``question``, ``answer``, optional ``reasoning``.
- ``video_summarization``: same, plus an optional ``timestamp: {start, end}``
  window (seconds, ``minimum: 0``).

Both files share the standard task envelope (``include_scene_id=False``;
the per-item ``video_id``/``image_id`` carries the cross-reference into
``contextual/``).

Each converter returns ``None`` when the input text is empty/whitespace,
because DAFT's ``minItems: 1`` on ``items`` means an empty stub is illegal —
the caller should skip writing in that case rather than emit something the
``tao-daft validate`` CLI will reject.
"""

from __future__ import annotations

import math
from typing import Any

from core.formats.daft.envelope import daft_envelope
from core.formats.daft.errors import DaftConvertError
from core.scene import SceneContext


def _clean(text: Any) -> str | None:
    """Return ``text.strip()`` if truthy, else ``None``.

    Centralizes the "treat empty/whitespace as missing" rule so converters
    don't accidentally emit ``"answer": ""`` (passes ``"type": "string"`` but
    is semantically garbage)."""
    if text is None:
        return None
    if not isinstance(text, str):
        return None
    s = text.strip()
    return s if s else None


def _scene_caption_item(
    *,
    ctx: SceneContext,
    question: str,
    answer: str,
    reasoning: str | None,
) -> dict[str, Any]:
    """Build a single captioning item with the right scene-id field.

    Shared by both prose-task converters since their item shapes only differ
    in ``video_summarization``'s optional ``timestamp`` field, which is layered
    on by the caller."""
    item: dict[str, Any] = {
        ctx.scene_id_field: ctx.media_id,
        "question": question,
        "answer": answer,
    }
    if reasoning is not None:
        item["reasoning"] = reasoning
    return item


def to_daft_scene_description(
    answer: str,
    *,
    ctx: SceneContext,
    question: str = "Describe the scene.",
    reasoning: str | None = None,
) -> dict[str, Any] | None:
    """Wrap a single scene-description string in a DAFT ``scene_description`` payload.

    Returns ``None`` when ``answer`` is empty/whitespace so the caller can skip
    writing (DAFT requires ``minItems: 1`` on ``items``).

    Works for both image and video scenes; the per-item scene-id field is
    chosen from ``ctx.is_image`` to satisfy the schema's
    ``oneOf(video_id, image_id)`` constraint."""
    answer_clean = _clean(answer)
    if answer_clean is None:
        return None
    question_clean = _clean(question)
    if question_clean is None:
        raise DaftConvertError("scene_description question must be a non-empty string")

    item = _scene_caption_item(
        ctx=ctx,
        question=question_clean,
        answer=answer_clean,
        reasoning=_clean(reasoning),
    )
    out = daft_envelope("scene_description", ctx, include_scene_id=False)
    out["items"] = [item]
    return out


def to_daft_video_summarization(
    answer: str,
    *,
    ctx: SceneContext,
    question: str = "Summarize the events in the video.",
    reasoning: str | None = None,
    timestamp: tuple[float, float] | None = None,
) -> dict[str, Any] | None:
    """Wrap a single video-summary string in a DAFT ``video_summarization`` payload.

    Returns ``None`` when ``answer`` is empty/whitespace so the caller can
    skip writing. Rejects image scenes — summarization is inherently
    temporal and the schema has no meaningful ``image_id``-only path for it.

    ``timestamp`` is an optional ``(start_seconds, end_seconds)`` window for
    the summary; service pivots pass ``(0.0, duration)`` when the video
    duration is available, so consumers know the summary covers the full clip.
    """
    if ctx.is_image:
        raise DaftConvertError(
            "video_summarization is not defined for image scenes; "
            "construct a SceneContext with is_image=False or skip the call"
        )
    answer_clean = _clean(answer)
    if answer_clean is None:
        return None
    question_clean = _clean(question)
    if question_clean is None:
        raise DaftConvertError("video_summarization question must be a non-empty string")

    item = _scene_caption_item(
        ctx=ctx,
        question=question_clean,
        answer=answer_clean,
        reasoning=_clean(reasoning),
    )
    if timestamp is not None:
        start, end = timestamp
        if not math.isfinite(start) or not math.isfinite(end):
            raise DaftConvertError(
                f"video_summarization timestamp values must be finite, got ({start}, {end})"
            )
        if start < 0 or end < 0:
            raise DaftConvertError(
                f"video_summarization timestamp must be >= 0, got ({start}, {end})"
            )
        if end < start:
            raise DaftConvertError(f"video_summarization timestamp end < start: ({start}, {end})")
        item["timestamp"] = {"start": float(start), "end": float(end)}

    out = daft_envelope("video_summarization", ctx, include_scene_id=False)
    out["items"] = [item]
    return out


__all__ = ["to_daft_scene_description", "to_daft_video_summarization"]
