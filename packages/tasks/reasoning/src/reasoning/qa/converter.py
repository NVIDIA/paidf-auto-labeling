# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured QA inputs -> DAFT open-ended task payloads.

Three sibling task schemas in ``metropolis-v3.0`` that share an
"answer the question, optionally with a reasoning trace" shape but
differ in the answer constraint:

- ``open_qa``: free-form answer string. No regex.
- ``mcq_openended``: answer must start with a single uppercase letter,
  then ``". "``, then 1+ characters of explanation
  (regex ``^[A-Z]\\. [\\s\\S]+``). Item may carry an ``options`` dict
  mapping letters to choice text; when present, the answer's leading
  letter must be a key in ``options``.
- ``bcq_openended``: answer must start with ``"Yes"`` or ``"No"``, then
  ``". "``, then 1+ characters of explanation
  (regex ``^(Yes|No)\\. [\\s\\S]+``).

All three share the same DAFT envelope (``include_scene_id=False``;
each item carries ``video_id`` xor ``image_id`` derived from the
``SceneContext``) and the same optional ``reasoning`` passthrough on
each item.

This module is a *pure transformation* layer — it takes structured
items (typically produced by an LLM call over per-window captions, but
the source is irrelevant) and returns a DAFT-compliant payload. No
network, no LLM, no domain assumptions. The shape constraints are
enforced *here*, never in the prompt: a misbehaving model can't poison
the on-disk DAFT output. The matching LLM adapter lives in
``reasoning.qa.llm``.

Use-case agnosticism: the question bank is entirely caller-supplied;
we never inject default questions, never rewrite the question text,
never enforce a domain-specific option taxonomy. ``mcq_openended``'s
``options`` is optional precisely because the schema lets callers
embed choices inline in the question prose; honoring that gives
callers two equally-valid authoring styles.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from reasoning.common import DaftConvertError, SceneContext, daft_envelope
from reasoning.converter_utils import require_nonempty_string as _require_nonempty_string

# ``mcq_openended`` answer regex (mirrors the schema literal). The
# explanation tail is required by ``[\\s\\S]+`` (at least one character
# after the letter+period+space prefix). Compiled once at module load.
_MCQ_OPENENDED_ANSWER_RE: re.Pattern[str] = re.compile(r"^[A-Z]\. [\s\S]+")

# ``bcq_openended`` answer regex (mirrors the schema literal).
_BCQ_OPENENDED_ANSWER_RE: re.Pattern[str] = re.compile(r"^(Yes|No)\. [\s\S]+")

# Letter alphabet used to validate ``options`` keys. The schema only
# constrains keys via ``additionalProperties.minLength: 1`` (no enum),
# but DAFT examples and the answer regex make A-Z the canonical key
# space; we enforce that to keep cross-format consistency with
# ``task.mcq``.
_OPTION_LETTER_RE: re.Pattern[str] = re.compile(r"^[A-Z]$")


# ---------------------------------------------------------------------------
# open_qa
# ---------------------------------------------------------------------------


def to_daft_open_qa(
    items: Iterable[dict[str, Any]],
    *,
    ctx: SceneContext,
) -> dict[str, Any] | None:
    """Convert structured QA items into a DAFT ``open_qa.json`` payload.

    Each input item must be a mapping with:

    - ``question`` (str, non-empty): the natural-language question.
    - ``answer`` (str, non-empty): the free-form answer.

    Optional per-item:

    - ``reasoning`` (str): step-by-step reasoning trace.

    The per-item scene id (``video_id`` for video scenes, ``image_id``
    for image scenes) is derived from ``ctx`` to satisfy the schema's
    ``oneOf(video_id, image_id)`` constraint.

    Returns a complete DAFT payload, or ``None`` when ``items`` is
    empty (the schema requires ``minItems: 1`` on ``items``; callers
    should skip writing rather than emit an illegal stub).

    Raises :class:`DaftConvertError` for malformed items (non-dict,
    missing fields, wrong types).
    """
    item_list = list(items)
    if not item_list:
        return None

    items_out = [_to_open_qa_item(it, idx, ctx=ctx) for idx, it in enumerate(item_list)]

    out = daft_envelope("open_qa", ctx, include_scene_id=False)
    out["items"] = items_out
    return out


def _to_open_qa_item(item: Any, idx: int, *, ctx: SceneContext) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise DaftConvertError(f"open_qa item[{idx}] must be a dict, got {type(item).__name__}")
    question = _require_nonempty_string(item.get("question"), field=f"open_qa item[{idx}].question")
    answer = _require_nonempty_string(item.get("answer"), field=f"open_qa item[{idx}].answer")

    out: dict[str, Any] = {
        ctx.scene_id_field: ctx.media_id,
        "question": question,
        "answer": answer,
    }
    reasoning = _optional_reasoning(item.get("reasoning"), idx=idx, kind="open_qa")
    if reasoning is not None:
        out["reasoning"] = reasoning
    return out


# ---------------------------------------------------------------------------
# mcq_openended
# ---------------------------------------------------------------------------


def to_daft_mcq_openended(
    items: Iterable[dict[str, Any]],
    *,
    ctx: SceneContext,
) -> dict[str, Any] | None:
    """Convert structured MCQ-with-explanation items into a DAFT payload.

    Each input item must be a mapping with:

    - ``question`` (str, non-empty).
    - ``answer`` (str): must match ``^[A-Z]\\. [\\s\\S]+`` — a single
      uppercase letter, ``". "``, then the open-ended explanation.

    Optional per-item:

    - ``options`` (dict[str, str]): mapping of answer letter to choice
      text (e.g. ``{"A": "passenger car", "B": "truck"}``). When
      present:
        * Must have ``minProperties: 2``.
        * Every key must be a single uppercase letter (``[A-Z]``).
        * Every value must be a non-empty string.
        * The answer's leading letter must be a key in ``options``
          (otherwise the on-disk file would silently disagree with
          itself).
    - ``reasoning`` (str): step-by-step reasoning trace.

    Returns a complete DAFT payload, or ``None`` when ``items`` is
    empty (schema requires ``minItems: 1``).
    """
    item_list = list(items)
    if not item_list:
        return None

    items_out = [_to_mcq_openended_item(it, idx, ctx=ctx) for idx, it in enumerate(item_list)]

    out = daft_envelope("mcq_openended", ctx, include_scene_id=False)
    out["items"] = items_out
    return out


def _to_mcq_openended_item(item: Any, idx: int, *, ctx: SceneContext) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise DaftConvertError(
            f"mcq_openended item[{idx}] must be a dict, got {type(item).__name__}"
        )
    question = _require_nonempty_string(
        item.get("question"), field=f"mcq_openended item[{idx}].question"
    )
    answer = _require_nonempty_string(item.get("answer"), field=f"mcq_openended item[{idx}].answer")
    if not _MCQ_OPENENDED_ANSWER_RE.match(answer):
        raise DaftConvertError(
            f"mcq_openended item[{idx}].answer must match '^[A-Z]\\. [\\s\\S]+' "
            f"(got {answer!r}); the answer must start with a single uppercase letter, "
            "then '. ', then an explanation"
        )

    out: dict[str, Any] = {
        ctx.scene_id_field: ctx.media_id,
        "question": question,
        "answer": answer,
    }

    raw_options = item.get("options")
    if raw_options is not None:
        options = _normalize_options(raw_options, idx=idx)
        # Schema-shape consistency: the leading letter MUST be a
        # registered option key. Otherwise we'd ship a file where the
        # answer points at a key that doesn't exist in ``options``,
        # which is technically schema-valid but semantically broken.
        leading_letter = answer[0]
        if leading_letter not in options:
            raise DaftConvertError(
                f"mcq_openended item[{idx}].answer starts with {leading_letter!r} "
                f"which is not a key in options {sorted(options)}; the leading "
                "letter must select a registered option"
            )
        out["options"] = options

    reasoning = _optional_reasoning(item.get("reasoning"), idx=idx, kind="mcq_openended")
    if reasoning is not None:
        out["reasoning"] = reasoning
    return out


def _normalize_options(raw: Any, *, idx: int) -> dict[str, str]:
    """Validate the optional ``options`` mapping for ``mcq_openended``."""
    if not isinstance(raw, dict):
        raise DaftConvertError(
            f"mcq_openended item[{idx}].options must be a dict (letter -> choice text), "
            f"got {type(raw).__name__}"
        )
    if len(raw) < 2:
        raise DaftConvertError(
            f"mcq_openended item[{idx}].options has {len(raw)} entries; "
            "DAFT requires minProperties: 2"
        )
    out: dict[str, str] = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not _OPTION_LETTER_RE.match(k):
            raise DaftConvertError(
                f"mcq_openended item[{idx}].options key {k!r} must be a single "
                "uppercase letter (A-Z) to match the answer-letter alphabet"
            )
        if not isinstance(v, str):
            raise DaftConvertError(
                f"mcq_openended item[{idx}].options[{k!r}] must be a string, got {type(v).__name__}"
            )
        s = v.strip()
        if not s:
            raise DaftConvertError(
                f"mcq_openended item[{idx}].options[{k!r}] is empty after strip()"
            )
        out[k] = s
    return out


# ---------------------------------------------------------------------------
# bcq_openended
# ---------------------------------------------------------------------------


def to_daft_bcq_openended(
    items: Iterable[dict[str, Any]],
    *,
    ctx: SceneContext,
) -> dict[str, Any] | None:
    """Convert structured Yes/No-with-explanation items into a DAFT payload.

    Each input item must be a mapping with:

    - ``question`` (str, non-empty): the yes/no question.
    - ``answer`` (str): must match ``^(Yes|No)\\. [\\s\\S]+`` —
      ``"Yes"`` or ``"No"``, then ``". "``, then the explanation.

    Optional per-item:

    - ``reasoning`` (str): step-by-step reasoning trace.

    Returns a complete DAFT payload, or ``None`` when ``items`` is
    empty (schema requires ``minItems: 1``).
    """
    item_list = list(items)
    if not item_list:
        return None

    items_out = [_to_bcq_openended_item(it, idx, ctx=ctx) for idx, it in enumerate(item_list)]

    out = daft_envelope("bcq_openended", ctx, include_scene_id=False)
    out["items"] = items_out
    return out


def _to_bcq_openended_item(item: Any, idx: int, *, ctx: SceneContext) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise DaftConvertError(
            f"bcq_openended item[{idx}] must be a dict, got {type(item).__name__}"
        )
    question = _require_nonempty_string(
        item.get("question"), field=f"bcq_openended item[{idx}].question"
    )
    answer = _require_nonempty_string(item.get("answer"), field=f"bcq_openended item[{idx}].answer")
    if not _BCQ_OPENENDED_ANSWER_RE.match(answer):
        raise DaftConvertError(
            f"bcq_openended item[{idx}].answer must match '^(Yes|No)\\. [\\s\\S]+' "
            f"(got {answer!r}); the answer must start with 'Yes' or 'No', then "
            "'. ', then an explanation"
        )

    out: dict[str, Any] = {
        ctx.scene_id_field: ctx.media_id,
        "question": question,
        "answer": answer,
    }
    reasoning = _optional_reasoning(item.get("reasoning"), idx=idx, kind="bcq_openended")
    if reasoning is not None:
        out["reasoning"] = reasoning
    return out


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _optional_reasoning(value: Any, *, idx: int, kind: str) -> str | None:
    """Validate an optional ``reasoning`` field; return stripped value or ``None``.

    Empty / whitespace-only values are silently dropped so the key is
    omitted (the schema field is optional, and emitting ``"reasoning":
    ""`` is semantic garbage that still passes the type check)."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise DaftConvertError(
            f"{kind} item[{idx}].reasoning must be a string, got {type(value).__name__}"
        )
    s = value.strip()
    return s if s else None


__all__ = [
    "to_daft_bcq_openended",
    "to_daft_mcq_openended",
    "to_daft_open_qa",
]
