# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auto-labeling task items -> DAFT task payloads.

Splits the auto-labeling-internal MCQ list into DAFT task types:

- ``mcq``: options become a letter-keyed dict (``{"A": ..., "B": ...}``) and
  the answer is reduced to a single letter (``^[A-Za-z]$``).
- ``bcq``: after exact duplicate removal, ``options`` is bare
  ``["Yes", "No"]`` or ``["No", "Yes"]`` and the answer is ``"Yes"`` or
  ``"No"``.
- ``open_qa``: missing or empty options with a non-empty free-form answer.

Input items use the normalized auto-labeling task shape::

    {"id": "1_1", "question": "...", "options": [str, ...], "answer": str}

``options`` is a list of strings in the auto-labeling bank-driven runners. For closed
choices, ``answer`` equals one of those strings (enforced upstream by
``filter_mcq_items_strict``). Missing or empty ``options`` means the item is
free-form; a non-empty answer is exported as DAFT ``open_qa``. Closed-choice
duplicate options are removed while preserving first-seen order; MCQ duplicates
are compared after stripping any ``"A. "``/``"A)"``-style prefix, while BCQ
routing compares only bare option strings so prefixed Yes/No choices remain MCQ.

When the VLM-as-judge step (``vlm_verify``) has attached a per-item
``reasoning_trace`` upstream, it is passed through to DAFT's optional
``reasoning`` field.
"""

from __future__ import annotations

from typing import Any

from al_utils.text import LETTER_ALPHABET, match_letter_prefix, strip_letter_prefix
from daft_export.common import DAFT_VERSION, DaftConvertError, metadata_block

_YES_NO: frozenset[str] = frozenset({"Yes", "No"})

_ITEM_REQUIRED: tuple[str, ...] = ("question", "answer")


def _extract_reasoning(item: dict[str, Any]) -> str | None:
    """Return the item's reasoning trace for DAFT's optional ``reasoning`` field.

    Prefers ``reasoning_trace`` (what ``vlm_verify`` attaches) over
    ``reasoning`` (the on-disk DAFT field name).
    Returns ``None`` when neither is present or the value is empty after strip,
    so the key is omitted from the DAFT item rather than emitted as ``""``.
    """
    for key in ("reasoning_trace", "reasoning"):
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _dedupe_options(options: list[Any]) -> list[str]:
    """Remove empty and exact duplicate option strings while preserving first-seen order."""
    out: list[str] = []
    seen: set[str] = set()
    for option in options:
        text = str(option).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _is_bcq(options: Any, answer: Any) -> bool:
    """Route an item to BCQ only when the bank declares a Yes/No choice."""
    if answer not in _YES_NO:
        return False
    if isinstance(options, list):
        deduped = _dedupe_options(options)
        return len(deduped) == 2 and set(deduped) == _YES_NO
    return False


def _is_open_qa(options: Any, answer: Any) -> bool:
    return (options is None or (isinstance(options, list) and len(_dedupe_options(options)) == 0)) and bool(
        str(answer).strip()
    )


def _normalize_mcq_options(options: Any, qid: str) -> tuple[dict[str, str], dict[str, str]]:
    """Coerce auto-labeling options into DAFT letter-keyed options plus answer aliases.

    MCQ output stores the display value after stripping any author-supplied
    letter prefix. Deduping by that display value avoids emitting two choices
    with the same DAFT text. Answer aliases are registered only for the options
    that survive dedupe, matching the normalized question bank seen by models.
    """
    if not isinstance(options, list):
        raise DaftConvertError(f"MCQ item {qid!r} 'options' must be a list, got {type(options).__name__}")

    opts_dict: dict[str, str] = {}
    aliases: dict[str, str] = {}
    value_to_letter: dict[str, str] = {}
    for option in options:
        raw = str(option)
        value = strip_letter_prefix(raw)
        letter = value_to_letter.get(value)
        if letter is None:
            if len(opts_dict) >= len(LETTER_ALPHABET):
                raise DaftConvertError(
                    f"MCQ item {qid!r} has more than {len(LETTER_ALPHABET)} unique options; "
                    f"DAFT's answer regex caps this at {len(LETTER_ALPHABET)} (A-Z a-z)"
                )
            letter = LETTER_ALPHABET[len(opts_dict)]
            value_to_letter[value] = letter
            opts_dict[letter] = value
            aliases[raw.strip()] = letter
            aliases[value] = letter
            m = match_letter_prefix(raw)
            if m:
                aliases[m.group(1)] = letter

    if not opts_dict:
        raise DaftConvertError(f"MCQ item {qid!r} has no options; route empty-option items to open_qa")
    if len(opts_dict) < 2:
        raise DaftConvertError(f"MCQ item {qid!r} has <2 unique options; DAFT requires minItems 2")
    return opts_dict, aliases


def _resolve_mcq_answer(answer: Any, opts_dict: dict[str, str], aliases: dict[str, str], qid: str) -> str:
    """Return the single DAFT letter for ``answer`` against the normalized
    options dict. Raises ``DaftConvertError`` if it can't be resolved — which
    should never happen in practice because upstream filtering guarantees
    ``answer in options``."""
    ans_str = str(answer).strip()

    if ans_str in aliases:
        return aliases[ans_str]

    if len(ans_str) == 1 and ans_str in opts_dict:
        return ans_str

    raise DaftConvertError(
        f"MCQ item {qid!r} answer {answer!r} not in options "
        f"{list(opts_dict.values())!r}; upstream filtering should have caught this"
    )


def _to_daft_mcq_item(item: dict[str, Any], *, video_id: str) -> dict[str, Any]:
    qid = str(item.get("id", "?"))
    opts_dict, aliases = _normalize_mcq_options(item.get("options"), qid)
    letter = _resolve_mcq_answer(item.get("answer"), opts_dict, aliases, qid)
    out: dict[str, Any] = {
        "video_id": video_id,
        "question": item["question"],
        "answer": letter,
        "options": opts_dict,
    }
    reasoning = _extract_reasoning(item)
    if reasoning is not None:
        out["reasoning"] = reasoning
    return out


def _to_daft_bcq_item(item: dict[str, Any], *, video_id: str) -> dict[str, Any]:
    # Internal `id` stays in the sidecar; DAFT output doesn't carry it.
    out: dict[str, Any] = {
        "video_id": video_id,
        "question": item["question"],
        "answer": item["answer"],
    }
    reasoning = _extract_reasoning(item)
    if reasoning is not None:
        out["reasoning"] = reasoning
    return out


def _to_daft_open_qa_item(item: dict[str, Any], *, video_id: str) -> dict[str, Any]:
    # Internal `id` stays in sidecars; DAFT open_qa items carry only schema fields.
    out: dict[str, Any] = {
        "video_id": video_id,
        "question": item["question"],
        "answer": str(item["answer"]).strip(),
    }
    reasoning = _extract_reasoning(item)
    if reasoning is not None:
        out["reasoning"] = reasoning
    return out


def to_daft_tasks(
    items: list[dict[str, Any]],
    *,
    video_id: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Split auto-labeling-internal task items into DAFT mcq + bcq + open_qa payloads.

    Returns ``(mcq_payload, bcq_payload, open_qa_payload)``. Each side is
    ``None`` when there are no items of that type, so callers should skip writing
    that file
    (DAFT requires ``minItems: 1`` on ``items``, so empty stubs are illegal).

    Raises ``DaftConvertError`` on upstream-invariant violations: answer not
    in closed-choice options, too few options, or too many options.
    """
    mcq_items: list[dict[str, Any]] = []
    bcq_items: list[dict[str, Any]] = []
    open_qa_items: list[dict[str, Any]] = []
    for item in items:
        qid = str(item.get("id", "?"))
        for f in _ITEM_REQUIRED:
            if f not in item:
                raise DaftConvertError(f"task item {qid!r} missing required field {f!r}")
        if _is_bcq(item.get("options"), item["answer"]):
            bcq_items.append(_to_daft_bcq_item(item, video_id=video_id))
        elif _is_open_qa(item.get("options"), item["answer"]):
            open_qa_items.append(_to_daft_open_qa_item(item, video_id=video_id))
        else:
            mcq_items.append(_to_daft_mcq_item(item, video_id=video_id))

    mcq_payload = (
        {
            "version": DAFT_VERSION,
            "metadata": metadata_block("mcq"),
            "items": mcq_items,
        }
        if mcq_items
        else None
    )
    bcq_payload = (
        {
            "version": DAFT_VERSION,
            "metadata": metadata_block("bcq"),
            "items": bcq_items,
        }
        if bcq_items
        else None
    )
    open_qa_payload = (
        {
            "version": DAFT_VERSION,
            "metadata": metadata_block("open_qa"),
            "items": open_qa_items,
        }
        if open_qa_items
        else None
    )
    return mcq_payload, bcq_payload, open_qa_payload


__all__ = ["to_daft_tasks"]
