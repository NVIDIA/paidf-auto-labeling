# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PL task items -> DAFT task payloads.

Splits PL's internal task list into mutually exclusive DAFT task types:

- ``mcq``: non-binary closed-choice items (items with a non-empty
  ``options`` list whose options do not reduce to ``{"Yes","No"}``) are
  emitted in MCQ form. Options become a letter-keyed dict
  (``{"A": ..., "B": ...}``) and the answer is reduced to a single letter
  (``^[A-Za-z]$``).
- ``bcq``: items whose options reduce to the binary set
  ``{"Yes","No"}`` are emitted in BCQ form only. The BCQ schema carries
  no ``options`` field and pins ``answer`` to the bare enum
  ``{"Yes","No"}``.
- ``open_qa``: missing or empty options with a non-empty free-form answer
  routes here exclusively (no MCQ/BCQ counterpart).

The exclusive routing keeps one DAFT task representation per source
question. That preserves stable question identity for downstream
comparison/evaluation tools while still using the DAFT schema that best
matches each task's semantics.

Input items use the normalized PL task shape::

    {"id": "1_1", "question": "...", "options": [str, ...], "answer": str}

``options`` is a list of strings in PL's bank-driven runners. For closed
choices, ``answer`` equals one of those strings (enforced upstream by
``filter_mcq_items_strict``). Missing or empty ``options`` means the item
is free-form; a non-empty answer is exported as DAFT ``open_qa``.
Closed-choice duplicate options are removed while preserving first-seen
order; MCQ duplicates are compared after stripping any
``"A. "``/``"A)"``-style prefix; BCQ recognition matches binary yes/no
semantics regardless of letter-prefix scaffolding (see
:func:`_normalize_bcq_answer`).

DAFT v3 requires every task item to carry ``video_id`` *or* ``image_id``
(``oneOf`` constraint in mcq/bcq/open_qa schemas); the right field is chosen
from ``ctx.is_image``. When the VLM-as-judge step (``vlm_verify``) has
attached a per-item ``reasoning_trace`` upstream, it is passed through to
DAFT's optional ``reasoning`` field on every emitted view of the item
(MCQ for non-binary items, BCQ for binary yes/no items, or open_qa for
free-form items).
"""

from __future__ import annotations

from typing import Any

from core.formats.daft.envelope import daft_envelope
from core.formats.daft.errors import DaftConvertError
from core.formats.daft.text import LETTER_ALPHABET, match_letter_prefix, strip_letter_prefix
from core.scene import SceneContext

_YES_NO: frozenset[str] = frozenset({"Yes", "No"})


def _canonical_yes_no(stripped: str) -> str | None:
    """Map a prefix-stripped option/answer string to canonical 'Yes'/'No'.

    DAFT v3.0 BCQ requires the bare canonical case (``answer ∈ {"Yes","No"}``);
    bank authors may write the surface form in any case. ``None`` signals
    "this string isn't a yes/no token", which the BCQ predicate uses to
    short-circuit on non-binary options.
    """
    s = stripped.strip()
    if s.lower() == "yes":
        return "Yes"
    if s.lower() == "no":
        return "No"
    return None


def _normalize_bcq_answer(options: Any, answer: Any) -> str | None:
    """Return canonical 'Yes' / 'No' iff this item is BCQ-shaped, else ``None``.

    BCQ in DAFT v3.0 is defined by **task semantics** (binary event
    verification), not by surface scaffolding. The on-disk BCQ schema
    has no ``options`` field at all and pins ``answer`` to the bare
    enum ``{"Yes","No"}``, so the converter's job is to recognize a
    binary yes/no item regardless of how the bank author chose to
    express it and emit the canonical bare answer.

    Recognized authoring styles (bank-side):

    * Bare BCQ form: ``options=["Yes","No"]`` — historical strict shape.
    * Letter-MCQ form: ``options=["A. Yes","B. No"]`` (or ``"B) No"``,
      ``"C: yes"``, etc. — any prefix accepted by
      :func:`reasoning.text.match_letter_prefix`).
    * Reversed order: ``["A. No","B. Yes"]`` — the letter prefix, not
      the position, drives the letter→value mapping for letter answers.
    * Mixed case: ``["A. yes","B. no"]`` — case-normalized to ``"Yes"``/``"No"``.

    Recognized answer styles (LLM-side):

    * Bare canonical: ``"Yes"`` / ``"No"`` (any case).
    * Bare letter: ``"A"`` / ``"B"`` (matched against the bank's letter prefixes).
    * Full bank string: ``"A. Yes"`` / ``"B. No"`` (letter prefix stripped, then matched).

    Returns ``None`` (and the caller routes to MCQ instead) for:

    * Any item whose option list, after dedupe + prefix strip + case-fold,
      isn't exactly the set ``{"Yes","No"}`` (e.g. ``["Day","Night"]``,
      ``["Yes","No","Maybe"]``, single-option lists).
    * Any answer that doesn't resolve to one of the two yes/no values.
    * Non-list options or non-string answers.

    The helper deliberately does *not* parse the question text — task
    routing is data-driven from the bank's structured fields, not from
    NLU on the question prose. That keeps routing deterministic and
    auditable.
    """
    if not isinstance(options, list) or not isinstance(answer, str):
        return None

    # First pass over options: strip prefixes, canonicalize values, dedupe.
    # Keep the letter→canonical map so letter answers can be resolved later
    # without a second pass.
    seen_canonical: set[str] = set()
    canonical_options: list[str] = []
    letter_to_canonical: dict[str, str] = {}
    for opt in options:
        raw = str(opt).strip()
        if not raw:
            continue
        m = match_letter_prefix(raw)
        if m is not None:
            letter = m.group(1).upper()
            stripped = m.group(2)
        else:
            letter = None
            stripped = raw
        canonical = _canonical_yes_no(stripped)
        if canonical is None:
            return None  # any non-yes/no option means this isn't BCQ
        if canonical in seen_canonical:
            continue
        seen_canonical.add(canonical)
        canonical_options.append(canonical)
        if letter is not None:
            letter_to_canonical[letter] = canonical

    if len(canonical_options) != 2 or frozenset(canonical_options) != _YES_NO:
        return None

    ans = answer.strip()
    if not ans:
        return None

    # Try as a single letter against the bank's letter prefixes.
    if len(ans) == 1 and ans.isalpha():
        canonical = letter_to_canonical.get(ans.upper())
        if canonical is not None:
            return canonical
        return None  # explicit letter that doesn't match any option

    # Try as bare canonical (any case).
    bare = _canonical_yes_no(ans)
    if bare is not None:
        return bare

    # Try as a full bank string ("A. Yes" or "B) No").
    m = match_letter_prefix(ans)
    if m is not None:
        stripped = m.group(2)
        canonical = _canonical_yes_no(stripped)
        if canonical is not None:
            return canonical

    return None


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
    """Predicate: should this closed-choice item be emitted as BCQ?

    Returns ``True`` when the item's options reduce to the binary set
    ``{"Yes","No"}`` (regardless of letter-prefix scaffolding) and the
    answer resolves to one of those two values. The DAFT v3.0 BCQ schema
    is defined by ``answer ∈ {"Yes","No"}`` and carries no options
    field, so the surface form on the bank side is incidental.

    Delegates to :func:`_normalize_bcq_answer` so authoring scaffolding
    (bare ``["Yes","No"]`` vs letter-prefixed ``["A. Yes","B. No"]``
    etc.) doesn't change recognition.
    """
    return _normalize_bcq_answer(options, answer) is not None


def _is_open_qa(options: Any, answer: Any) -> bool:
    return (
        options is None or (isinstance(options, list) and len(_dedupe_options(options)) == 0)
    ) and bool(str(answer).strip())


def _normalize_mcq_options(options: Any, qid: str) -> tuple[dict[str, str], dict[str, str]]:
    """Coerce PL options into DAFT letter-keyed options plus answer aliases.

    MCQ output stores the display value after stripping any author-supplied
    letter prefix. Deduping by that display value avoids emitting two choices
    with the same DAFT text. Answer aliases are registered only for the options
    that survive dedupe, matching the normalized question bank seen by models.

    Empty and whitespace-only options are dropped (consistent with
    :func:`_dedupe_options` and :func:`_normalize_bcq_answer`); leading
    and trailing whitespace on real options is stripped before letter-
    prefix detection so that authoring noise like ``" Yes "`` doesn't
    create a separate display value from ``"Yes"``.
    """
    if not isinstance(options, list):
        raise DaftConvertError(
            f"MCQ item {qid!r} 'options' must be a list, got {type(options).__name__}"
        )

    opts_dict: dict[str, str] = {}
    aliases: dict[str, str] = {}
    value_to_letter: dict[str, str] = {}
    for option in options:
        raw = str(option).strip()
        if not raw:
            continue
        value = strip_letter_prefix(raw).strip()
        if not value:
            continue
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
            aliases[raw] = letter
            aliases[value] = letter
            m = match_letter_prefix(raw)
            if m:
                aliases[m.group(1)] = letter

    if not opts_dict:
        raise DaftConvertError(
            f"MCQ item {qid!r} has no options; route empty-option items to open_qa"
        )
    if len(opts_dict) < 2:
        raise DaftConvertError(f"MCQ item {qid!r} has <2 unique options; DAFT requires minItems 2")
    return opts_dict, aliases


def _resolve_mcq_answer(
    answer: Any, opts_dict: dict[str, str], aliases: dict[str, str], qid: str
) -> str:
    """Return the single DAFT letter for ``answer`` against the normalized
    options dict. Raises ``DaftConvertError`` if it can't be resolved — which
    should never happen in practice because upstream filtering guarantees
    ``answer in options``.

    Single-letter answers fall back to a case-swapped lookup so that a
    lowercase LLM answer (``"a"``) matches an uppercase bank letter
    (``"A"``) and vice versa. The fallback only triggers when the
    swapped letter is *unambiguously* the only valid interpretation —
    that is, when the original case is not present in ``opts_dict`` —
    which preserves correct routing for items with >26 options where
    ``LETTER_ALPHABET`` wraps into lowercase.
    """
    ans_str = str(answer).strip()

    if ans_str in aliases:
        return aliases[ans_str]

    if len(ans_str) == 1 and ans_str in opts_dict:
        return ans_str

    if len(ans_str) == 1 and ans_str.isalpha():
        swapped = ans_str.swapcase()
        if swapped in opts_dict:
            return swapped

    raise DaftConvertError(
        f"MCQ item {qid!r} answer {answer!r} not in options "
        f"{list(opts_dict.values())!r}; upstream filtering should have caught this"
    )


def _to_daft_mcq_item(item: dict[str, Any], *, ctx: SceneContext) -> dict[str, Any]:
    qid = str(item.get("id", "?"))
    opts_dict, aliases = _normalize_mcq_options(item.get("options"), qid)
    letter = _resolve_mcq_answer(item.get("answer"), opts_dict, aliases, qid)
    out: dict[str, Any] = {
        ctx.scene_id_field: ctx.media_id,
        "question": item["question"],
        "answer": letter,
        "options": opts_dict,
    }
    reasoning = _extract_reasoning(item)
    if reasoning is not None:
        out["reasoning"] = reasoning
    return out


def _to_daft_bcq_item(item: dict[str, Any], *, ctx: SceneContext) -> dict[str, Any]:
    # Internal `id` stays in the sidecar; DAFT output doesn't carry it.
    # The DAFT BCQ schema pins ``answer`` to the bare enum ``{"Yes","No"}``
    # (and forbids an ``options`` field), so we re-run normalization here
    # rather than echo whatever shape the bank/LLM sent us — the bank may
    # be letter-prefixed ("A") and the schema validator will reject it
    # unless we translate to the bare canonical value.
    canonical_answer = _normalize_bcq_answer(item.get("options"), item["answer"])
    if canonical_answer is None:
        # Defensive: routing already filtered via :func:`_is_bcq`, so this
        # path is unreachable in practice; raising keeps the contract local
        # rather than letting a malformed item slip through to schema validation.
        raise DaftConvertError(
            f"BCQ item {item.get('id', '?')!r} not normalizable: "
            f"options={item.get('options')!r} answer={item.get('answer')!r}"
        )
    out: dict[str, Any] = {
        ctx.scene_id_field: ctx.media_id,
        "question": item["question"],
        "answer": canonical_answer,
    }
    reasoning = _extract_reasoning(item)
    if reasoning is not None:
        out["reasoning"] = reasoning
    return out


def _to_daft_open_qa_item(item: dict[str, Any], *, ctx: SceneContext) -> dict[str, Any]:
    # Internal `id` stays in sidecars; DAFT open_qa items carry only schema fields.
    # ``ctx.scene_id_field`` chooses ``video_id`` vs ``image_id`` to satisfy the
    # ``oneOf`` constraint on open_qa items (same shape as mcq/bcq).
    out: dict[str, Any] = {
        ctx.scene_id_field: ctx.media_id,
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
    ctx: SceneContext,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Split PL internal task items into DAFT mcq + bcq + open_qa payloads.

    Returns ``(mcq_payload, bcq_payload, open_qa_payload)``. Each side is
    ``None`` when there are no items of that type, so callers should skip
    writing that file (DAFT requires ``minItems: 1`` on ``items``, so empty
    stubs are illegal).

    **Routing contract (exclusive):**

    - Free-form items (``options`` empty/missing, non-empty ``answer``)
      route to ``open_qa`` and nowhere else.
    - Binary closed-choice items (options reduce to ``{"Yes","No"}``)
      populate ``bcq`` as a schema-canonical projection
      (no ``options`` field, ``answer ∈ {"Yes","No"}``).
    - Other closed-choice items (non-empty ``options``) populate ``mcq``.

    Reasoning traces are passed through to the emitted view of an item.

    Each emitted item carries ``video_id`` or ``image_id`` (chosen from
    ``ctx.is_image``) to satisfy DAFT v3's ``oneOf(video_id, image_id)``
    requirement on task items.

    Raises ``DaftConvertError`` on upstream-invariant violations: answer
    not in closed-choice options, too few options, or too many options.
    """
    mcq_items: list[dict[str, Any]] = []
    bcq_items: list[dict[str, Any]] = []
    open_qa_items: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            raise DaftConvertError(f"task item {item!r} is not a dict/mapping")
        qid = str(item.get("id", "?"))
        for f in _ITEM_REQUIRED:
            if f not in item:
                raise DaftConvertError(f"task item {qid!r} missing required field {f!r}")
        options = item.get("options")
        answer = item["answer"]
        if _is_open_qa(options, answer):
            open_qa_items.append(_to_daft_open_qa_item(item, ctx=ctx))
            continue
        if _is_bcq(options, answer):
            bcq_items.append(_to_daft_bcq_item(item, ctx=ctx))
            continue
        # Other closed-choice items are emitted in MCQ form. The MCQ
        # converter handles letter-prefixed and bare option lists
        # uniformly.
        mcq_items.append(_to_daft_mcq_item(item, ctx=ctx))

    mcq_payload = _wrap_task("mcq", mcq_items, ctx) if mcq_items else None
    bcq_payload = _wrap_task("bcq", bcq_items, ctx) if bcq_items else None
    open_qa_payload = _wrap_task("open_qa", open_qa_items, ctx) if open_qa_items else None
    return mcq_payload, bcq_payload, open_qa_payload


def _wrap_task(task_type: str, items: list[dict[str, Any]], ctx: SceneContext) -> dict[str, Any]:
    """Wrap items in the DAFT task envelope. Tasks have no top-level scene id
    (the scope is the file's metadata.type); the per-item ``video_id`` /
    ``image_id`` carries the cross-reference to ``contextual/``."""
    out = daft_envelope(task_type, ctx, include_scene_id=False)
    out["items"] = items
    metadata = out.pop("metadata")
    out["metadata"] = metadata
    return out


__all__ = ["to_daft_tasks"]
