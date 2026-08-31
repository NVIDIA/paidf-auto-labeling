# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM grounding pass that produces structured input for ``to_daft_causal_linkage``.

Pipeline-side glue between PL's events catalogue / per-window VLM
captions and the :func:`reasoning.causal_linkage.converter.to_daft_causal_linkage`
converter. The LLM is asked, for each (t1, t2) pair, to explain the
causal relationship between the event at t1 and the situation at t2.

Two pair-source modes (selected by the ``mode`` argument):

- ``"auto_from_events"``: derive (t1, t2) pairs from a parsed
  ``contextual/events.json`` payload. The default pairing strategy is
  "consecutive pairs": each event's ``start_time`` is paired with the
  next event's ``start_time`` (or with the scene-end timecode for the
  last event). Capped at ``max_pairs`` to avoid runaway LLM cost on
  events-dense clips.
- ``"explicit"``: caller supplies a list of pair dicts (inline or via
  YAML/JSON file) shaped as ``{"t1": ..., "t2": ..., "question"?:
  ..., "video_type"?: ...}``.

A single LLM call grounds every pair against the per-window captions.
The adapter returns the parsed item list; the converter is the source
of truth for shape validation and DAFT envelope construction.

Use-case agnosticism: no domain vocabulary, no hard-coded video_type
inference, no question template baked into Python. The default
question template ("Explain the relationship between the event at
{t1} and the situation at {t2}.") is taken verbatim from the schema's
own example and is therefore use-case neutral; callers who want
different phrasing override the question on a per-pair basis in
``explicit`` mode, or ship a domain-specific prompt variant.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import yaml

from reasoning._llm_context import (
    block as _block,
)
from reasoning._llm_context import (
    format_windows_block as _format_windows_block,
)
from reasoning.llm_client import (
    call_chat_object_with_structured_fallback,
    call_required_json_object_with_fallback,
    get_llm_api_key,
    require_items_list,
)
from reasoning.prompts import PromptVariant
from reasoning.timecodes import seconds_to_timecode, timecode_to_seconds

CausalMode = Literal["auto_from_events", "explicit"]

# Default natural-language template used when a pair has no
# caller-supplied question. Lifted from the schema's own example to
# stay domain-neutral. ``{t1}`` / ``{t2}`` are formatted in by the
# adapter just before the LLM call.
DEFAULT_QUESTION_TEMPLATE: str = (
    "Explain the relationship between the event at {t1} and the situation at {t2}."
)


class CausalLinkageLLMError(RuntimeError):
    """Raised when the causal-linkage LLM call fails to return usable items.

    Distinct from :class:`reasoning.common.DaftConvertError`: this
    fires for transport / parse / shape failures *before* the result
    reaches the converter. Pipeline-level callers swallow this and
    skip writing the file (best-effort)."""


# ---------------------------------------------------------------------------
# Pair sourcing
# ---------------------------------------------------------------------------


def derive_pairs_from_events(
    events: Iterable[Any],
    *,
    duration: float | None,
    max_pairs: int = 8,
    question_template: str = DEFAULT_QUESTION_TEMPLATE,
    default_video_type: str | None = None,
) -> list[dict[str, Any]]:
    """Derive (t1, t2) pairs from a parsed ``events.json`` events list.

    Pairing strategy: each event's ``start_time`` is paired with the
    next event's ``start_time``. The last event is paired with
    ``duration`` (when supplied) or its own ``end_time`` (when
    available) as a fallback. Events without a parseable ``start_time``
    are skipped.

    Per-pair ``question`` is rendered from ``question_template`` with
    ``{t1}`` / ``{t2}`` substituted from the timecodes; callers can
    override the template to use a domain-specific phrasing.

    ``default_video_type``, when set to ``"anomaly"`` or ``"normal"``,
    is attached to every derived pair so the LLM can preserve it in
    its answer items (the converter will still validate the enum).

    Returns the (possibly empty) pair list, capped at ``max_pairs``.
    """
    # Keep ``end_time`` alongside ``start_time`` so the trailing event
    # can fall back to its own end (per the docstring contract) when
    # neither a successor event nor a clip duration is available — that
    # was the dropped-final-pair regression.
    parsed: list[tuple[float, float | None]] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        st = _safe_seconds(ev.get("start_time"))
        if st is None:
            st = _safe_seconds(ev.get("start"))
        if st is None:
            continue
        et = _safe_seconds(ev.get("end_time"))
        if et is None:
            et = _safe_seconds(ev.get("end"))
        parsed.append((st, et))

    parsed.sort(key=lambda x: x[0])

    pairs: list[dict[str, Any]] = []
    n = len(parsed)
    for i, (t1_secs, t1_end_secs) in enumerate(parsed):
        if i + 1 < n:
            t2_secs = parsed[i + 1][0]
        elif duration is not None:
            t2_secs = float(duration)
        elif t1_end_secs is not None and t1_end_secs > t1_secs:
            # Trailing event's own ``end_time`` is the documented
            # fallback when no clip duration anchor is available.
            t2_secs = float(t1_end_secs)
        else:
            # No next event, no duration, no usable end_time -> drop
            # the trailing pair rather than ship a degenerate (t1, t1).
            continue
        # Skip degenerate pairs where t2 collapses onto t1 (within a
        # frame of one another). Avoids LLM confusion on identical
        # cause/effect timestamps.
        if t2_secs - t1_secs < 0.001:
            continue
        t1_tc = seconds_to_timecode(t1_secs)
        t2_tc = seconds_to_timecode(t2_secs)
        pair: dict[str, Any] = {
            "t1": t1_tc,
            "t2": t2_tc,
            "question": question_template.format(t1=t1_tc, t2=t2_tc),
        }
        if default_video_type is not None:
            pair["video_type"] = default_video_type
        pairs.append(pair)
        if len(pairs) >= max_pairs:
            break
    return pairs


def load_pair_bank(
    *,
    pairs: list[Any] | None,
    pair_file: Path | str | None,
    question_template: str = DEFAULT_QUESTION_TEMPLATE,
) -> list[dict[str, Any]]:
    """Resolve the per-scene causal-pair bank from inline config + optional file.

    Supported file shapes (any one):

    1. Bare list of pair dicts: ``[{"t1":..., "t2":..., ...}, ...]``.
    2. ``{"pairs": [...]}``.
    3. ``{"items": [...]}``.

    Each pair dict is normalized to carry at least ``t1``, ``t2``, and
    ``question`` (rendered from ``question_template`` when omitted).
    Optional keys ``video_type`` and any other caller-supplied keys
    are passed through verbatim. Pairs missing ``t1`` or ``t2`` are
    silently dropped (the converter would reject them anyway, and
    silently dropping keeps pipeline-side warnings focused on the LLM
    call rather than the bank loader).

    Returns the merged, deduplicated bank (dedupe key is
    ``(t1, t2, question)``). Returns ``[]`` when neither source
    yields a usable pair.
    """
    out: list[dict[str, Any]] = []

    if pairs:
        for entry in pairs:
            normalized = _normalize_pair(entry, template=question_template)
            if normalized is not None:
                out.append(normalized)

    if pair_file is not None:
        path = Path(pair_file)
        if not path.is_file():
            raise FileNotFoundError(f"causal_linkage pair_file not found: {path}")
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in (".yaml", ".yml"):
            obj = yaml.safe_load(text)
        else:
            obj = json.loads(text)
        for entry in _extract_pairs_from_obj(obj, source=str(path)):
            normalized = _normalize_pair(entry, template=question_template)
            if normalized is not None:
                out.append(normalized)

    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for p in out:
        key = (str(p.get("t1")), str(p.get("t2")), str(p.get("question")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    return deduped


def _normalize_pair(entry: Any, *, template: str) -> dict[str, Any] | None:
    """Coerce a pair entry into ``{"t1", "t2", "question", ...}``."""
    if not isinstance(entry, dict):
        return None
    t1 = entry.get("t1")
    t2 = entry.get("t2")
    if t1 is None or t2 is None:
        return None
    # Normalize numeric inputs to canonical timecodes here so the
    # rendered question template (which refers to ``{t1}`` / ``{t2}``)
    # uses the same string the converter will see. Tolerant of bad
    # input — drop instead of raising; the LLM call will surface a
    # clearer error if the bank is wholly empty.
    t1_tc = _safe_normalize_timecode(t1)
    t2_tc = _safe_normalize_timecode(t2)
    if t1_tc is None or t2_tc is None:
        return None

    out: dict[str, Any] = dict(entry)
    out["t1"] = t1_tc
    out["t2"] = t2_tc
    q = entry.get("question")
    if not isinstance(q, str) or not q.strip():
        out["question"] = template.format(t1=t1_tc, t2=t2_tc)
    else:
        out["question"] = q.strip()
    return out


def _extract_pairs_from_obj(obj: Any, *, source: str) -> list[Any]:
    if isinstance(obj, list):
        return list(obj)
    if isinstance(obj, dict):
        for key in ("pairs", "items"):
            if isinstance(obj.get(key), list):
                return list(obj[key])
    raise ValueError(
        f"causal_linkage pair file {source!r} must be a list or an object "
        "with a 'pairs' or 'items' list"
    )


def _safe_seconds(value: Any) -> float | None:
    """Best-effort numeric / timecode -> seconds. Returns ``None`` on failure."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        if value < 0:
            return None
        return float(value)
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return None
        try:
            return timecode_to_seconds(v)
        except ValueError:
            return None
    return None


def _safe_normalize_timecode(value: Any) -> str | None:
    secs = _safe_seconds(value)
    if secs is None:
        return None
    return seconds_to_timecode(secs)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def generate_causal_linkages_with_llm(
    *,
    pairs: list[dict[str, Any]],
    windows: Iterable[dict[str, Any]],
    scene_description: str | None,
    duration: float | None,
    prompt: PromptVariant,
    llm_url: str,
    llm_model: str,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    top_p: float = 1.0,
    timeout: int = 600,
    structured_output: str = "auto",
    seed: int | None = None,
    retries: int = 2,
    retry_backoff_s: float = 5.0,
    api_key: str | None = None,
    logger: logging.Logger | None = None,
    description_keys: tuple[str, ...] = (
        "enhanced_caption",
        "description",
        "caption",
        "summary",
    ),
) -> list[dict[str, Any]]:
    """Run one LLM call to ground every pair and return parsed items.

    Each returned item is the LLM's raw structured output (not yet a
    DAFT payload). Hand the list to
    :func:`reasoning.causal_linkage.converter.to_daft_causal_linkage` to
    validate and emit a DAFT-compliant payload.

    The adapter re-attaches the input ``video_type`` (when the caller
    supplied one in the pair bank but the LLM elided it from its
    response) so the on-disk file preserves caller intent.

    Raises :class:`CausalLinkageLLMError` for empty pairs, no usable
    windows, empty / unparseable LLM response, or ``items`` not a
    non-empty list of dicts. The caller should swallow these and skip
    writing the file."""
    log = logger or logging.getLogger(__name__)

    if not pairs:
        raise CausalLinkageLLMError("no usable causal pairs supplied; skipping causal_linkage")

    win_lines = _format_windows_block(
        windows, description_keys=description_keys, log=log, tag="causal_linkage"
    )
    if not win_lines:
        raise CausalLinkageLLMError(
            "no usable windows for causal_linkage LLM call (each window "
            "must have start, end, and a caption-like field)"
        )

    pairs_block = _format_pairs_block(pairs)

    user_text = prompt.render_user(
        windows_block="\n".join(win_lines),
        scene_description_block=_block(
            "Scene description:",
            scene_description,
            fallback="(no scene-level description provided)",
        ),
        pairs_block=pairs_block,
    )

    messages = [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": user_text},
    ]

    parsed = call_required_json_object_with_fallback(
        base_url=llm_url,
        model=llm_model,
        messages=messages,
        timeout=int(timeout),
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        logger=log,
        retries=int(retries),
        retry_backoff_s=float(retry_backoff_s),
        structured_output=str(structured_output),
        seed=seed,
        guided_json_schema=_causal_linkage_guided_schema(),
        retry_stage="causal_linkage",
        api_key=api_key or get_llm_api_key(),
        error_type=CausalLinkageLLMError,
        no_parseable_message="causal_linkage LLM returned no parseable JSON. Raw: {snippet_repr}",
        non_object_message="causal_linkage LLM returned a non-object payload: {type_name}",
        caller=call_chat_object_with_structured_fallback,
    )
    items = require_items_list(
        parsed,
        error_type=CausalLinkageLLMError,
        missing_message="causal_linkage LLM payload missing 'items' list, got {type_name}",
        empty_message="causal_linkage LLM returned an empty 'items' list",
    )

    # Re-attach video_type from the bank when the LLM elided it. Index
    # by (t1, t2) since the prompt asks the model to echo both.
    bank_by_pair = {(p["t1"], p["t2"]): p for p in pairs}
    out_items: list[dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            log.warning("[causal_linkage] dropping non-dict item from LLM payload: %r", it)
            continue
        merged = dict(it)
        key = (merged.get("t1"), merged.get("t2"))
        bank_entry = bank_by_pair.get(key)
        if bank_entry is not None:
            if "video_type" not in merged and "video_type" in bank_entry:
                merged["video_type"] = bank_entry["video_type"]
        out_items.append(merged)

    if not out_items:
        raise CausalLinkageLLMError("causal_linkage LLM returned 'items' but none were dicts")

    # Pin every item's duration awareness for the converter — the
    # converter accepts a top-level ``duration`` to clamp timecodes.
    # Nothing to attach to the items themselves; just return.
    del duration  # silence linters; duration flows through the converter, not the LLM
    return out_items


# ---------------------------------------------------------------------------
# Prompt-block formatting
# ---------------------------------------------------------------------------


def _format_pairs_block(pairs: list[dict[str, Any]]) -> str:
    """Render the per-pair prompt block."""
    lines: list[str] = []
    for i, p in enumerate(pairs, start=1):
        t1 = p.get("t1", "?")
        t2 = p.get("t2", "?")
        q = p.get("question", "")
        vt = p.get("video_type")
        suffix = f"  (video_type={vt})" if isinstance(vt, str) else ""
        lines.append(f"{i}. t1={t1}, t2={t2}{suffix}")
        lines.append(f"   question: {q}")
    return "\n".join(lines)


def _causal_linkage_guided_schema() -> dict[str, Any]:
    """Guided-JSON schema mirroring the per-pair item shape the LLM emits.

    NIM endpoints respect this hint; OpenAI-style endpoints ignore it
    and rely on response_format=json_object instead. The converter is
    the source of truth for shape validation."""
    return {
        "type": "object",
        "required": ["items"],
        "properties": {
            "items": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["t1", "t2", "question", "answer"],
                    "properties": {
                        "t1": {"type": "string"},
                        "t2": {"type": "string"},
                        "question": {"type": "string"},
                        "answer": {"type": "string"},
                        "reasoning": {"type": "string"},
                        "video_type": {
                            "type": "string",
                            "enum": ["anomaly", "normal"],
                        },
                    },
                },
            },
        },
    }


__all__ = [
    "CausalLinkageLLMError",
    "CausalMode",
    "DEFAULT_QUESTION_TEMPLATE",
    "derive_pairs_from_events",
    "generate_causal_linkages_with_llm",
    "load_pair_bank",
]
