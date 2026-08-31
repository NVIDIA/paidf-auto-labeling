# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM grounding pass that produces structured input for ``to_daft_temporal_localization``.

Pipeline-side glue: feeds per-window VLM captions plus a list of
natural-language event queries to the LLM, asks it to return one
``{question, answer: {start, end}, reasoning}`` object per query, and
returns the parsed list. The pure converter
:func:`reasoning.temporal_localization.converter.to_daft_temporal_localization`
turns that list into a DAFT payload.

Use-case agnosticism (the explicit user constraint):

- The query bank is **entirely the caller's data**. This module never
  ships a default question list; an empty / missing query bank means
  the stage is silently skipped. Domain specialization happens at the
  config layer (``queries:`` inline list, ``query_file:`` path, or
  reusing the MCQ ``question_bank_file``) — never here.
- The prompt comes from a :class:`PromptVariant` the caller picks
  from the registry. The bundled ``temporal_localization_default`` is
  generic; ship a YAML drop-in for domain-specific phrasing.
- The LLM endpoint comes from the caller's :class:`EndpointResolver`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

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


class TemporalLocalizationLLMError(RuntimeError):
    """Raised when the temporal-localization LLM call fails to return
    usable structured items.

    Distinct from :class:`reasoning.common.DaftConvertError`: this
    fires for transport / parse / shape failures *before* the result
    reaches the converter. Pipeline-level callers swallow this and
    skip writing the file."""


def generate_temporal_localizations_with_llm(
    *,
    queries: list[str],
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
    """Run one LLM call to ground every query and return the parsed items.

    Each returned item is the LLM's raw structured output (not yet a
    DAFT payload). Hand the list to
    :func:`reasoning.temporal_localization.converter.to_daft_temporal_localization`
    to validate and emit a DAFT-compliant payload.

    The default ``t1``/``t2`` per item are filled in by the converter
    (``t1=0``, ``t2=duration``) when the LLM omits them; we don't ask
    the LLM for the search-window bounds because they're constant per
    scene.

    Raises :class:`TemporalLocalizationLLMError` for empty queries
    list, no usable windows, empty / unparseable LLM response, or a
    non-list ``items`` field. The caller should swallow these and
    skip writing the file."""
    log = logger or logging.getLogger(__name__)

    cleaned_queries = _clean_queries(queries)
    if not cleaned_queries:
        raise TemporalLocalizationLLMError(
            "no usable queries (after stripping); skipping temporal_localization"
        )

    win_lines = _format_windows_block(
        windows, description_keys=description_keys, log=log, tag="temporal_localization"
    )
    if not win_lines:
        raise TemporalLocalizationLLMError(
            "no usable windows for temporal_localization LLM call (each window "
            "must have start, end, and a caption-like field)"
        )

    user_text = prompt.render_user(
        windows_block="\n".join(win_lines),
        scene_description_block=_block(
            "Scene description:",
            scene_description,
            fallback="(no scene-level description provided)",
        ),
        queries_block="\n".join(f"- {q}" for q in cleaned_queries),
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
        guided_json_schema=_temporal_localization_guided_schema(),
        retry_stage="temporal_localization",
        api_key=api_key or get_llm_api_key(),
        error_type=TemporalLocalizationLLMError,
        no_parseable_message=(
            "temporal_localization LLM returned no parseable JSON. Raw: {snippet_repr}"
        ),
        non_object_message=("temporal_localization LLM returned a non-object payload: {type_name}"),
        caller=call_chat_object_with_structured_fallback,
    )
    items = require_items_list(
        parsed,
        error_type=TemporalLocalizationLLMError,
        missing_message=("temporal_localization LLM payload missing 'items' list, got {type_name}"),
        empty_message="temporal_localization LLM returned an empty 'items' list",
    )

    # Default in t1/t2 from duration when the LLM omitted them; the
    # converter will accept the same defaults (or raise if neither the
    # LLM nor we can supply t2). Doing it here keeps the converter
    # decoupled from "duration came from video.json" assumptions.
    out_items: list[dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            log.warning("[temporal_localization] dropping non-dict item from LLM payload: %r", it)
            continue
        merged = dict(it)
        if "t1" not in merged:
            merged["t1"] = 0.0
        if "t2" not in merged and duration is not None:
            merged["t2"] = float(duration)
        out_items.append(merged)

    if not out_items:
        raise TemporalLocalizationLLMError(
            "temporal_localization LLM returned 'items' but none were dicts"
        )

    return out_items


def load_query_bank(
    *,
    queries: list[str] | None,
    query_file: Path | None,
    section: str | None = None,
) -> list[str]:
    """Resolve the per-scene query list from inline config + optional file.

    Accepts:
    - ``queries``: inline list from the pipeline config (``reasoning.
      temporal_localization.queries``).
    - ``query_file``: absolute or pre-resolved path to a YAML/JSON file
      with one of the supported shapes (see below).
    - ``section``: when supplied, prefer entries from
      ``obj[<section>]`` if the parsed file is a dict containing that
      key. Lets a single unified bank file carry per-task sections
      (``open_qa``, ``mcq_openended``, ``bcq_openended``,
      ``temporal_localization``) and drive all of them from one path
      (see :func:`reasoning.qa.llm.load_qa_bank` for the parallel
      QA-side knob).

    Both sources are concatenated in order (``queries`` first), then
    deduplicated while preserving order. Whitespace-only entries are
    silently dropped.

    Supported file shapes (any one):

    1. ``["query 1", "query 2", ...]`` — bare list of strings
    2. ``{"queries": [...]}`` — object with a ``queries`` key
    3. ``{"questions": [{"text": "...", ...}, ...]}`` — MCQ-style
       question bank (we extract ``text``); other fields are ignored,
       so a single bank can drive both MCQ and temporal_localization
       runs.
    4. **Unified / sectioned** ``{<section>: [<entry>, ...], ...}`` —
       only consulted when ``section`` is supplied. The section wins
       over shape 2/3; entries in the section may be plain strings
       or dicts carrying ``query`` / ``text`` / ``question``.

    Returns the deduplicated query list. Returns ``[]`` when neither
    source yields a usable string — callers should treat empty as
    "skip the stage".
    """
    out: list[str] = []
    if queries:
        out.extend(queries)

    if query_file is not None:
        path = Path(query_file)
        if not path.is_file():
            raise FileNotFoundError(f"temporal_localization query_file not found: {path}")
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in (".yaml", ".yml"):
            obj = yaml.safe_load(text)
        else:
            obj = json.loads(text)
        out.extend(_extract_queries_from_obj(obj, source=str(path), section=section))

    return _clean_queries(out)


def _extract_queries_from_obj(obj: Any, *, source: str, section: str | None = None) -> list[str]:
    """Pull a flat list of query strings out of a parsed YAML/JSON value.

    When ``section`` is supplied AND the object is a dict containing
    that key as a list, that section's entries are coerced to query
    strings (each entry may be a plain string or a dict carrying
    ``query`` / ``text`` / ``question``). Otherwise the legacy
    bare-list / ``{"queries"}`` / ``{"questions"}`` shapes are
    consulted in that order.
    """
    if isinstance(obj, list):
        return [str(x) for x in obj if isinstance(x, (str, int, float))]
    if isinstance(obj, dict):
        if section is not None and isinstance(obj.get(section), list):
            return _coerce_queries(obj[section])
        if isinstance(obj.get("queries"), list):
            return [str(x) for x in obj["queries"] if isinstance(x, (str, int, float))]
        questions = obj.get("questions")
        if isinstance(questions, list):
            return _coerce_queries(questions)
    raise ValueError(
        f"temporal_localization query file {source!r} must be a list of strings, "
        "an object with a 'queries' list, or an MCQ-style 'questions' list"
        + (f" (or a {section!r} list when using a unified bank)" if section else "")
    )


def _coerce_queries(entries: list[Any]) -> list[str]:
    """Coerce a heterogeneous list of strings/dicts to query strings.

    Strings pass through; dicts contribute the first non-empty value
    of ``query`` / ``text`` / ``question``. Anything else is dropped.
    """
    out: list[str] = []
    for q in entries:
        if isinstance(q, str):
            out.append(q)
        elif isinstance(q, dict):
            text = q.get("query") or q.get("text") or q.get("question")
            if isinstance(text, str):
                out.append(text)
    return out


def _clean_queries(queries: Iterable[Any]) -> list[str]:
    """Strip, drop blanks, dedupe (preserving order). Tolerant of mixed types."""
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        if not isinstance(q, str):
            continue
        s = q.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _temporal_localization_guided_schema() -> dict[str, Any]:
    """Guided-JSON schema mirroring the ``items`` wrapper the LLM emits.

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
                    "required": ["question", "answer"],
                    "properties": {
                        "question": {"type": "string"},
                        "answer": {
                            "type": "object",
                            "required": ["start", "end"],
                            "properties": {
                                "start": {"type": "string"},
                                "end": {"type": "string"},
                            },
                        },
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
    "TemporalLocalizationLLMError",
    "generate_temporal_localizations_with_llm",
    "load_query_bank",
]
