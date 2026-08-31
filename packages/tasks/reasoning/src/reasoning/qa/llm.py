# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM adapter shared by the three open-ended QA task families.

Pipeline-side glue between PL's per-window VLM captions and the three
pure converters in :mod:`reasoning.qa`. One adapter handles all three
families because they share inputs (windows + scene prose + a question
bank) and only differ in:

- the answer-shape constraint enforced downstream by the converter, and
- the optional ``options`` slot that ``mcq_openended`` carries.

Architectural posture (mirrors :mod:`reasoning.msted.llm` and
:mod:`reasoning.temporal_localization.llm`):

- **Pure adapter, no DAFT shaping.** Returns a list of structured
  items; the converter is the source of truth for shape validation.
- **Use-case agnostic.** Prompt comes from a :class:`PromptVariant`
  the caller picks from the registry. Bundled defaults
  (``open_qa_default`` / ``mcq_openended_default`` / ``bcq_openended_default``)
  are domain-neutral; ship a YAML drop-in to specialize.
- **Question bank is caller-supplied data.** :func:`load_qa_bank`
  accepts three file shapes (bare list, ``{"questions": [...]}``,
  MCQ-style ``{"questions": [{"question": "..."}, ...]}``) so a
  single bank file can drive multiple stages. Empty bank -> caller
  skips the stage.
- **No URLs / models / API keys baked in.** Endpoint args come from
  the caller's :class:`EndpointResolver`.
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

QaKind = Literal["open_qa", "mcq_openended", "bcq_openended"]

_KIND_VALUES: tuple[str, ...] = ("open_qa", "mcq_openended", "bcq_openended")


class QaLLMError(RuntimeError):
    """Raised when a QA LLM call fails to return usable structured items.

    Distinct from :class:`reasoning.common.DaftConvertError`: this
    fires for transport / parse / shape failures *before* the result
    reaches the converter. Pipeline-level callers swallow this and
    skip writing the file (best-effort)."""


# ---------------------------------------------------------------------------
# Question / item bank loading
# ---------------------------------------------------------------------------


def load_qa_bank(
    *,
    questions: list[Any] | None,
    question_file: Path | str | None,
    section: str | None = None,
) -> list[dict[str, Any]]:
    """Resolve the per-scene QA bank from inline config + optional file.

    The returned list normalizes each entry to a dict shape:

    - ``{"question": "...", "options": {...}}`` for ``mcq_openended``
      (or any caller that supplies options).
    - ``{"question": "..."}`` otherwise (open_qa / bcq_openended /
      mcq_openended without inline options — the LLM is then expected
      to embed options inline in its answer prose, per the schema).

    Inputs:

    - ``questions``: inline list from the pipeline config. Each entry
      may be a plain string (treated as ``{"question": <str>}``) or a
      dict with at least ``question`` (alias ``text``) and optionally
      ``options``.
    - ``question_file``: path to a YAML or JSON file with one of the
      supported shapes (see below).
    - ``section``: when supplied, prefer entries from
      ``obj[<section>]`` if the parsed file is a dict containing that
      key. This lets a single "unified" bank file carry per-task
      sections (``open_qa``, ``mcq_openended``, ``bcq_openended``,
      ``temporal_localization``) and serve all of them from one path.
      The legacy top-level ``questions`` / ``items`` array remains
      available for callers that don't pass a section (notably the
      ``mcq_generation`` ``question-driven-vlm-llm`` mode reads it
      via a different code path), so adding sections to an existing
      bank file is fully backward-compatible.

    Both sources are concatenated in order (``questions`` first), then
    deduplicated on the question text while preserving order.
    Whitespace-only entries are silently dropped.

    Supported file shapes (any one):

    1. Bare list of strings or item dicts.
    2. ``{"questions": [<entry>, ...]}``.
    3. MCQ-style ``{"questions": [{"text": "...", ...}, ...]}`` —
       a single bank can drive both an MCQ generation pass and an
       open-ended QA pass; non-question fields are ignored.
    4. **Unified / sectioned** ``{<section>: [<entry>, ...], ...}`` —
       only consulted when ``section`` is supplied. Coexists with
       shape 2/3 (the section wins; the legacy ``questions`` array
       is ignored for this caller but kept available for the
       ``mcq_generation`` consumer).

    Returns the deduplicated bank. Returns ``[]`` when neither source
    yields a usable entry — callers should treat empty as "skip the
    stage".
    """
    out: list[dict[str, Any]] = []

    if questions:
        for entry in questions:
            normalized = _normalize_entry(entry, source="inline")
            if normalized is not None:
                out.append(normalized)

    if question_file is not None:
        path = Path(question_file)
        if not path.is_file():
            raise FileNotFoundError(f"qa question_file not found: {path}")
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in (".yaml", ".yml"):
            obj = yaml.safe_load(text)
        else:
            obj = json.loads(text)
        for entry in _extract_entries_from_obj(obj, source=str(path), section=section):
            normalized = _normalize_entry(entry, source=str(path))
            if normalized is not None:
                out.append(normalized)

    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for entry in out:
        q = entry["question"]
        if q in seen:
            continue
        seen.add(q)
        deduped.append(entry)
    return deduped


def _normalize_entry(entry: Any, *, source: str) -> dict[str, Any] | None:
    """Coerce an entry into ``{"question": str, "options"?: dict}`` or drop it."""
    if isinstance(entry, str):
        s = entry.strip()
        return {"question": s} if s else None
    if not isinstance(entry, dict):
        return None
    raw_q = entry.get("question") or entry.get("text") or entry.get("query")
    if not isinstance(raw_q, str):
        return None
    q = raw_q.strip()
    if not q:
        return None
    out: dict[str, Any] = {"question": q}
    options = entry.get("options")
    if isinstance(options, dict) and options:
        # Preserve the caller's options as-is; the converter validates
        # shape (letter keys, non-empty string values) at write time.
        out["options"] = dict(options)
    return out


def _extract_entries_from_obj(obj: Any, *, source: str, section: str | None = None) -> list[Any]:
    """Pull a flat list of entries out of a parsed YAML/JSON value.

    When ``section`` is supplied AND the object is a dict containing
    that key as a list, that section's entries are returned (unified
    bank shape). If the requested section exists but is not a list, the
    bank is malformed and we raise instead of silently falling back.
    Otherwise the legacy bare-list / ``{"questions"}`` / ``{"items"}``
    shapes are consulted, in that order.
    """
    if isinstance(obj, list):
        return list(obj)
    if isinstance(obj, dict):
        if section is not None and section in obj:
            entries = obj[section]
            if not isinstance(entries, list):
                raise ValueError(
                    f"qa bank section {section!r} in {source!r} must be a list, "
                    f"got {type(entries).__name__}"
                )
            return list(entries)
        if isinstance(obj.get("questions"), list):
            return list(obj["questions"])
        if isinstance(obj.get("items"), list):
            return list(obj["items"])
    raise ValueError(
        f"qa bank file {source!r} must be a list or an object with a "
        "'questions' / 'items' list"
        + (f" (or a {section!r} list when using a unified bank)" if section else "")
    )


# ---------------------------------------------------------------------------
# LLM dispatcher
# ---------------------------------------------------------------------------


def generate_qa_with_llm(
    *,
    kind: QaKind,
    bank: list[dict[str, Any]],
    windows: Iterable[Any] | None = None,
    image_caption: str | None = None,
    scene_description: str | None,
    event_summary: str | None,
    prompt: PromptVariant,
    llm_url: str,
    llm_model: str,
    max_tokens: int = 2048,
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
    """Run one LLM call and return parsed QA items for ``kind``.

    The dispatcher selects:

    - the guided-JSON schema for the LLM (per-``kind`` answer shape).
    - the per-item ``options`` carry-through for ``mcq_openended``
      (we re-attach the caller's bank options to the LLM's response so
      the converter sees a full ``{question, options, answer}`` item
      even when the LLM elided ``options`` from its JSON).

    Inputs:

    - ``kind``: which family to generate for. Must be one of
      ``"open_qa"``, ``"mcq_openended"``, ``"bcq_openended"``.
    - ``bank``: the normalized QA bank (output of :func:`load_qa_bank`).
      Each entry is ``{"question": ..., "options"?: {...}}``.
    - ``windows`` *(video mode)*: per-segment dicts (same shape PL's
      window MCQ runners write to ``sidecars/metadata.json``). Mutually
      exclusive with ``image_caption``.
    - ``image_caption`` *(image mode)*: a single caption string for an
      image scene. Mutually exclusive with ``windows``. The DAFT QA
      schemas accept both ``video_id`` and ``image_id``, so the same
      LLM family + prompt set serves images by feeding a single
      "Image caption" line where the per-window block would be.
    - ``scene_description`` / ``event_summary``: scene-level prose
      threaded into the prompt as additional context. Either may be
      ``None``. For image scenes, ``scene_description`` typically
      mirrors ``image_caption`` and ``event_summary`` is omitted.

    Returns the parsed item list (raw LLM JSON, not yet DAFT-shaped).
    Hand the list to the matching ``to_daft_*`` converter in
    :mod:`reasoning.qa`.

    Raises :class:`QaLLMError` on empty bank, no usable windows /
    image caption, empty / unparseable LLM response, or ``items`` not a
    non-empty list of dicts. Re-raises :class:`PromptError` for
    prompt-rendering failures (caller can distinguish "LLM is broken"
    from "config is broken").
    """
    if kind not in _KIND_VALUES:
        raise QaLLMError(f"qa_llm.kind must be one of {list(_KIND_VALUES)}, got {kind!r}")

    log = logger or logging.getLogger(__name__)

    if not bank:
        raise QaLLMError(f"qa_llm[{kind}]: bank is empty after normalization; skipping")

    # Mode selection: exactly one of (windows, image_caption) must be
    # supplied. Both / neither is a programmer error and we surface it
    # before spending an LLM round-trip.
    has_windows = windows is not None
    image_caption_text = image_caption.strip() if isinstance(image_caption, str) else ""
    has_caption = image_caption_text != ""
    if has_windows == has_caption:
        raise QaLLMError(
            f"qa_llm[{kind}]: exactly one of 'windows' (video mode) or "
            "'image_caption' (image mode) must be supplied; "
            f"got windows={'set' if has_windows else 'None'}, "
            f"image_caption={'set' if has_caption else 'unset/empty'}"
        )

    if has_caption:
        # Image mode: render a single line that fills the same
        # ``{windows_block}`` slot the video prompts already use, so we
        # don't have to fork the bundled prompt set just to swap a
        # label. The DAFT QA schemas (``open_qa`` / ``mcq_openended`` /
        # ``bcq_openended``) all accept ``image_id`` via ``oneOf``, so
        # the converter happily emits an image-keyed payload.
        rendered_windows_block = f"Image caption: {image_caption_text}"
        scene_label = "image"
    else:
        win_lines = _format_windows_block(
            windows or (),
            description_keys=description_keys,
            log=log,
            tag="qa_llm",
        )
        if not win_lines:
            raise QaLLMError(
                f"qa_llm[{kind}]: no usable windows (each window must have start, "
                "end, and a caption-like field); skipping"
            )
        rendered_windows_block = "\n".join(win_lines)
        scene_label = "video"

    questions_block = _format_questions_block(bank, kind=kind)

    user_text = prompt.render_user(
        windows_block=rendered_windows_block,
        scene_description_block=_block(
            "Scene description:",
            scene_description,
            fallback="(no scene-level description provided)",
        ),
        event_summary_block=_block(
            "Event summary:",
            event_summary,
            fallback="",
        ),
        questions_block=questions_block,
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
        guided_json_schema=_guided_schema_for(kind),
        retry_stage=f"qa:{kind}:{scene_label}",
        api_key=api_key or get_llm_api_key(),
        error_type=QaLLMError,
        no_parseable_message=f"qa_llm[{kind}] returned no parseable JSON. Raw: {{snippet_repr}}",
        non_object_message=f"qa_llm[{kind}] returned a non-object payload: {{type_name}}",
        caller=call_chat_object_with_structured_fallback,
    )
    items = require_items_list(
        parsed,
        error_type=QaLLMError,
        missing_message=f"qa_llm[{kind}] payload missing 'items' list, got {{type_name}}",
        empty_message=f"qa_llm[{kind}] returned an empty 'items' list",
    )

    # Re-attach options from the bank when the LLM elided them. The
    # bank is the source of truth for the option text; the LLM's job
    # is to pick the right letter and explain. Prefer stable ids when
    # available; otherwise use the LLM item order, which is less brittle
    # than matching the question text after the model rewrites it.
    bank_by_stable_id, bank_by_index, bank_by_position = _index_bank_entries(bank)
    out_items: list[dict[str, Any]] = []
    for item_idx, it in enumerate(items):
        if not isinstance(it, dict):
            log.warning("[qa_llm:%s] dropping non-dict item from LLM payload: %r", kind, it)
            continue
        merged = dict(it)
        if kind == "mcq_openended":
            bank_entry = _resolve_bank_entry(
                merged,
                item_idx=item_idx,
                bank_by_stable_id=bank_by_stable_id,
                bank_by_index=bank_by_index,
                bank_by_position=bank_by_position,
            )
            if bank_entry is not None:
                bank_options = bank_entry.get("options")
                if isinstance(bank_options, dict) and bank_options and "options" not in merged:
                    merged["options"] = dict(bank_options)
        out_items.append(merged)

    if not out_items:
        raise QaLLMError(f"qa_llm[{kind}] returned 'items' but none were dicts")

    return out_items


def _index_bank_entries(
    bank: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    by_stable_id: dict[str, dict[str, Any]] = {}
    by_index: dict[int, dict[str, Any]] = {}
    by_position: dict[int, dict[str, Any]] = {}
    for idx, entry in enumerate(bank):
        by_position[idx] = entry
        by_index[idx + 1] = entry
        for key in ("id", "question_id"):
            value = entry.get(key)
            if value is not None:
                by_stable_id[str(value)] = entry
        value = entry.get("index")
        index = _coerce_index(value)
        if index is not None:
            by_index[index] = entry
    return by_stable_id, by_index, by_position


def _resolve_bank_entry(
    merged: dict[str, Any],
    *,
    item_idx: int,
    bank_by_stable_id: dict[str, dict[str, Any]],
    bank_by_index: dict[int, dict[str, Any]],
    bank_by_position: dict[int, dict[str, Any]],
) -> dict[str, Any] | None:
    for key in ("question_id", "id"):
        value = merged.get(key)
        if value is None:
            continue
        match = bank_by_stable_id.get(str(value))
        if match is not None:
            return match

    index = _coerce_index(merged.get("index"))
    if index is not None and index in bank_by_index:
        return bank_by_index[index]
    return bank_by_position.get(item_idx)


def _coerce_index(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Prompt-block formatting
# ---------------------------------------------------------------------------


def _format_questions_block(bank: list[dict[str, Any]], *, kind: QaKind) -> str:
    """Render the per-question prompt block.

    For ``mcq_openended`` items that carry ``options``, we expand them
    inline as ``A: <text>`` / ``B: <text>`` lines so the LLM sees the
    choice space without re-deriving it. The default prompt already
    instructs the model to use these letters; this just supplies the
    catalogue."""
    lines: list[str] = []
    for i, entry in enumerate(bank, start=1):
        q = entry.get("question", "")
        lines.append(f"{i}. {q}")
        if kind == "mcq_openended":
            options = entry.get("options")
            if isinstance(options, dict) and options:
                # Stable A..Z ordering.
                for letter in sorted(options):
                    val = options[letter]
                    if isinstance(val, str) and val.strip():
                        lines.append(f"   {letter}: {val.strip()}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Guided-JSON schemas
# ---------------------------------------------------------------------------


def _guided_schema_for(kind: QaKind) -> dict[str, Any]:
    """Return a guided-JSON schema mirroring the kind-specific item shape.

    NIM endpoints respect this hint to constrain decoding; OpenAI-style
    endpoints ignore it and rely on response_format=json_object. The
    converter is the source of truth for validation either way."""
    if kind == "open_qa":
        item: dict[str, Any] = {
            "type": "object",
            "required": ["question", "answer"],
            "properties": {
                "question": {"type": "string"},
                "answer": {"type": "string"},
                "reasoning": {"type": "string"},
            },
        }
    elif kind == "mcq_openended":
        item = {
            "type": "object",
            "required": ["question", "answer"],
            "properties": {
                "question": {"type": "string"},
                "answer": {
                    "type": "string",
                    "pattern": r"^[A-Z]\. [\s\S]+",
                },
                "options": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
                "reasoning": {"type": "string"},
            },
        }
    else:  # bcq_openended
        item = {
            "type": "object",
            "required": ["question", "answer"],
            "properties": {
                "question": {"type": "string"},
                "answer": {
                    "type": "string",
                    "pattern": r"^(Yes|No)\. [\s\S]+",
                },
                "reasoning": {"type": "string"},
            },
        }
    return {
        "type": "object",
        "required": ["items"],
        "properties": {
            "items": {
                "type": "array",
                "minItems": 1,
                "items": item,
            },
        },
    }


__all__ = [
    "QaKind",
    "QaLLMError",
    "generate_qa_with_llm",
    "load_qa_bank",
]
