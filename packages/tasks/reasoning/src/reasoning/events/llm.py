# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM aggregation pass that produces structured input for ``to_daft_events``.

Pipeline-side glue between PL's per-window dense captions
(``sidecars/metadata_chunk.json``) and
:func:`reasoning.events.converter.to_daft_events`. Keeping this module
separate from the converter means:

- Tests for the converter never need to mock LLM clients.
- Callers that already have structured event data (cached, hand-written,
  produced by a non-LLM source) can call the converter directly without
  importing this module — and therefore without dragging in the
  ``openai`` package, NVCF detection, or the prompt registry.

Use-case agnosticism: the prompt is a :class:`PromptVariant` the caller
picks from the registry. The bundled default
(``events_from_chunks.yaml``) is generic; domain-specific variants are
YAML drop-ins. **No prompt text is hard-coded in this module.**
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from typing import Any

from reasoning._llm_context import (
    format_windows_block as _format_windows_block,
)
from reasoning.llm_client import (
    call_chat_object_with_structured_fallback,
    call_required_json_object_with_fallback,
    get_llm_api_key,
)
from reasoning.prompts import PromptVariant


class EventsLLMError(RuntimeError):
    """Raised when the events aggregation pass fails to return usable JSON.

    Distinct from :class:`reasoning.common.DaftConvertError`: this fires
    for transport / parse / empty-response failures from the LLM, *before*
    any DAFT validation runs. Pipeline-level callers catch this to skip
    writing rather than failing the run.
    """


def generate_events_with_llm(
    *,
    windows: Iterable[Any],
    instances_catalogue: Iterable[Any] | None,
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
        "description",
        "enhanced_caption",
        "caption",
        "summary",
    ),
) -> dict[str, Any]:
    """Run the LLM aggregation pass and return a structured events dict.

    The returned dict is the raw LLM JSON: it has not yet been validated
    against the DAFT events schema. Hand it to
    :func:`reasoning.events.converter.to_daft_events` to get a
    DAFT-compliant payload (which is what enforces the schema).

    Inputs:

    - ``windows``: per-segment dicts (same shape PL's dense-caption stage
      writes to ``sidecars/metadata_chunk.json``). Each dict needs a
      start timestamp (``start_s`` / ``start`` / ``start_seconds`` /
      ``start_time``), an end timestamp (same family), and a caption
      (looked up via ``description_keys`` in order). Windows missing any
      of these are skipped with a debug log, not an error — partial data
      is better than no events file.
    - ``instances_catalogue``: optional iterable of dicts with at least
      ``object_id`` (or ``id``) keys, used to bound which IDs the LLM may
      reference in ``events[].instances``. Pass ``None`` or an empty
      iterable to omit the catalogue from the prompt entirely (the
      bundled prompt instructs the LLM to omit ``instances`` in that
      case).

    Endpoint args (``llm_url`` / ``llm_model`` / ``timeout`` / etc.) are
    passed straight to :func:`call_chat_object_with_structured_fallback`.
    ``api_key`` defaults to :func:`get_llm_api_key` so the same env-var
    fallback chain MCQ uses applies here too.

    Raises :class:`EventsLLMError` when the LLM returns nothing parseable.
    Re-raises :class:`PromptError` from prompt rendering as-is.
    """
    log = logger or logging.getLogger(__name__)

    win_lines = _format_windows_block(
        windows, description_keys=description_keys, log=log, tag="events_llm"
    )
    if not win_lines:
        raise EventsLLMError(
            "no usable windows for events LLM call (each window must have "
            "start, end, and a caption-like field)"
        )
    windows_block = "\n".join(win_lines)

    instances_lines: list[str] = []
    if instances_catalogue is not None:
        for entry in instances_catalogue:
            if not isinstance(entry, dict):
                continue
            oid = entry.get("object_id") or entry.get("id")
            if not isinstance(oid, str) or not oid.strip():
                continue
            label = entry.get("label") or entry.get("class") or entry.get("object_type")
            track_id = entry.get("track_id")
            extras: list[str] = []
            if isinstance(label, str) and label.strip():
                extras.append(f"class={label.strip()}")
            if track_id is not None:
                extras.append(f"track_id={track_id}")
            suffix = f" ({', '.join(extras)})" if extras else ""
            instances_lines.append(f"- {oid.strip()}{suffix}")
    instances_block = (
        "\n".join(instances_lines)
        if instances_lines
        else "(no instances catalogue available; omit 'instances' from each event)"
    )

    user_text = prompt.render_user(
        windows_block=windows_block,
        instances_block=instances_block,
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
        guided_json_schema=_events_guided_schema(),
        retry_stage="events",
        api_key=api_key or get_llm_api_key(),
        error_type=EventsLLMError,
        no_parseable_message=(
            "events LLM call returned no parseable JSON object. "
            "Raw response (truncated): {snippet_repr}"
        ),
        non_object_message="events LLM returned a non-object payload: {type_name}",
        caller=call_chat_object_with_structured_fallback,
    )

    return parsed


def _events_guided_schema() -> dict[str, Any]:
    """Return a guided-JSON schema mirroring the DAFT events structure.

    NIM endpoints respect this hint to constrain decoding; OpenAI-style
    endpoints ignore it and rely on response_format=json_object instead.
    Either way the converter is the source of truth — this schema is
    just a generation hint, not the final validation step.

    Kept loose intentionally:
    - ``events`` is allowed to be empty (an LLM that finds no noteworthy
      occurrences should emit ``"events": []`` rather than fabricating).
    - ``severity`` is enum-restricted at the converter, not here, so
      decoding doesn't fail mid-stream on a near-enum value the converter
      can normalize.
    """
    return {
        "type": "object",
        "required": ["events"],
        "properties": {
            "events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["event_id", "start_time", "end_time"],
                    "properties": {
                        "event_id": {"type": "string"},
                        "start_time": {"type": "string"},
                        "end_time": {"type": "string"},
                        "category": {"type": "string"},
                        "sub_category": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "instances": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "event_caption": {"type": "string"},
                        "severity": {"type": "string"},
                        "group_id": {"type": "string"},
                    },
                },
            },
            "groups": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["group_id"],
                    "properties": {
                        "group_id": {"type": "string"},
                        "description": {"type": "string"},
                    },
                },
            },
        },
    }


def parse_cached_events(text: str) -> dict[str, Any]:
    """Parse a cached events JSON blob (e.g. from a previous LLM run).

    Convenience for callers that store the LLM JSON to disk and want to
    re-feed it to :func:`to_daft_events` later. Strict: the file must
    parse to a dict, otherwise :class:`EventsLLMError` is raised so the
    caller falls back instead of writing garbage.
    """
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EventsLLMError(f"cached events JSON is not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise EventsLLMError(f"cached events JSON must be an object, got {type(obj).__name__}")
    return obj


__all__ = [
    "EventsLLMError",
    "generate_events_with_llm",
    "parse_cached_events",
]
