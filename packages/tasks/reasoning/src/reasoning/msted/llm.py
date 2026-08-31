# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM aggregation pass that produces structured input for ``to_daft_msted``.

Pipeline-side glue between PL's per-window VLM output and the
:func:`reasoning.msted.converter.to_daft_msted` converter. Keeping this as its
own module (separate from ``msted.py``) means:

- Tests for the converter never need to mock LLM clients.
- A user who already has structured MSTED data (cached, hand-written,
  produced by a different LLM) can call :func:`to_daft_msted` directly
  without ever importing this module — and therefore without dragging in
  the ``openai`` package, NVCF detection, or the prompt registry.

Use-case agnosticism (the explicit user constraint):

- The prompt comes from a :class:`PromptVariant` the caller picks from
  the registry. The bundled default is generic; domain-specific variants
  are YAML files. **No prompt text is hard-coded in this module.**
- The placeholders the prompt template can reference
  (``scene_description_block``, ``windows_block``, ``event_summary_block``)
  are built from PL data using neutral formatting; if a domain wants a
  different framing, it ships a different YAML.
- The LLM endpoint comes from the caller's :class:`EndpointResolver`.
  No URLs, models, or API keys are baked in.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from typing import Any

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
)
from reasoning.prompts import PromptVariant


class MstedLLMError(RuntimeError):
    """Raised when the LLM aggregation pass fails to return usable JSON.

    Distinct from :class:`reasoning.common.DaftConvertError`: this
    fires for transport / parse / empty-response failures from the LLM,
    *before* any DAFT validation runs. Pipeline-level callers catch
    this to fall back to skipping the file rather than failing the run.
    """


def generate_msted_with_llm(
    *,
    windows: Iterable[dict[str, Any]],
    scene_description: str | None,
    event_summary: str | None,
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
) -> dict[str, Any]:
    """Run the LLM aggregation pass and return a structured MSTED dict.

    The returned dict is the raw LLM JSON: it has not yet been validated
    against the DAFT MSTED schema. Hand it to
    :func:`reasoning.msted.converter.to_daft_msted` to get a DAFT-compliant
    payload (which is what enforces the schema).

    Inputs:

    - ``windows``: per-segment dicts (same shape PL's window MCQ runners
      write to ``sidecars/metadata.json``). Each dict needs a start
      timestamp (``start_s`` / ``start`` / ``start_seconds`` /
      ``start_time``), an end timestamp (same family), and a caption
      (looked up via ``description_keys`` in order). Windows missing any
      of these are skipped with a warning, not an error — partial data
      is better than no MSTED file.
    - ``scene_description``: optional scene-level prose to thread into
      the prompt as additional context.
    - ``event_summary``: optional scene-level event summary, similarly
      threaded in.
    - ``duration``: optional clip duration; not directly placed in the
      prompt but kept in the signature so callers can validate on the
      converter side.

    Endpoint args (``llm_url`` / ``llm_model`` / ``timeout`` / etc.) are
    passed straight to :func:`call_chat_object_with_structured_fallback`,
    which handles guided-JSON vs response-format selection per endpoint
    type. ``api_key`` defaults to :func:`get_llm_api_key` so the same
    env-var fallback chain MCQ uses applies here too.

    Raises :class:`MstedLLMError` when the LLM returns nothing parseable.
    Re-raises :class:`PromptError` from prompt rendering as-is (the
    caller can distinguish "LLM is broken" from "config is broken")."""
    log = logger or logging.getLogger(__name__)

    win_lines = _format_windows_block(
        windows, description_keys=description_keys, log=log, tag="msted_llm"
    )
    if not win_lines:
        raise MstedLLMError(
            "no usable windows for MSTED LLM call (each window must have "
            "start, end, and a caption-like field)"
        )
    windows_block = "\n".join(win_lines)

    scene_description_block = _block(
        "Scene description:", scene_description, fallback="(no scene-level description provided)"
    )
    event_summary_block = _block("Event summary:", event_summary, fallback="")

    user_text = prompt.render_user(
        windows_block=windows_block,
        scene_description_block=scene_description_block,
        event_summary_block=event_summary_block,
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
        guided_json_schema=_msted_guided_schema(),
        retry_stage="msted",
        api_key=api_key or get_llm_api_key(),
        error_type=MstedLLMError,
        no_parseable_message=(
            "MSTED LLM call returned no parseable JSON object. "
            "Raw response (truncated): {snippet_repr}"
        ),
        non_object_message="MSTED LLM returned a non-object payload: {type_name}",
        caller=call_chat_object_with_structured_fallback,
    )

    return parsed


def _msted_guided_schema() -> dict[str, Any]:
    """Return a guided-JSON schema mirroring the DAFT MSTED structure.

    NIM endpoints respect this hint to constrain decoding; OpenAI-style
    endpoints ignore it and rely on response_format=json_object instead.
    Either way the converter is the source of truth — this schema is
    just a generation hint, not the final validation step.

    Kept loose intentionally:
    - ``event_description.additionalProperties: {type: string}`` mirrors
      the DAFT schema's free-form event characterization, so the LLM
      can pick whichever keys best describe the use case rather than
      being constrained to a fixed taxonomy."""
    return {
        "type": "object",
        "required": [
            "scene_description",
            "temporal_spatial_localization",
            "event_description",
        ],
        "properties": {
            "scene_description": {"type": "string"},
            "temporal_spatial_localization": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["start", "end", "description"],
                    "properties": {
                        "start": {"type": "string"},
                        "end": {"type": "string"},
                        "description": {"type": "string"},
                        "spatial_region": {"type": "string"},
                    },
                },
            },
            "event_description": {
                "type": "object",
                "minProperties": 1,
                "additionalProperties": {"type": "string"},
            },
        },
    }


def parse_cached_msted(text: str) -> dict[str, Any]:
    """Parse a cached MSTED JSON blob (e.g. from a previous LLM run).

    Convenience for callers that store the LLM JSON to disk and want to
    re-feed it to :func:`to_daft_msted` later. Strict: the file must
    parse to a dict, otherwise :class:`MstedLLMError` is raised so the
    caller falls back instead of writing garbage."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MstedLLMError(f"cached MSTED JSON is not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise MstedLLMError(f"cached MSTED JSON must be an object, got {type(obj).__name__}")
    return obj


__all__ = [
    "MstedLLMError",
    "generate_msted_with_llm",
    "parse_cached_msted",
]
