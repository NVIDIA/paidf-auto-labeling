# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM classification pass for anomaly mode.

Pipeline-side glue between the per-window VLM captions and the anomaly
sidecar (:func:`reasoning.anomaly.sidecar.to_anomaly_sidecar`). The LLM
is asked, for a whole clip, to decide ``anomaly`` vs ``normal``, justify
the verdict with a reasoning trace, and — for anomalies — name the root
cause / consequence and localize the highlight window.

When a prior person-attribute-search pass ran, its grounded per-person
attributes (action, potential-anomaly flag/type, caption) are passed via
``person_attributes`` and rendered as a corroborating evidence block, so
the anomaly verdict can adjudicate *over* that perception rather than
re-deriving everything from captions alone. It stays optional: with no
PAS evidence the block degrades to a fallback line and behavior is
unchanged.

Mirrors :mod:`reasoning.msted.llm` deliberately:

- The prompt comes from a :class:`PromptVariant` the caller selects from
  the registry. The bundled default is use-case neutral; domain-specific
  variants are YAML drop-ins. **No prompt text or anomaly taxonomy is
  hard-coded in this module.**
- The endpoint (url / model / api key) comes from the caller. No URLs,
  models, or keys are baked in.
- The function is pure with respect to the filesystem: it returns a
  normalized dict and never writes anything.
"""

from __future__ import annotations

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
from reasoning.timecodes import seconds_to_timecode, timecode_to_seconds

#: The two verdicts the converter / causal-linkage stage understand.
#: Kept aligned with the DAFT causal-linkage ``video_type`` enum so the
#: anomaly verdict can flow straight into auto-tagging.
ANOMALY_LABELS: tuple[str, str] = ("anomaly", "normal")

#: Confidence buckets the prompt asks for. A free-form value is coerced
#: to ``None`` rather than rejected — confidence is advisory metadata.
_CONFIDENCE_LEVELS: frozenset[str] = frozenset({"low", "medium", "high"})


class AnomalyLLMError(RuntimeError):
    """Raised when the anomaly LLM call fails to return a usable verdict.

    Fires for transport / parse / empty-response failures and for a
    missing-or-invalid ``classification`` field — i.e. anything that
    means "we have no trustworthy verdict". Pipeline-level callers catch
    this and skip writing the sidecar (best-effort), exactly like the
    other LLM adapters in this package.
    """


def classify_anomaly_with_llm(
    *,
    windows: Iterable[dict[str, Any]],
    scene_description: str | None,
    prompt: PromptVariant,
    llm_url: str,
    llm_model: str,
    person_attributes: str | None = None,
    localize_highlight: bool = True,
    max_tokens: int = 1024,
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
    """Run one LLM call and return a normalized anomaly verdict dict.

    The returned dict is normalized but *not* the on-disk payload — hand
    it to :func:`reasoning.anomaly.sidecar.to_anomaly_sidecar` to build
    the sidecar (including the re-perception request). Shape::

        {
          "classification": "anomaly" | "normal",
          "confidence": "low" | "medium" | "high" | None,
          "reasoning": "<trace>",
          "root_cause": "<str>",     # anomalies only, else absent
          "consequence": "<str>",    # anomalies only, else absent
          "highlight": {             # anomalies only + localize_highlight
            "start": "MM:SS", "end": "MM:SS", "description": "<str>"
          },
        }

    Normalization rules:

    - ``classification`` is lowercased and must be one of
      :data:`ANOMALY_LABELS`; otherwise :class:`AnomalyLLMError`.
    - For a ``normal`` verdict, ``root_cause`` / ``consequence`` /
      ``highlight`` are dropped even if the model emitted them — they are
      only meaningful for anomalies.
    - ``highlight`` is kept only when both timecodes parse and
      ``end > start``; an invalid or reversed window is dropped (with a
      warning) rather than propagated to the captioning service.

    Raises :class:`AnomalyLLMError` on empty windows, an empty /
    unparseable response, a missing / invalid classification, or an
    empty / missing ``reasoning`` trace.
    Re-raises :class:`reasoning.prompts.PromptError` from rendering as-is.
    """
    log = logger or logging.getLogger(__name__)

    win_lines = _format_windows_block(
        windows, description_keys=description_keys, log=log, tag="anomaly"
    )
    if not win_lines:
        raise AnomalyLLMError(
            "no usable windows for anomaly LLM call (each window must have "
            "start, end, and a caption-like field)"
        )

    user_text = prompt.render_user(
        windows_block="\n".join(win_lines),
        scene_description_block=_block(
            "Scene description:",
            scene_description,
            fallback="(no scene-level description provided)",
        ),
        person_attributes_block=_block(
            "Person attributes (from person-attribute search):",
            person_attributes,
            fallback="(no person-attribute evidence provided)",
        ),
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
        guided_json_schema=_anomaly_guided_schema(),
        retry_stage="anomaly",
        api_key=api_key or get_llm_api_key(),
        error_type=AnomalyLLMError,
        no_parseable_message=(
            "anomaly LLM call returned no parseable JSON object. "
            "Raw response (truncated): {snippet_repr}"
        ),
        non_object_message="anomaly LLM returned a non-object payload: {type_name}",
        caller=call_chat_object_with_structured_fallback,
    )

    return _normalize_verdict(parsed, localize_highlight=localize_highlight, log=log)


def _normalize_verdict(
    parsed: dict[str, Any],
    *,
    localize_highlight: bool,
    log: logging.Logger,
) -> dict[str, Any]:
    """Coerce a raw LLM payload into the normalized verdict contract."""
    label = parsed.get("classification")
    if not isinstance(label, str) or label.strip().lower() not in ANOMALY_LABELS:
        raise AnomalyLLMError(
            f"anomaly LLM payload has missing/invalid 'classification': {label!r}; "
            f"expected one of {ANOMALY_LABELS}"
        )
    classification = label.strip().lower()

    out: dict[str, Any] = {"classification": classification}

    confidence = parsed.get("confidence")
    if isinstance(confidence, str) and confidence.strip().lower() in _CONFIDENCE_LEVELS:
        out["confidence"] = confidence.strip().lower()
    else:
        out["confidence"] = None

    reasoning = parsed.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        # ``reasoning`` is a required field (see the guided schema) and the
        # whole point of anomaly mode is the trace. An empty trace is a
        # contract violation, not a usable verdict — reject it rather than
        # persisting an empty string that downstream trace consumers trust.
        raise AnomalyLLMError(
            f"anomaly LLM payload has an empty/missing 'reasoning' trace: {reasoning!r}; "
            "rejecting verdict (reasoning is a required field)"
        )
    out["reasoning"] = reasoning.strip()

    # Root cause / consequence / highlight are anomaly-only signals.
    if classification == "anomaly":
        for key in ("root_cause", "consequence"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                out[key] = value.strip()
        if localize_highlight:
            highlight = _normalize_highlight(parsed.get("highlight"), log=log)
            if highlight is not None:
                out["highlight"] = highlight

    return out


def _normalize_highlight(
    raw: Any,
    *,
    log: logging.Logger,
) -> dict[str, str] | None:
    """Validate the highlight window; drop it when malformed.

    A highlight only earns its place in the sidecar (and the downstream
    re-perception request) when both timecodes parse against the DAFT
    timecode grammar and ``end`` is strictly after ``start``. Anything
    else is logged and dropped — a bad window is worse than no window
    because it would mislead the captioning re-perception pass.
    """
    if not isinstance(raw, dict):
        return None
    start = raw.get("start")
    end = raw.get("end")
    if not isinstance(start, str) or not isinstance(end, str):
        log.debug("[anomaly] highlight missing start/end strings; dropping")
        return None
    try:
        start_s = timecode_to_seconds(start.strip())
        end_s = timecode_to_seconds(end.strip())
    except ValueError:
        log.warning(
            "[anomaly] highlight has invalid timecode(s) start=%r end=%r; dropping highlight",
            start,
            end,
        )
        return None
    if end_s <= start_s:
        log.warning(
            "[anomaly] highlight end %r not after start %r; dropping highlight",
            end,
            start,
        )
        return None
    highlight = {
        "start": seconds_to_timecode(start_s),
        "end": seconds_to_timecode(end_s),
    }
    description = raw.get("description")
    if isinstance(description, str) and description.strip():
        highlight["description"] = description.strip()
    return highlight


def _anomaly_guided_schema() -> dict[str, Any]:
    """Guided-JSON schema mirroring the verdict shape the LLM emits.

    NIM endpoints respect this hint to constrain decoding; OpenAI-style
    endpoints ignore it and rely on response_format=json_object instead.
    :func:`_normalize_verdict` is the source of truth — this schema is a
    generation hint, not the final validation step.
    """
    return {
        "type": "object",
        "required": ["classification", "reasoning"],
        "properties": {
            "classification": {"type": "string", "enum": list(ANOMALY_LABELS)},
            "confidence": {"type": "string", "enum": sorted(_CONFIDENCE_LEVELS)},
            "reasoning": {"type": "string"},
            "root_cause": {"type": "string"},
            "consequence": {"type": "string"},
            "highlight": {
                "type": "object",
                "required": ["start", "end"],
                "properties": {
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                    "description": {"type": "string"},
                },
            },
        },
    }


__all__ = [
    "ANOMALY_LABELS",
    "AnomalyLLMError",
    "classify_anomaly_with_llm",
]
