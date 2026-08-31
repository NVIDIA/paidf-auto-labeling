# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM reasoning enrichment for DAFT task items.

Produces the optional ``reasoning`` field that the prose task schemas
(``scene_description.json``, ``video_summarization.json``,
``temporal_description.json``) expose. The pipeline already writes the
``question`` and ``answer`` for each item; this module re-prompts the
LLM with the (question, answer, optional context) triple and asks for a
short justification trace.

Architectural posture (mirrors :mod:`reasoning.msted.llm`):

- **Pure adapter, no DAFT shaping.** :func:`generate_reasoning_for_item`
  returns a plain string; the caller decides whether to attach it to a
  task item or drop it.
- **Use-case agnostic.** Prompt comes from a :class:`PromptVariant` the
  caller picks from the registry. The bundled ``reasoning_default``
  variant works for any (question, answer, context) triple; ship a
  YAML drop-in to specialize.
- **No domain placeholders.** The template only references ``question``,
  ``answer``, and a ``context_block`` that the caller composes from
  whichever upstream signals (per-window captions, scene description,
  events) make sense for the use case.
- **Plain text response.** Reasoning traces are short prose, so this
  module uses ``call_chat_raw`` rather than the structured-JSON path.
  The LLM prompt explicitly asks for "no markdown, no labels, no
  preamble" so the returned string is paste-ready.

Why "best-effort" matters here: a missing reasoning trace is fine
(the schema field is optional). A failed LLM call must NEVER prevent
the underlying task file from being written. Callers should swallow
:class:`ReasoningLLMError` and skip just the reasoning field.
"""

from __future__ import annotations

import logging

from reasoning.llm_client import call_chat_raw, get_llm_api_key
from reasoning.prompts import PromptVariant

# Hard cap on the rendered reasoning length. The schema doesn't impose
# one but pasting a 10k-char reasoning trace into a task item is almost
# certainly an LLM hallucination, not a useful annotation. The default
# is generous (a few paragraphs) so a structured observation ->
# inference -> conclusion trace isn't truncated.
_DEFAULT_MAX_CHARS: int = 4_000


class ReasoningLLMError(RuntimeError):
    """Raised when the reasoning LLM call fails to return usable text.

    Distinct from :class:`reasoning.common.DaftConvertError`: this
    fires for transport / empty-response failures, *before* the result
    is offered to a converter. Pipeline callers swallow this and emit
    the underlying task file without the optional ``reasoning`` field
    rather than failing the run."""


def generate_reasoning_for_item(
    *,
    question: str,
    answer: str,
    context: str | None,
    prompt: PromptVariant,
    llm_url: str,
    llm_model: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    timeout: int = 120,
    seed: int | None = None,
    retries: int = 2,
    retry_backoff_s: float = 5.0,
    api_key: str | None = None,
    logger: logging.Logger | None = None,
    max_chars: int = _DEFAULT_MAX_CHARS,
    retry_stage: str = "reasoning",
) -> str:
    """Return a reasoning trace for the given (question, answer, context).

    The prompt template's ``{context_block}`` placeholder receives a
    pre-composed string: either ``"Context:\\n<context>"`` when
    ``context`` is non-empty, or an empty string. The caller is in
    charge of deciding what counts as "context" for the use case
    (per-window captions, scene_description, events list, ...) — this
    module never reaches into PL data structures, keeping it
    domain-neutral.

    Returns the LLM's reply, stripped of leading/trailing whitespace
    and capped to ``max_chars``. Empty replies (whitespace only) raise
    :class:`ReasoningLLMError` so the caller can skip the field rather
    than emit ``"reasoning": ""``.

    Raises :class:`ReasoningLLMError` on empty / unusable response.
    Re-raises :class:`PromptError` from prompt rendering as-is so the
    caller can distinguish "config is broken" from "LLM is broken"."""
    log = logger or logging.getLogger(__name__)

    q = _require_nonempty(question, "question")
    a = _require_nonempty(answer, "answer")

    context_block = _compose_context_block(context)

    user_text = prompt.render_user(
        question=q,
        answer=a,
        context_block=context_block,
    )

    messages = [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": user_text},
    ]

    raw = call_chat_raw(
        base_url=llm_url,
        model=llm_model,
        messages=messages,
        timeout=int(timeout),
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        seed=seed,
        retries=int(retries),
        retry_backoff_s=float(retry_backoff_s),
        logger=log,
        retry_stage=retry_stage,
        api_key=api_key or get_llm_api_key(),
    )

    text = (raw or "").strip()
    if not text:
        raise ReasoningLLMError("reasoning LLM call returned empty/whitespace text")

    # Strip common LLM verbosity: a leading "Reasoning:" label or a
    # wrapping pair of quotes. The prompt asks the LLM not to emit
    # these, but defenses-in-depth makes the field paste-ready
    # regardless.
    text = _strip_label_prefix(text)
    text = _strip_outer_quotes(text)

    if len(text) > max_chars:
        log.warning(
            "[reasoning] LLM produced %d chars, truncating to %d (likely a hallucination)",
            len(text),
            max_chars,
        )
        text = text[:max_chars].rstrip()

    if not text:
        raise ReasoningLLMError("reasoning text became empty after post-processing")

    return text


def _compose_context_block(context: str | None) -> str:
    """Render the optional ``{context_block}`` value the template expects.

    Returns either ``"Context:\\n<text>"`` or ``""`` so the prompt
    template stays clean — no awkward "Context: None" line when the
    caller has nothing to add."""
    if context is None or not isinstance(context, str):
        return ""
    s = context.strip()
    if not s:
        return ""
    return f"Context:\n{s}"


def _strip_label_prefix(text: str) -> str:
    """Drop a common LLM "Reasoning:" / "Rationale:" / "Explanation:" prefix.

    Case-insensitive on the label, tolerant of optional surrounding
    whitespace. Preserves the rest of the response verbatim — the LLM
    sometimes writes the label even when explicitly told not to, and
    we'd rather strip it than ship a self-labeled trace."""
    stripped = text.lstrip()
    for label in ("reasoning:", "rationale:", "explanation:", "answer:"):
        if stripped.lower().startswith(label):
            return stripped[len(label) :].lstrip()
    return text


def _strip_outer_quotes(text: str) -> str:
    """Drop a single pair of wrapping quotes if the entire text is quoted.

    LLMs sometimes wrap a "single sentence" answer in quotes despite
    the prompt asking for plain text; strip those once at the
    boundaries (never inside the body, never if quotes are unbalanced)."""
    s = text.strip()
    if len(s) < 2:
        return text
    pairs = (("'", "'"), ('"', '"'), ("`", "`"))
    for opener, closer in pairs:
        if s.startswith(opener) and s.endswith(closer):
            inner = s[len(opener) : -len(closer)]
            # Refuse to strip when the inner string contains the same
            # quote char — that's two separate quoted spans, not one
            # wrapper.
            if opener not in inner and closer not in inner:
                return inner.strip()
    return text


def _require_nonempty(value: object, label: str) -> str:
    """Validate that ``value`` is a non-empty string; raise otherwise.

    The reasoning prompt is meaningful only with both a question and
    an answer; we fail fast at the adapter boundary rather than letting
    the LLM hallucinate based on a blank prompt slot."""
    if not isinstance(value, str):
        raise ReasoningLLMError(f"reasoning {label} must be a string, got {type(value).__name__}")
    s = value.strip()
    if not s:
        raise ReasoningLLMError(f"reasoning {label} is empty after strip()")
    return s


__all__ = ["ReasoningLLMError", "generate_reasoning_for_item"]
