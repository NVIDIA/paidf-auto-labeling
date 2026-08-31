# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Single-call tiered "query bundle" generation (PAS image-augmentation flow).

The image-augmentation pipeline consumes pre-extracted person crops and emits
synonymous retrieval queries in the shape
``{"queries": {"easy": [...], "medium": [...], "hard": [...]}}`` produced by a
single LLM call over the structured visual description. This differs from the
legacy per-tier generation (``llm_queries``) where ``easy`` is template-based and
the LLM only fills ``medium``/``hard``.

The prompt is fully caller-supplied (config ``query_prompt_text`` /
``query_prompt_file``) so the query specification can evolve without code
changes. A ``.json`` prompt file in the Visual QA question-bank shape
(``{"questions": [{"id": ..., "question": ...}]}``) is accepted and its question
text is used as the prompt; any other file is treated as raw prompt text.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from core.llm.json_extract import extract_json_object
from core.model_clients import ChatRequest, EndpointClient

from person_attribute_search.export.benchmark import attributes_to_legacy_dict
from person_attribute_search.queries import QuerySet
from person_attribute_search.response_retry import generate_with_response_retries
from person_attribute_search.schema import PersonAttributes

_DESCRIPTION_PLACEHOLDERS = ("{description}", "{attributes}")


def resolve_bundle_prompt(*, prompt_text: str | None, prompt_file: str | None) -> str:
    """Resolve the query-bundle prompt from inline text or a file.

    Inline ``prompt_text`` wins. A ``.json`` file in the question-bank shape
    contributes the concatenated ``question`` fields; any other file is used
    verbatim as the prompt text.

    Raises:
        ValueError: If neither source yields a non-empty prompt.
    """
    if prompt_text and prompt_text.strip():
        return prompt_text.strip()
    if prompt_file:
        text = _load_prompt_file(Path(prompt_file))
        if text:
            return text
    raise ValueError("bundle_query_generation requires query_prompt_text or query_prompt_file")


def _load_prompt_file(path: Path) -> str:
    """Load a prompt file, extracting question text from question-bank JSON."""
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return raw.strip()
        questions = payload.get("questions") if isinstance(payload, dict) else None
        if isinstance(questions, list):
            texts = [
                str(item.get("question") or "").strip()
                for item in questions
                if isinstance(item, dict) and str(item.get("question") or "").strip()
            ]
            if texts:
                return "\n\n".join(texts)
    return raw.strip()


def render_bundle_prompt(prompt: str, *, description_json: str) -> str:
    """Inject the structured visual description into the bundle prompt.

    Substitutes the first supported placeholder (``{description}`` or
    ``{attributes}``) when present; otherwise appends the description under a
    labelled section so a placeholder-free prompt still receives the context it
    references.
    """
    for placeholder in _DESCRIPTION_PLACEHOLDERS:
        if placeholder in prompt:
            return prompt.replace(placeholder, description_json)
    return f"{prompt}\n\nStructured visual description:\n{description_json}"


def _coerce_query_list(value: object, *, cap: int) -> list[str]:
    """Coerce a parsed tier value into de-duplicated, capped query strings."""
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for entry in value:
        text = entry.get("query") if isinstance(entry, dict) else entry
        text = str(text).strip() if text is not None else ""
        if text and text not in seen:
            seen.add(text)
            out.append(text)
        if cap > 0 and len(out) >= cap:
            break
    return out


def parse_bundle_response(
    text: str,
    *,
    easy_cap: int = 0,
    medium_cap: int = 0,
    hard_cap: int = 0,
    strict_count: int = 0,
) -> QuerySet:
    """Parse a ``{"queries": {"easy", "medium", "hard"}}`` LLM response.

    Tolerates a top-level ``easy``/``medium``/``hard`` object (no ``queries``
    wrapper) and entries given either as strings or ``{"query": ...}`` objects.
    A per-tier ``cap`` of ``0`` means unlimited. ``strict_count`` of ``0``
    preserves tolerant pipeline behavior; a positive value requires at least
    that many de-duplicated queries per tier, then applies the requested caps.

    Args:
        text: Raw model response text.
        easy_cap: Optional maximum easy queries (``0`` = unlimited).
        medium_cap: Optional maximum medium queries (``0`` = unlimited).
        hard_cap: Optional maximum hard queries (``0`` = unlimited).
        strict_count: Minimum number of distinct queries per tier (``0`` =
            tolerant mode). Extra valid queries are capped instead of rejected.

    Returns:
        A :class:`QuerySet`; empty tiers when absent or unparseable.
    """
    if strict_count < 0:
        raise ValueError("strict_count must be non-negative")
    parsed = extract_json_object(text)
    if not isinstance(parsed, dict):
        if strict_count:
            raise ValueError("LLM response did not contain a JSON object")
        return QuerySet()
    tiers = parsed.get("queries")
    if not isinstance(tiers, dict):
        tiers = parsed
    cap_before_validation = 0 if strict_count else easy_cap
    easy_texts = _coerce_query_list(tiers.get("easy"), cap=cap_before_validation)
    cap_before_validation = 0 if strict_count else medium_cap
    medium_texts = _coerce_query_list(tiers.get("medium"), cap=cap_before_validation)
    cap_before_validation = 0 if strict_count else hard_cap
    hard = _coerce_query_list(tiers.get("hard"), cap=cap_before_validation)
    if strict_count:
        tier_values = {"easy": easy_texts, "medium": medium_texts, "hard": hard}
        for tier, values in tier_values.items():
            if len(values) < strict_count:
                raise ValueError(
                    f"LLM returned {len(values)} distinct {tier} queries; expected at least "
                    f"{strict_count}"
                )
        easy_texts = easy_texts[:easy_cap] if easy_cap else easy_texts
        medium_texts = medium_texts[:medium_cap] if medium_cap else medium_texts
        hard = hard[:hard_cap] if hard_cap else hard
    easy = [(query, "") for query in easy_texts]
    medium = [(query, "") for query in medium_texts]
    return QuerySet(easy=easy, medium=medium, hard=hard)


@dataclass(frozen=True)
class BundleQueryParams:
    """Tunables for the single-call query-bundle builder."""

    prompt: str
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    easy_cap: int = 0
    medium_cap: int = 0
    hard_cap: int = 0
    strict_count: int = 0


class BundleQuerySetBuilder:
    """Build a tiered :class:`QuerySet` per person with one LLM call.

    Callable as ``builder(attributes, upstream_hard) -> QuerySet`` so it drops
    into the same seam as the deterministic assembler and the per-tier LLM
    builder. ``upstream_hard`` is unused: the bundle prompt produces all three
    tiers directly from the structured visual description.
    """

    def __init__(
        self,
        client: EndpointClient,
        params: BundleQueryParams,
        *,
        response_retries: int = 0,
        response_retry_backoff_s: float = 0.0,
        logger: logging.Logger | None = None,
    ) -> None:
        """Store the LLM client and generation params (rendered prompt included)."""
        self._client = client
        self._params = params
        self._response_retries = response_retries
        self._response_retry_backoff_s = response_retry_backoff_s
        self._logger = logger

    def __call__(
        self,
        attributes: PersonAttributes,
        upstream_hard: Iterable[str] = (),
    ) -> QuerySet:
        """Generate a tiered ``QuerySet`` for one person via a single LLM call."""
        description_json = json.dumps(attributes_to_legacy_dict(attributes), indent=2)
        prompt = render_bundle_prompt(self._params.prompt, description_json=description_json)
        request = ChatRequest(
            prompt=prompt,
            max_tokens=self._params.max_tokens,
            temperature=self._params.temperature,
            top_p=self._params.top_p,
        )
        raw = generate_with_response_retries(
            self._client,
            request,
            validate_response=self._validation_error,
            stage="pas:bundle_queries",
            retries=self._response_retries,
            retry_backoff_s=self._response_retry_backoff_s,
            logger=self._logger,
        )
        return parse_bundle_response(
            raw,
            easy_cap=self._params.easy_cap,
            medium_cap=self._params.medium_cap,
            hard_cap=self._params.hard_cap,
            strict_count=self._params.strict_count,
        )

    def _validation_error(self, raw: str) -> str | None:
        """Return why a bundle response is unusable, or ``None`` when valid."""
        try:
            query_set = parse_bundle_response(
                raw,
                easy_cap=self._params.easy_cap,
                medium_cap=self._params.medium_cap,
                hard_cap=self._params.hard_cap,
                strict_count=self._params.strict_count,
            )
        except ValueError as exc:
            return str(exc)
        missing = [
            tier
            for tier, values in (
                ("easy", query_set.easy),
                ("medium", query_set.medium),
                ("hard", query_set.hard),
            )
            if not values
        ]
        if missing:
            return f"missing usable {', '.join(missing)} query tier(s)"
        return None


def query_set_to_bundle(query_set: QuerySet) -> dict[str, list[str]]:
    """Render a :class:`QuerySet` as the flat ``{easy, medium, hard}`` bundle."""
    return query_set.to_flat_dict()


__all__ = [
    "BundleQueryParams",
    "BundleQuerySetBuilder",
    "parse_bundle_response",
    "query_set_to_bundle",
    "render_bundle_prompt",
    "resolve_bundle_prompt",
]
