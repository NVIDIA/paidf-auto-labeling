# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LLM difficulty-tiered query generation (legacy ``generate_queries.py`` parity).

This reproduces the original Person Attribute Search query-generation **LLM op**:
given a person's structured attributes and natural-language caption, call a text
LLM with the ``PROMPT_V3`` prompt to produce ``medium`` and ``hard`` retrieval
queries (or the ``PROMPT_V3_HARD_ONLY`` variant to produce ``hard`` only when
``medium`` is template-generated). ``easy`` queries are always template-based in
both the legacy and UPA pipelines, so they are not produced here.

The assembly task stays deterministic; this module is the single place that
performs the query-generation model call, mirroring how ``visual_qa`` owns its
own model clients.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field

from core.llm.json_extract import extract_json_object
from core.model_clients import ChatRequest, EndpointClient

from person_attribute_search.export.benchmark import attributes_to_legacy_dict
from person_attribute_search.prompts import render_query_prompt
from person_attribute_search.queries import QuerySet, assemble_query_set
from person_attribute_search.response_retry import generate_with_response_retries
from person_attribute_search.schema import PersonAttributes
from person_attribute_search.templates import QueryPair, build_easy_queries, build_medium_queries


def _coerce_medium_pairs(value: object) -> list[QueryPair]:
    """Coerce a parsed ``medium`` value into ``[(query, attrs_used), ...]``.

    Accepts the legacy ``[[query, used], ...]`` shape and tolerates bare strings
    (``used`` defaults to empty). Entries without a query string are dropped.
    """
    if not isinstance(value, list):
        return []
    pairs: list[QueryPair] = []
    for entry in value:
        if isinstance(entry, str):
            query, used = entry, ""
        elif isinstance(entry, (list, tuple)) and entry:
            query = entry[0]
            used = entry[1] if len(entry) > 1 else ""
        elif isinstance(entry, dict):
            query = entry.get("query", "")
            used_value = entry.get("attributes_used", "")
            used = (
                ", ".join(str(item) for item in used_value)
                if isinstance(used_value, list)
                else used_value
            )
        else:
            continue
        query_text = str(query).strip()
        if query_text:
            pairs.append((query_text, str(used).strip()))
    return pairs


def _coerce_hard_list(value: object) -> list[str]:
    """Coerce a parsed ``hard`` value into a list of non-empty query strings."""
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for entry in value:
        text = str(entry).strip()
        if text:
            out.append(text)
    return out


def parse_query_response(text: str) -> tuple[list[QueryPair], list[str]]:
    """
    Parse an LLM query-generation response into medium pairs and hard queries.

    Args:
        text: Raw model response text.

    Returns:
        ``(medium, hard)`` where ``medium`` is ``[(query, attrs_used), ...]`` and
        ``hard`` is ``[query, ...]``. Either list is empty when absent or
        unparseable.
    """
    parsed = extract_json_object(text)
    if parsed is None:
        return [], []
    return _coerce_medium_pairs(parsed.get("medium")), _coerce_hard_list(parsed.get("hard"))


def generate_tiered_queries(
    client: EndpointClient,
    *,
    attributes_json: str,
    caption: str,
    hard_only: bool,
    max_tokens: int = 500,
    temperature: float = 0.7,
    top_p: float = 0.9,
    response_retries: int = 0,
    response_retry_backoff_s: float = 0.0,
    logger: logging.Logger | None = None,
) -> tuple[list[QueryPair], list[str]]:
    """
    Run the legacy query-generation LLM op for one person.

    Args:
        client: A text LLM endpoint client.
        attributes_json: The person's attributes rendered as a JSON string.
        caption: The person's natural-language caption.
        hard_only: When ``True`` use the ``PROMPT_V3_HARD_ONLY`` prompt (medium is
            template-generated upstream); otherwise the ``PROMPT_V3`` prompt that
            yields both medium and hard queries.
        max_tokens: Max output tokens (legacy default 500).
        temperature: Sampling temperature (legacy default 0.7).
        top_p: Nucleus sampling value.

    Returns:
        ``(medium, hard)``; ``medium`` is empty when ``hard_only`` is ``True``.
    """
    caption_text = caption.strip() if caption and caption.strip() else "No caption available"
    prompt = render_query_prompt(
        attributes=attributes_json, caption=caption_text, hard_only=hard_only
    )
    request = ChatRequest(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    def validation_error(raw: str) -> str | None:
        medium, hard = parse_query_response(raw)
        missing: list[str] = []
        if not hard_only and not medium:
            missing.append("medium")
        if not hard:
            missing.append("hard")
        if missing:
            return f"missing usable {', '.join(missing)} queries"
        return None

    raw = generate_with_response_retries(
        client,
        request,
        validate_response=validation_error,
        stage="pas:tiered_queries",
        retries=response_retries,
        retry_backoff_s=response_retry_backoff_s,
        logger=logger,
    )
    medium, hard = parse_query_response(raw)
    if hard_only:
        return [], hard
    return medium, hard


def attributes_to_json(legacy_attributes: dict[str, object]) -> str:
    """Render the legacy flat attribute dict as the prompt's JSON block."""
    return json.dumps(legacy_attributes, indent=2)


def filter_query_pairs(pairs: Iterable[QueryPair]) -> list[QueryPair]:
    """Drop pairs whose query text is empty after stripping."""
    return [(q.strip(), used) for q, used in pairs if q and q.strip()]


@dataclass(frozen=True)
class LlmQueryParams:
    """Tunables for the LLM tiered-query builder (legacy-aligned defaults)."""

    use_template_for_medium: bool = False
    easy_count: int = 2
    medium_count: int = 2
    stop_words: tuple[str, ...] = field(default_factory=tuple)
    max_tokens: int = 500
    temperature: float = 0.7
    top_p: float = 0.9


class LlmQuerySetBuilder:
    """
    Build a :class:`QuerySet` per person using the legacy query-generation LLM op.

    Easy queries are always template-generated. Medium and hard queries come from
    the text LLM (``PROMPT_V3``); when ``use_template_for_medium`` is set the LLM
    produces hard queries only (``PROMPT_V3_HARD_ONLY``) and medium queries are
    template-generated. If the LLM yields no hard queries, the builder falls back
    to the deterministic upstream/caption-derived hard queries so the contract is
    never left empty.

    Instances are callable as ``builder(attributes, upstream_hard) -> QuerySet``
    so they can be dropped into the same seam as the model-free assembler.
    """

    def __init__(
        self,
        client: EndpointClient,
        params: LlmQueryParams | None = None,
        *,
        response_retries: int = 0,
        response_retry_backoff_s: float = 0.0,
        logger: logging.Logger | None = None,
    ) -> None:
        """Store the LLM client and (optional) generation params."""
        self._client = client
        self._params = params or LlmQueryParams()
        self._response_retries = response_retries
        self._response_retry_backoff_s = response_retry_backoff_s
        self._logger = logger

    def __call__(
        self,
        attributes: PersonAttributes,
        upstream_hard: Iterable[str] = (),
    ) -> QuerySet:
        """Generate a tiered ``QuerySet`` for one person via the LLM."""
        params = self._params
        easy = build_easy_queries(attributes, count=params.easy_count, stop_words=params.stop_words)
        attributes_json = attributes_to_json(attributes_to_legacy_dict(attributes))
        caption = attributes.natural_caption or ""
        medium_pairs, hard = generate_tiered_queries(
            self._client,
            attributes_json=attributes_json,
            caption=caption,
            hard_only=params.use_template_for_medium,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            response_retries=self._response_retries,
            response_retry_backoff_s=self._response_retry_backoff_s,
            logger=self._logger,
        )
        if params.use_template_for_medium:
            medium = build_medium_queries(
                attributes, count=params.medium_count, stop_words=params.stop_words
            )
        else:
            medium = filter_query_pairs(medium_pairs)[: params.medium_count]
        if not hard:
            hard = assemble_query_set(
                attributes,
                hard_queries=upstream_hard,
                easy_count=params.easy_count,
                medium_count=params.medium_count,
                stop_words=params.stop_words,
            ).hard
        return QuerySet(easy=easy, medium=medium, hard=hard)


__all__ = [
    "LlmQueryParams",
    "LlmQuerySetBuilder",
    "attributes_to_json",
    "filter_query_pairs",
    "generate_tiered_queries",
    "parse_query_response",
]
