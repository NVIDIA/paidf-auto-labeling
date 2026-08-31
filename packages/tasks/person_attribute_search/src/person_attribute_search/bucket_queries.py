# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Chunk-level 3-bucket retrieval query generation (legacy ``query_generation.py``
parity).

The legacy pipeline emitted ``chunk.queries`` as three buckets — ``PAS``,
``Anomaly``, and ``Caption`` — produced by a single text-LLM call grounded in
the scene/dense captions, the voted anomaly categories, and the per-person
attributes (``script/utils/query_generation.py``). UPA's per-person tiered
generator (:mod:`person_attribute_search.llm_queries`) only fills the ``PAS``
bucket, so the ``Anomaly`` and ``Caption`` buckets were previously empty.

This module ports that chunk-level generator: it builds the textual context from
captions + anomaly categories and asks a text LLM for all three buckets, each as
``[query, evidence]`` pairs. The PAS task keeps its existing person-attribute
queries for the ``PAS`` bucket and uses this module to fill ``Anomaly`` and
``Caption``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from typing import Any

from core import ScenePaths, read_json
from core.llm.json_extract import extract_json_object
from core.model_clients import ChatRequest, EndpointClient

from person_attribute_search.annotation_schema_adapter import (
    build_anomaly_gt,
    build_captions,
    detect_voter_model,
)
from person_attribute_search.prompts import load_prompt
from person_attribute_search.response_retry import generate_with_response_retries
from person_attribute_search.sources import find_sidecar

QUERY_BUCKETS: tuple[str, ...] = ("PAS", "Anomaly", "Caption")

#: Prompt asset stem under ``data/prompts/`` for the 3-bucket generator.
#: Unversioned by design: git is the source of truth for prompt history.
BUCKET_QUERIES_PROMPT = "bucket_queries"


def _normalize_query_item(item: Any) -> list[str] | None:
    """Coerce one parsed query entry into a ``[query, evidence]`` pair.

    Accepts a bare string, a ``[query, evidence]`` list/tuple, or a dict with
    ``query``/``text`` and ``source``/``evidence`` keys. Returns ``None`` when no
    non-empty query text is present.
    """
    if isinstance(item, str):
        query = item.strip()
        return [query, ""] if query else None
    if isinstance(item, (list, tuple)) and item:
        query = str(item[0]).strip()
        source = str(item[1]).strip() if len(item) > 1 else ""
        return [query, source] if query else None
    if isinstance(item, dict):
        query = str(item.get("query") or item.get("text") or "").strip()
        source = str(item.get("source") or item.get("evidence") or "").strip()
        return [query, source] if query else None
    return None


def normalize_buckets(parsed: Any, *, per_bucket: int = 5) -> dict[str, list[list[str]]]:
    """Normalize a parsed LLM response into capped ``{bucket: [[q, ev], ...]}``.

    Unknown buckets are ignored and each known bucket is capped at ``per_bucket``
    entries, mirroring the legacy normalizer.
    """
    if not isinstance(parsed, dict):
        return {bucket: [] for bucket in QUERY_BUCKETS}
    out: dict[str, list[list[str]]] = {}
    for bucket in QUERY_BUCKETS:
        pairs: list[list[str]] = []
        items = parsed.get(bucket)
        if not isinstance(items, (list, tuple)):
            items = []
        for item in items:
            normalized = _normalize_query_item(item)
            if normalized is not None:
                pairs.append(normalized)
        out[bucket] = pairs[:per_bucket]
    return out


def _caption_text(block: Any, key_names: tuple[str, ...]) -> str:
    """Extract caption text from a legacy ``{raw, parsed}`` block or a string."""
    if isinstance(block, str):
        return block.strip()
    if not isinstance(block, dict):
        return ""
    parsed = block.get("parsed")
    if isinstance(parsed, dict):
        for key in key_names:
            value = parsed.get(key)
            if value:
                return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    raw = block.get("raw")
    return str(raw).strip() if raw else ""


def build_query_context(
    scene_caption: Any,
    dense_caption: Any,
    anomaly_gt: dict[str, Any] | None = None,
) -> str:
    """Assemble the textual context used for chunk-level query generation.

    Combines scene context, the voted anomaly categories (or an explicit
    "use normal behavior" instruction when none fired), and the dense caption,
    matching the legacy ``build_query_context``.
    """
    parts: list[str] = []
    scene_text = _caption_text(scene_caption, ("scene_caption", "scene caption"))
    dense_text = _caption_text(dense_caption, ("dense_caption", "dense caption", "dense"))

    if scene_text:
        parts.append(f"Scene context: {scene_text}")

    if anomaly_gt is not None:
        voted = anomaly_gt.get("voted_categories") or []
        if voted:
            parts.append(
                "Anomaly categories present (multi-model majority): "
                + ", ".join(str(v) for v in voted)
            )
        else:
            parts.append(
                "Anomaly categories present (multi-model majority): none; "
                "use normal visible behavior for Anomaly queries."
            )

    if dense_text:
        parts.append(f"Dense caption:\n{dense_text}")

    return "\n\n".join(parts)


def generate_bucket_queries(
    client: EndpointClient,
    *,
    context_text: str,
    people: Sequence[dict[str, Any]] | Iterable[dict[str, Any]],
    per_bucket: int = 5,
    max_tokens: int = 800,
    temperature: float = 0.5,
    top_p: float = 0.9,
    response_retries: int = 0,
    response_retry_backoff_s: float = 0.0,
    logger: logging.Logger | None = None,
) -> dict[str, list[list[str]]]:
    """
    Run the legacy chunk-level query-generation LLM op.

    Args:
        client: A text LLM endpoint client.
        context_text: Scene/anomaly + dense-caption context from
            :func:`build_query_context`.
        people: Per-person attribute dicts (the chunk's ``pas.people``).
        per_bucket: Max queries kept per bucket (legacy default 5).
        max_tokens: Max output tokens.
        temperature: Sampling temperature (legacy default 0.5).
        top_p: Nucleus sampling value.

    Returns:
        ``{"PAS": [...], "Anomaly": [...], "Caption": [...]}`` with each entry a
        ``[query, evidence]`` pair. Buckets are empty when the response is
        unparseable.
    """
    people_list = list(people)
    people_text = (
        json.dumps(people_list, indent=2, ensure_ascii=False)
        if people_list
        else "No person attributes available."
    )
    prompt_template = load_prompt(BUCKET_QUERIES_PROMPT).replace(
        "{{bucket_count}}", str(per_bucket)
    )
    prompt = (
        prompt_template
        + "\n\nScene/anomaly and dense caption context:\n"
        + context_text
        + "\n\nPerson attributes:\n"
        + people_text
    )
    request = ChatRequest(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    def validation_error(raw: str) -> str | None:
        normalized = normalize_buckets(extract_json_object(raw), per_bucket=per_bucket)
        missing = [bucket for bucket in ("Anomaly", "Caption") if not normalized[bucket]]
        if missing:
            return f"missing usable {', '.join(missing)} bucket(s)"
        return None

    raw = generate_with_response_retries(
        client,
        request,
        validate_response=validation_error,
        stage="pas:bucket_queries",
        retries=response_retries,
        retry_backoff_s=response_retry_backoff_s,
        logger=logger,
    )
    return normalize_buckets(extract_json_object(raw), per_bucket=per_bucket)


def build_chunk_query_buckets(
    client: EndpointClient | None,
    *,
    paths: ScenePaths,
    people: Sequence[dict[str, Any]],
    flat_queries: list[str],
    video_captions_sidecars: tuple[str, ...],
    anomaly_items_sidecars: tuple[str, ...],
    model_name: str,
    per_bucket: int = 5,
    max_tokens: int = 800,
    temperature: float = 0.5,
    top_p: float = 0.9,
    response_retries: int = 0,
    response_retry_backoff_s: float = 0.0,
    logger: logging.Logger | None = None,
    optional_failures: list[str] | None = None,
) -> dict[str, list[list[str]]] | None:
    """
    Build legacy 3-bucket chunk queries around the PAS flat query list.

    ``None`` means bucket generation is disabled. When enabled, the PAS bucket
    always mirrors ``flat_queries`` so existing PAS behavior is unchanged; model
    output only fills the Anomaly and Caption buckets.
    """
    if client is None:
        return None

    pas_only_buckets = {
        "PAS": [[query, ""] for query in flat_queries],
        "Anomaly": [],
        "Caption": [],
    }
    video_captions_path = find_sidecar(paths, video_captions_sidecars)
    anomaly_items_path = find_sidecar(paths, anomaly_items_sidecars)
    if video_captions_path is None and anomaly_items_path is None:
        message = "bucket_query_context_missing: no captions/anomaly artifacts found"
        if optional_failures is not None:
            optional_failures.append(message)
        if logger is not None:
            logger.info(
                "Bucket query generation enabled but no captions/anomaly artifacts "
                "found under %s; emitting PAS bucket only.",
                paths.sidecars_dir,
            )
            logger.info(
                "PAS bucket queries: PAS=%d Anomaly=%d Caption=%d",
                len(pas_only_buckets["PAS"]),
                0,
                0,
            )
        return pas_only_buckets

    anomaly_items = read_json(anomaly_items_path) if anomaly_items_path else None
    video_captions = read_json(video_captions_path) if video_captions_path else None

    voter = detect_voter_model(paths.sidecars_dir) or model_name or "vlm"
    anomaly_gt = build_anomaly_gt(anomaly_items, voter)
    scene_caption, dense_caption = build_captions(
        video_captions,
        has_anomaly=bool(anomaly_gt["voted_categories"]),
    )
    context_text = build_query_context(scene_caption, dense_caption, anomaly_gt)

    generation_failed = False
    try:
        generated = generate_bucket_queries(
            client,
            context_text=context_text,
            people=people,
            per_bucket=per_bucket,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            response_retries=response_retries,
            response_retry_backoff_s=response_retry_backoff_s,
            logger=logger,
        )
    except Exception as exc:  # noqa: BLE001 - fail soft so PAS still writes
        generation_failed = True
        if optional_failures is not None:
            optional_failures.append(f"bucket_query_generation_failed: {exc}")
        if logger is not None:
            logger.warning("Bucket query generation failed; PAS bucket only: %s", exc)
        generated = {"PAS": [], "Anomaly": [], "Caption": []}
    if not generation_failed and not generated.get("Anomaly") and not generated.get("Caption"):
        message = "bucket_query_generation_failed: LLM returned no usable Anomaly/Caption queries"
        if optional_failures is not None and message not in optional_failures:
            optional_failures.append(message)
        if logger is not None:
            logger.warning("Bucket query generation produced no usable optional buckets")

    buckets: dict[str, list[list[str]]] = {
        "PAS": [[query, ""] for query in flat_queries],
        "Anomaly": generated.get("Anomaly", []),
        "Caption": generated.get("Caption", []),
    }
    if logger is not None:
        logger.info(
            "PAS bucket queries: PAS=%d Anomaly=%d Caption=%d",
            len(buckets["PAS"]),
            len(buckets["Anomaly"]),
            len(buckets["Caption"]),
        )
    return buckets


__all__ = [
    "BUCKET_QUERIES_PROMPT",
    "QUERY_BUCKETS",
    "build_chunk_query_buckets",
    "build_query_context",
    "generate_bucket_queries",
    "normalize_buckets",
]
