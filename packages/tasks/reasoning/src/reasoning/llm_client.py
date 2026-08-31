# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DAFT-specific LLM compatibility helpers.

Shared transport, retry, structured-output fallback, and JSON extraction live
in ``core.llm``. This module keeps DAFT's legacy import path stable and owns
only DAFT-specific schema defaults.
"""

from __future__ import annotations

import json
import os
from typing import Any

from core.llm.clients import (
    call_chat_raw,
    ensure_v1,
    get_api_key,
    get_llm_api_key,
    get_vlm_api_key,
    reset_openai_client_cache,
)
from core.llm.json_extract import (
    extract_json_object_from_llm_text,
    parse_strict_json_object,
    wrap_json_fence,
)
from core.llm.structured import (
    StructuredObjectCaller,
    build_structured_request_options,
    call_chat_object_with_structured_fallback,
    call_required_json_object_with_fallback,
    require_items_list,
    resolve_structured_output_mode,
)
from core.llm.structured import (
    guided_json_enabled as _core_guided_json_enabled,
)

MCQ_GUIDED_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "version": {"type": "number"},
        "video_id": {"type": "string"},
        "mcq": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "question": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}},
                    "answer": {"type": "string"},
                },
                "required": ["id", "question", "options", "answer"],
            },
        },
    },
    "required": ["mcq"],
}


def guided_json_enabled() -> bool:
    """DAFT compatibility wrapper around the task-neutral core toggle."""
    legacy = str(os.environ.get("MCQ_DISABLE_GUIDED_JSON", "")).strip().lower()
    if legacy in {"1", "true", "yes", "y", "on"}:
        return False
    return _core_guided_json_enabled()


def call_chat_structured_guided_json(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    timeout: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    logger: Any,
    retries: int = 3,
    structured_output: str = "auto",
) -> dict[str, Any] | None:
    """Return an MCQ JSON object using DAFT's MCQ guided schema."""
    mode, extra_body, response_format = build_structured_request_options(
        structured_output=str(structured_output or "auto"),
        base_url=base_url,
        model=model,
        guided_json_schema=MCQ_GUIDED_JSON_SCHEMA,
        invalid_fallback="auto",
    )
    if mode == "off":
        return None

    attempts: list[tuple[dict[str, Any] | None, dict[str, Any] | None]] = []
    if mode == "nim" and guided_json_enabled():
        attempts.append((extra_body, None))
        attempts.append((None, {"type": "json_object"}))
    elif mode == "openai":
        attempts.append((None, response_format))

    for attempt_extra_body, attempt_response_format in attempts:
        try:
            text = call_chat_raw(
                base_url=base_url,
                model=model,
                messages=messages,
                timeout=timeout,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                extra_body=attempt_extra_body,
                response_format=attempt_response_format,
                logger=logger,
                retries=int(retries or 0),
            ).strip()
        except Exception:
            continue
        parsed = _parse_mcq_payload(text)
        if parsed is not None:
            return parsed
    return None


def call_chat_json_with_structured_fallback(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    timeout: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    logger: Any,
    retries: int = 3,
    retry_backoff_s: float = 5.0,
    structured_output: str = "auto",
    seed: int | None = None,
    retry_stage: str = "",
    api_key: str | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """Shared DAFT MCQ JSON flow using the DAFT MCQ schema hint."""
    obj, raw_text = call_chat_object_with_structured_fallback(
        base_url=base_url,
        model=model,
        messages=messages,
        timeout=timeout,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        logger=logger,
        retries=retries,
        retry_backoff_s=retry_backoff_s,
        structured_output=structured_output,
        seed=seed,
        guided_json_schema=MCQ_GUIDED_JSON_SCHEMA,
        invalid_fallback="auto",
        retry_stage=retry_stage,
        api_key=api_key,
    )
    if isinstance(obj, dict) and isinstance(obj.get("mcq"), list):
        return obj, raw_text
    return None, raw_text


def _parse_mcq_payload(text: str) -> dict[str, Any] | None:
    if not text.strip():
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = extract_json_object_from_llm_text(text)
    if isinstance(parsed, dict) and isinstance(parsed.get("mcq"), list):
        return parsed
    return None


__all__ = [
    "MCQ_GUIDED_JSON_SCHEMA",
    "StructuredObjectCaller",
    "build_structured_request_options",
    "call_chat_json_with_structured_fallback",
    "call_chat_object_with_structured_fallback",
    "call_chat_raw",
    "call_chat_structured_guided_json",
    "call_required_json_object_with_fallback",
    "ensure_v1",
    "extract_json_object_from_llm_text",
    "get_api_key",
    "get_llm_api_key",
    "get_vlm_api_key",
    "guided_json_enabled",
    "parse_strict_json_object",
    "require_items_list",
    "reset_openai_client_cache",
    "resolve_structured_output_mode",
    "wrap_json_fence",
]
