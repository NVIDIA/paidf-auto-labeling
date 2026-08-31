# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured-output helpers for OpenAI-compatible LLM endpoints."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from core.llm.clients import call_chat_raw, ensure_v1
from core.llm.json_extract import extract_json_object_from_llm_text, parse_strict_json_object

StructuredObjectCaller = Callable[..., tuple[Any, str]]
_GENERIC_OBJECT_SCHEMA: dict[str, Any] = {"type": "object"}


def guided_json_enabled() -> bool:
    """Return whether NIM guided JSON should be attempted."""
    value = str(os.environ.get("LLM_DISABLE_GUIDED_JSON", "")).strip().lower()
    return value not in {"1", "true", "yes", "y", "on"}


def resolve_structured_output_mode(
    *,
    structured_output: str,
    base_url: str,
    model: str,
    invalid_fallback: str = "auto",
) -> str:
    """Resolve structured-output mode to ``auto``, ``nim``, ``openai``, or ``off``."""
    mode = str(structured_output or "").strip().lower()
    if mode not in {"auto", "nim", "openai", "off"}:
        mode = str(invalid_fallback or "auto").strip().lower()
        if mode not in {"auto", "nim", "openai", "off"}:
            mode = "auto"
    if mode == "auto":
        mode = "nim" if _is_nim_endpoint(base_url=base_url, model=model) else "openai"
    return mode


def build_structured_request_options(
    *,
    structured_output: str,
    base_url: str,
    model: str,
    guided_json_schema: dict[str, Any] | None = None,
    invalid_fallback: str = "auto",
) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
    """Return ``(mode, extra_body, response_format)`` for structured output."""
    mode = resolve_structured_output_mode(
        structured_output=structured_output,
        base_url=base_url,
        model=model,
        invalid_fallback=invalid_fallback,
    )
    if mode == "nim":
        schema = (
            guided_json_schema if isinstance(guided_json_schema, dict) else _GENERIC_OBJECT_SCHEMA
        )
        return mode, {"nvext": {"guided_json": schema}, "guided_json": schema}, None
    if mode == "openai":
        return mode, None, {"type": "json_object"}
    return mode, None, None


def call_chat_object_with_structured_fallback(
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
    guided_json_schema: dict[str, Any] | None = None,
    invalid_fallback: str = "auto",
    retry_stage: str = "",
    retry_dump_dir: str | None = None,
    api_key: str | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """Try structured JSON first, then raw JSON extraction."""
    mode, extra_body, response_format = build_structured_request_options(
        structured_output=str(structured_output or "auto"),
        base_url=base_url,
        model=model,
        guided_json_schema=guided_json_schema,
        invalid_fallback=invalid_fallback,
    )
    stage = str(retry_stage or "").strip() or "unspecified"
    dump_root = _dump_root(retry_dump_dir)

    try:
        retry_max_tokens_cap = int(os.environ.get("LLM_LENGTH_RETRY_MAX_TOKENS", "16384"))
    except (TypeError, ValueError):
        retry_max_tokens_cap = 32768
    retry_max_tokens_cap = max(int(max_tokens), retry_max_tokens_cap)

    n_retries = int(retries or 0)
    structured_max_tokens = int(max_tokens)
    raw_max_tokens = int(max_tokens)
    raw_text = ""

    if mode != "off":
        parsed, raw_text, structured_max_tokens = _try_json_attempts(
            phase="structured",
            base_url=base_url,
            model=model,
            messages=messages,
            timeout=timeout,
            max_tokens=structured_max_tokens,
            retry_max_tokens_cap=retry_max_tokens_cap,
            temperature=temperature,
            top_p=top_p,
            logger=logger,
            retries=n_retries,
            retry_backoff_s=retry_backoff_s,
            retry_stage=stage,
            seed=seed,
            api_key=api_key,
            extra_body=extra_body,
            response_format=response_format,
            dump_root=dump_root,
        )
        if isinstance(parsed, dict):
            return parsed, raw_text

    parsed, fallback_raw, raw_max_tokens = _try_json_attempts(
        phase="raw",
        base_url=base_url,
        model=model,
        messages=messages,
        timeout=timeout,
        max_tokens=raw_max_tokens,
        retry_max_tokens_cap=retry_max_tokens_cap,
        temperature=temperature,
        top_p=top_p,
        logger=logger,
        retries=n_retries,
        retry_backoff_s=retry_backoff_s,
        retry_stage=stage,
        seed=seed,
        api_key=api_key,
        extra_body=None,
        response_format=None,
        dump_root=dump_root,
    )
    _ = raw_max_tokens
    if isinstance(parsed, dict):
        return parsed, fallback_raw
    return None, fallback_raw or raw_text


def call_required_json_object_with_fallback(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    timeout: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    logger: Any,
    error_type: type[RuntimeError],
    retries: int = 3,
    retry_backoff_s: float = 5.0,
    structured_output: str = "auto",
    seed: int | None = None,
    guided_json_schema: dict[str, Any] | None = None,
    retry_stage: str = "",
    api_key: str | None = None,
    no_parseable_message: str | None = None,
    non_object_message: str | None = None,
    caller: StructuredObjectCaller = call_chat_object_with_structured_fallback,
) -> dict[str, Any]:
    """Run structured/raw fallback and require a parsed JSON object."""
    parsed, raw = caller(
        base_url=base_url,
        model=model,
        messages=messages,
        timeout=int(timeout),
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        logger=logger,
        retries=int(retries),
        retry_backoff_s=float(retry_backoff_s),
        structured_output=str(structured_output),
        seed=seed,
        guided_json_schema=guided_json_schema,
        retry_stage=retry_stage,
        api_key=api_key,
    )
    if parsed is None:
        snippet = (raw or "")[:500]
        message = no_parseable_message or (
            "LLM call returned no parseable JSON object. Raw response (truncated): {snippet_repr}"
        )
        raise error_type(message.format(snippet=snippet, snippet_repr=repr(snippet)))
    if not isinstance(parsed, dict):
        message = non_object_message or "LLM returned a non-object payload: {type_name}"
        raise error_type(message.format(type_name=type(parsed).__name__))
    return cast(dict[str, Any], parsed)


def require_items_list(
    payload: dict[str, Any],
    *,
    error_type: type[RuntimeError],
    missing_message: str,
    empty_message: str,
) -> list[Any]:
    """Return ``payload['items']`` as a non-empty list."""
    items = payload.get("items")
    if not isinstance(items, list):
        raise error_type(missing_message.format(type_name=type(items).__name__))
    if not items:
        raise error_type(empty_message)
    return items


def _try_json_attempts(
    *,
    phase: str,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    timeout: int,
    max_tokens: int,
    retry_max_tokens_cap: int,
    temperature: float,
    top_p: float,
    logger: Any,
    retries: int,
    retry_backoff_s: float,
    retry_stage: str,
    seed: int | None,
    api_key: str | None,
    extra_body: dict[str, Any] | None,
    response_format: dict[str, Any] | None,
    dump_root: Path | None,
) -> tuple[dict[str, Any] | None, str, int]:
    current_max_tokens = int(max_tokens)
    last_raw = ""
    for attempt in range(retries + 1):
        response_meta: dict[str, Any] = {}
        raw_len = 0
        finish_reason = ""
        try:
            raw = call_chat_raw(
                base_url=base_url,
                model=model,
                messages=messages,
                timeout=timeout,
                max_tokens=current_max_tokens,
                temperature=temperature,
                top_p=top_p,
                extra_body=extra_body,
                response_format=response_format,
                seed=seed,
                logger=logger,
                retries=0,
                retry_stage=f"{retry_stage}:{phase}",
                response_meta=response_meta,
                api_key=api_key,
            ).strip()
            last_raw = raw
            _dump_attempt_raw(
                dump_root=dump_root,
                stage=retry_stage,
                phase=phase,
                attempt_idx=attempt,
                raw_text=raw,
            )
            raw_len = len(raw)
            finish_reason = str(response_meta.get("finish_reason") or "").strip().lower()
            parsed = parse_strict_json_object(raw) or extract_json_object_from_llm_text(raw)
            if isinstance(parsed, dict):
                return parsed, raw, current_max_tokens
            _warn_parse_failure(
                logger=logger,
                phase=phase,
                stage=retry_stage,
                model=model,
                attempt=attempt,
                retries=retries,
                raw_len=raw_len,
                finish_reason=finish_reason,
                empty=not bool(raw.strip()),
            )
        except Exception as exc:
            if logger is not None:
                logger.warning(
                    "%s JSON chat call raised (stage=%s model=%s attempt %d/%d): %s: %s",
                    phase.capitalize(),
                    retry_stage,
                    model,
                    attempt + 1,
                    retries + 1,
                    exc.__class__.__name__,
                    exc,
                )
        if (
            finish_reason == "length"
            and attempt < retries
            and attempt >= 1
            and current_max_tokens < retry_max_tokens_cap
        ):
            current_max_tokens = _expand_tokens(
                current_max_tokens=current_max_tokens,
                retry_max_tokens_cap=retry_max_tokens_cap,
                logger=logger,
                model=model,
                stage=retry_stage,
                phase=phase,
                attempt=attempt,
                retries=retries,
            )
        if attempt < retries:
            sleep_s = min(60.0, float(retry_backoff_s) * (2**attempt))
            if logger is not None:
                logger.warning(
                    "%s JSON parse retry scheduled "
                    "(stage=%s model=%s attempt %d/%d raw_len=%d finish_reason=%s); "
                    "retrying in %.1fs",
                    phase.capitalize(),
                    retry_stage,
                    model,
                    attempt + 1,
                    retries + 1,
                    raw_len,
                    finish_reason or "unknown",
                    sleep_s,
                )
            time.sleep(sleep_s)
        elif logger is not None:
            logger.warning(
                "%s JSON parse retries exhausted (stage=%s model=%s attempts=%d)",
                phase.capitalize(),
                retry_stage,
                model,
                retries + 1,
            )
    return None, last_raw, current_max_tokens


def _is_nim_endpoint(*, base_url: str, model: str) -> bool:
    normalized = ensure_v1(base_url).lower()
    if not normalized:
        return False
    if any(marker in normalized for marker in ("nvidia.com", "nvcf", "nim")):
        return True
    return str(model or "").strip().lower().startswith("nvidia/")


def _dump_root(retry_dump_dir: str | None) -> Path | None:
    value = str(retry_dump_dir or "").strip()
    if not value:
        return None
    return Path(value)


def _dump_attempt_raw(
    *,
    dump_root: Path | None,
    stage: str,
    phase: str,
    attempt_idx: int,
    raw_text: str,
) -> None:
    if dump_root is None:
        return
    try:
        dump_root.mkdir(parents=True, exist_ok=True)
        safe_stage = "".join(c if c.isalnum() or c in "._-" else "_" for c in stage)[:120]
        safe_phase = "".join(c if c.isalnum() or c in "._-" else "_" for c in phase)[:40]
        path = dump_root / f"{safe_stage}.{safe_phase}.attempt_{attempt_idx:02d}.txt"
        path.write_text(raw_text or "", encoding="utf-8")
    except OSError:
        return


def _warn_parse_failure(
    *,
    logger: Any,
    phase: str,
    stage: str,
    model: str,
    attempt: int,
    retries: int,
    raw_len: int,
    finish_reason: str,
    empty: bool,
) -> None:
    if logger is None:
        return
    logger.warning(
        "%s JSON parse failed "
        "(stage=%s model=%s attempt %d/%d reason=%s raw_len=%d finish_reason=%s)",
        phase.capitalize(),
        stage,
        model,
        attempt + 1,
        retries + 1,
        "empty_response" if empty else "non_empty_but_unparseable",
        raw_len,
        finish_reason or "unknown",
    )


def _expand_tokens(
    *,
    current_max_tokens: int,
    retry_max_tokens_cap: int,
    logger: Any,
    model: str,
    stage: str,
    phase: str,
    attempt: int,
    retries: int,
) -> int:
    next_tokens = min(
        retry_max_tokens_cap,
        max(current_max_tokens * 2, current_max_tokens + 1),
    )
    if next_tokens > current_max_tokens and logger is not None:
        logger.warning(
            "Length-triggered token expansion in %s JSON retry "
            "(stage=%s model=%s attempt %d/%d max_tokens=%d->%d)",
            phase,
            stage,
            model,
            attempt + 1,
            retries + 1,
            current_max_tokens,
            next_tokens,
        )
    return int(next_tokens)
