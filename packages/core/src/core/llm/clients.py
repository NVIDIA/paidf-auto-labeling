# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared LLM endpoint clients and retry helpers."""

from __future__ import annotations

import os
import time
from typing import Any

from core.cost_performance import timed_model_call
from core.llm.types import ChatRequest
from core.model_clients import (
    content_text,
    create_openai_chat_completion,
    create_openai_client,
    strip_think_blocks,
)
from core.model_clients.openai_compatible import (
    is_retryable_openai_error,
    model_dump_or_value,
)

_OPENAI_CLIENT_CACHE: dict[tuple[str, str, int], Any] = {}
_DRY_RUN_STUB = '{"version": 1.0, "items": []}'
_NVIDIA_API_KEY_ENV = "NVIDIA_API_KEY"


class LlmEndpointError(RuntimeError):
    """Raised when an LLM endpoint response cannot be used."""


class OpenAICompatibleTextClient:
    """Small reusable client for OpenAI chat-completions compatible endpoints."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: int = 600,
        retries: int = 3,
        retry_backoff_s: float = 5.0,
        logger: Any = None,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries
        self.retry_backoff_s = retry_backoff_s
        self.logger = logger

    def generate(self, request: ChatRequest) -> str:
        """Generate text for ``request``."""
        return call_chat_raw(
            base_url=self.base_url,
            model=self.model,
            messages=request.messages,
            timeout=self.timeout,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            extra_body=request.extra_body,
            response_format=request.response_format,
            seed=request.seed,
            retries=self.retries,
            retry_backoff_s=self.retry_backoff_s,
            logger=self.logger,
            api_key=self.api_key,
        )


def ensure_v1(base_url: str) -> str:
    """Normalize an OpenAI-compatible endpoint base URL to end in ``/v1``."""
    value = str(base_url or "").strip().rstrip("/")
    if not value:
        return value
    return value if value.endswith("/v1") else f"{value}/v1"


def get_vlm_api_key() -> str:
    """API key for VLM endpoints. Always reads ``NVIDIA_API_KEY``."""
    return os.environ.get(_NVIDIA_API_KEY_ENV) or "EMPTY"


def get_llm_api_key() -> str:
    """API key for LLM endpoints. Always reads ``NVIDIA_API_KEY``."""
    return os.environ.get(_NVIDIA_API_KEY_ENV) or "EMPTY"


def get_api_key() -> str:
    """API key for OpenAI-compatible endpoints. Always reads ``NVIDIA_API_KEY``."""
    return os.environ.get(_NVIDIA_API_KEY_ENV) or "EMPTY"


def reset_openai_client_cache() -> None:
    """Drop cached OpenAI SDK clients."""
    _OPENAI_CLIENT_CACHE.clear()


def _get_openai_client(*, base_url: str, api_key: str | None, timeout: int) -> Any:
    api_key_str = api_key or ""
    key = (ensure_v1(base_url), api_key_str, int(timeout))
    cached = _OPENAI_CLIENT_CACHE.get(key)
    if cached is None:
        cached = create_openai_client(
            api_key=api_key_str,
            endpoint_url=key[0],
            timeout_s=float(timeout),
        )
        _OPENAI_CLIENT_CACHE[key] = cached
    return cached


def _dry_run_enabled() -> bool:
    return str(os.environ.get("LLM_DRY_RUN", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _message_reasoning_text(message: object) -> str:
    for attr in ("reasoning", "reasoning_content"):
        value = getattr(message, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    if hasattr(message, "model_dump"):
        try:
            dumped = message.model_dump()
        except Exception:
            return ""
        if isinstance(dumped, dict):
            for key in ("reasoning", "reasoning_content"):
                value = dumped.get(key)
                if isinstance(value, str) and value.strip():
                    return value
    return ""


def call_chat_raw(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    timeout: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    extra_body: dict[str, Any] | None = None,
    response_format: dict[str, Any] | None = None,
    seed: int | None = None,
    retries: int = 3,
    retry_backoff_s: float = 5.0,
    logger: Any = None,
    retry_stage: str = "",
    response_meta: dict[str, Any] | None = None,
    api_key: str | None = None,
) -> str:
    """Call an OpenAI-compatible chat endpoint and return message text."""
    stage = str(retry_stage or "").strip() or "llm"
    timer = timed_model_call(
        kind="llm",
        provider="openai-compatible",
        model=model,
        endpoint_url=base_url,
    )
    if _dry_run_enabled():
        if logger is not None:
            logger.info("[%s] LLM_DRY_RUN=1 set; returning stub response", stage)
        if isinstance(response_meta, dict):
            response_meta["finish_reason"] = "stop"
            response_meta["dry_run"] = True
        timer.record(success=True, retry_count=0)
        return _DRY_RUN_STUB

    client = _get_openai_client(
        base_url=base_url,
        api_key=api_key if api_key is not None else get_api_key(),
        timeout=timeout,
    )
    current_max_tokens = int(max_tokens)
    try:
        retry_max_tokens_cap = int(os.environ.get("LLM_LENGTH_RETRY_MAX_TOKENS", "16384"))
    except (TypeError, ValueError):
        retry_max_tokens_cap = 32768
    retry_max_tokens_cap = max(current_max_tokens, retry_max_tokens_cap)

    last_exc: Exception | None = None
    for attempt in range(int(retries) + 1):
        try:
            response = create_openai_chat_completion(
                client,
                model=model,
                messages=messages,
                max_tokens=current_max_tokens,
                temperature=temperature,
                top_p=top_p,
                extra_body=extra_body,
                response_format=response_format,
                seed=seed,
            )
            choice = response.choices[0] if getattr(response, "choices", None) else None
            message = getattr(choice, "message", None)
            finish_reason = getattr(choice, "finish_reason", None)
            usage = getattr(response, "usage", None)
            if isinstance(response_meta, dict):
                response_meta["finish_reason"] = finish_reason
                response_meta["usage"] = model_dump_or_value(usage)

            content = getattr(message, "content", None) if message is not None else None
            text = content_text(content, list_separator="\n").strip() if content is not None else ""
            text = strip_think_blocks(text)
            if not text.strip() and message is not None:
                text = strip_think_blocks(_message_reasoning_text(message))

            if not text.strip():
                _log_empty_response(
                    logger=logger,
                    response=response,
                    message=message,
                    model=model,
                    stage=stage,
                    finish_reason=finish_reason,
                )
                if attempt < int(retries):
                    current_max_tokens = _maybe_expand_token_budget(
                        current_max_tokens=current_max_tokens,
                        retry_max_tokens_cap=retry_max_tokens_cap,
                        finish_reason=finish_reason,
                        attempt=attempt,
                        logger=logger,
                        model=model,
                        stage=stage,
                        retries=retries,
                    )
                    _sleep_before_retry(
                        logger=logger,
                        retry_backoff_s=retry_backoff_s,
                        attempt=attempt,
                        message=(
                            "Empty chat content from stage=%s model=%s "
                            "(finish_reason=%r attempt %d/%d). Retrying in %.1fs"
                        ),
                        args=(stage, model, finish_reason, attempt + 1, int(retries) + 1),
                    )
                    continue
                if logger is not None:
                    logger.warning(
                        "Empty chat content retries exhausted "
                        "(stage=%s model=%s attempts=%d finish_reason=%r)",
                        stage,
                        model,
                        int(retries) + 1,
                        finish_reason,
                    )
            timer.record(success=True, retry_count=attempt, usage=usage)
            return text
        except Exception as exc:
            last_exc = exc
            retryable = is_retryable_openai_error(exc)
            if attempt >= int(retries) or not retryable:
                timer.record(success=False, retry_count=attempt)
                if logger is not None:
                    logger.warning(
                        "Chat call failed without retry "
                        "(stage=%s model=%s attempt %d/%d retryable=%s): %s",
                        stage,
                        model,
                        attempt + 1,
                        int(retries) + 1,
                        retryable,
                        exc,
                    )
                raise
            _sleep_before_retry(
                logger=logger,
                retry_backoff_s=retry_backoff_s,
                attempt=attempt,
                message="Transient error calling model=%s (attempt %d/%d): %s. Retrying in %.1fs",
                args=(model, attempt + 1, int(retries) + 1, exc),
            )
    if last_exc is not None:
        timer.record(success=False, retry_count=int(retries))
        raise last_exc
    timer.record(success=False, retry_count=int(retries))
    return ""


def _log_empty_response(
    *,
    logger: Any,
    response: object,
    message: object,
    model: str,
    stage: str,
    finish_reason: object,
) -> None:
    if logger is None:
        return
    usage = getattr(response, "usage", None)
    logger.warning(
        "Empty chat content (stage=%s model=%s finish_reason=%r usage=%r refusal=%r tool_calls=%s)",
        stage,
        model,
        finish_reason,
        model_dump_or_value(usage),
        getattr(message, "refusal", None) if message is not None else None,
        bool(getattr(message, "tool_calls", None)) if message is not None else False,
    )


def _maybe_expand_token_budget(
    *,
    current_max_tokens: int,
    retry_max_tokens_cap: int,
    finish_reason: object,
    attempt: int,
    logger: Any,
    model: str,
    stage: str,
    retries: int,
) -> int:
    if str(finish_reason or "").strip().lower() != "length":
        return current_max_tokens
    next_max_tokens = min(
        retry_max_tokens_cap,
        max(current_max_tokens * 2, current_max_tokens + 1),
    )
    if next_max_tokens > current_max_tokens and logger is not None:
        logger.warning(
            "Length-triggered retry token expansion "
            "(stage=%s model=%s attempt %d/%d max_tokens=%d->%d)",
            stage,
            model,
            attempt + 1,
            int(retries) + 1,
            current_max_tokens,
            next_max_tokens,
        )
    return int(next_max_tokens)


def _sleep_before_retry(
    *,
    logger: Any,
    retry_backoff_s: float,
    attempt: int,
    message: str,
    args: tuple[object, ...],
) -> None:
    sleep_s = min(60.0, float(retry_backoff_s) * (2**attempt))
    if logger is not None:
        logger.warning(message, *args, sleep_s)
    time.sleep(sleep_s)
