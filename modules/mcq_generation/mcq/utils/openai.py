# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from openai import OpenAI


def ensure_v1(base_url: str) -> str:
    b = str(base_url or "").strip().rstrip("/")
    if not b:
        return b
    return b if b.endswith("/v1") else f"{b}/v1"


def get_vlm_api_key() -> str:
    """API key for VLM endpoints. Prefers VLM_API_KEY, falls back to shared keys."""
    return (
        os.environ.get("VLM_API_KEY") or os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    )


def get_llm_api_key() -> str:
    """API key for LLM endpoints. Prefers LLM_API_KEY, falls back to shared keys."""
    return (
        os.environ.get("LLM_API_KEY") or os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    )


def get_api_key() -> str:
    """Generic key: tries VLM_API_KEY first, then LLM_API_KEY, then shared fallbacks."""
    return get_vlm_api_key() or get_llm_api_key() or "EMPTY"


def guided_json_enabled() -> bool:
    """
    Toggle for guided_json structured generation.

    For A/B comparisons (or endpoints that ignore guided_json), you can force-disable by setting
    MCQ_DISABLE_GUIDED_JSON=1 (or true/yes/on).
    """
    v = str(os.environ.get("MCQ_DISABLE_GUIDED_JSON", "")).strip().lower()
    return v not in {"1", "true", "yes", "y", "on"}


def _is_nim_endpoint(*, base_url: str, model: str) -> bool:
    """
    Best-effort: detect NVIDIA NIM / NVCF endpoints.
    """
    b = ensure_v1(base_url).lower()
    if not b:
        return False
    if any(s in b for s in ("nvidia.com", "nvcf", "nim")):
        return True
    # Local NIM often uses OpenAI-compatible base_url like http://0.0.0.0:8000/v1, but model
    # names commonly have the "nvidia/..." prefix.
    if str(model or "").strip().lower().startswith("nvidia/"):
        return True
    return False


def _response_format_unsupported(exc: Exception) -> bool:
    """
    Best-effort detect endpoints that don't accept response_format.
    """
    msg = str(exc).lower()
    return "response_format" in msg and any(
        s in msg
        for s in (
            "unrecognized",
            "unknown",
            "unsupported",
            "invalid",
            "extra fields not permitted",
            "additional properties are not allowed",
        )
    )


# Cache per (base_url_v1, model) to avoid repeatedly probing unsupported response_format.
_RESPONSE_FORMAT_SUPPORT_CACHE: Dict[Tuple[str, str], bool] = {}


# JSON schema for NVIDIA NIM structured generation (guided_json).
# See: https://docs.nvidia.com/nim/large-language-models/latest/structured-generation.html
MCQ_GUIDED_JSON_SCHEMA: Dict[str, Any] = {
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


def resolve_structured_output_mode(
    *,
    structured_output: str,
    base_url: str,
    model: str,
    invalid_fallback: str = "auto",
) -> str:
    """
    Resolve structured output mode into one of: auto/nim/openai/off.

    - Supports "auto" -> nim/openai by endpoint detection.
    - invalid_fallback controls behavior for unknown values.
    """
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
    guided_json_schema: Optional[Dict[str, Any]] = None,
    invalid_fallback: str = "auto",
) -> Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Shared structured-output request preparation.

    Returns:
      (resolved_mode, extra_body, response_format)
    """
    mode = resolve_structured_output_mode(
        structured_output=structured_output,
        base_url=base_url,
        model=model,
        invalid_fallback=invalid_fallback,
    )
    if mode == "nim":
        schema = guided_json_schema if isinstance(guided_json_schema, dict) else MCQ_GUIDED_JSON_SCHEMA
        return mode, {"nvext": {"guided_json": schema}}, None
    if mode == "openai":
        return mode, None, {"type": "json_object"}
    return mode, None, None


def call_chat_raw(
    *,
    base_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    timeout: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    extra_body: Optional[Dict[str, Any]] = None,
    response_format: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    retries: int = 3,
    retry_backoff_s: float = 5.0,
    logger: Any = None,
    retry_stage: str = "",
    response_meta: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    Call OpenAI-compatible chat.completions with best-effort retry.

    This reduces "leaked" clips when the endpoint temporarily resets/refuses connections.
    The OpenAI SDK already retries some requests, but we still see transient failures
    during long runs (e.g., LLM restarts).
    """
    client = OpenAI(
        api_key=api_key if api_key is not None else get_api_key(), base_url=ensure_v1(base_url), timeout=timeout
    )
    stage = str(retry_stage or "").strip() or "unspecified"
    current_max_tokens = int(max_tokens)
    try:
        # Upper bound for length-triggered dynamic max_tokens expansion.
        # Keep configurable to avoid unbounded growth on unstable backends.
        retry_max_tokens_cap = int(os.environ.get("MCQ_LENGTH_RETRY_MAX_TOKENS", "16384"))
    except Exception:
        retry_max_tokens_cap = 32768
    retry_max_tokens_cap = max(current_max_tokens, retry_max_tokens_cap)

    def _is_retryable(exc: Exception) -> bool:
        name = exc.__class__.__name__
        if name in {
            "APIConnectionError",
            "APITimeoutError",
            "RateLimitError",
            "InternalServerError",
            "ServiceUnavailableError",
            "ConnectError",
            "ReadTimeout",
            "RemoteProtocolError",
        }:
            return True
        msg = str(exc).lower()
        return any(
            s in msg
            for s in (
                "connection refused",
                "connection reset",
                "connection error",
                "timed out",
                "server disconnected",
                "503",
                "502",
                "504",
            )
        )

    create_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": int(current_max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
    }
    if extra_body:
        create_kwargs["extra_body"] = extra_body
    if response_format:
        create_kwargs["response_format"] = response_format
    if seed is not None:
        create_kwargs["seed"] = int(seed)

    last_exc: Optional[Exception] = None
    for attempt in range(int(retries) + 1):
        try:
            create_kwargs["max_tokens"] = int(current_max_tokens)
            resp = client.chat.completions.create(**create_kwargs)
            choice0 = resp.choices[0] if getattr(resp, "choices", None) else None
            message0 = getattr(choice0, "message", None)
            raw_content = getattr(message0, "content", None) if message0 is not None else None
            finish_reason = getattr(choice0, "finish_reason", None)
            if isinstance(response_meta, dict):
                response_meta["finish_reason"] = finish_reason

            text = ""
            if isinstance(raw_content, str):
                text = raw_content
            elif isinstance(raw_content, list):
                # Some OpenAI-compatible servers return content parts.
                parts: List[str] = []
                for part in raw_content:
                    if isinstance(part, dict):
                        t = part.get("text")
                        if isinstance(t, str) and t.strip():
                            parts.append(t)
                    else:
                        t = getattr(part, "text", None)
                        if isinstance(t, str) and t.strip():
                            parts.append(t)
                text = "\n".join(parts).strip()
            if not text.strip() and message0 is not None:
                # Some OpenAI-compatible servers (e.g., Qwen reasoning variants) may populate a separate
                # reasoning field while leaving message.content empty/null.
                for attr in ("reasoning", "reasoning_content"):
                    v = getattr(message0, attr, None)
                    if isinstance(v, str) and v.strip():
                        text = v
                        break
                if not text.strip() and hasattr(message0, "model_dump"):
                    try:
                        md = message0.model_dump()
                        if isinstance(md, dict):
                            for k in ("reasoning", "reasoning_content"):
                                v = md.get(k)
                                if isinstance(v, str) and v.strip():
                                    text = v
                                    break
                    except Exception:
                        pass

            if not text.strip() and logger is not None:
                usage = getattr(resp, "usage", None)
                usage_obj = usage.model_dump() if hasattr(usage, "model_dump") else usage
                refusal = getattr(message0, "refusal", None) if message0 is not None else None
                tool_calls = getattr(message0, "tool_calls", None) if message0 is not None else None
                logger.warning(
                    "Empty chat content (stage=%s model=%s finish_reason=%r usage=%r refusal=%r tool_calls=%s)",
                    stage,
                    model,
                    finish_reason,
                    usage_obj,
                    refusal,
                    bool(tool_calls),
                )
            if not text.strip() and attempt < int(retries):
                # Start token expansion from the second retry attempt.
                # attempt=0: first retry, keep original budget; attempt>=1: expand.
                if str(finish_reason or "").strip().lower() == "length" and attempt >= 1:
                    next_max_tokens = min(retry_max_tokens_cap, max(current_max_tokens * 2, current_max_tokens + 1))
                    if next_max_tokens > current_max_tokens:
                        try:
                            if logger is not None:
                                logger.warning(
                                    "Length-triggered retry token expansion (stage=%s model=%s attempt %d/%d max_tokens=%d->%d)",
                                    stage,
                                    model,
                                    attempt + 1,
                                    int(retries) + 1,
                                    current_max_tokens,
                                    next_max_tokens,
                                )
                        except Exception:
                            pass
                        current_max_tokens = int(next_max_tokens)
                sleep_s = min(60.0, float(retry_backoff_s) * (2**attempt))
                try:
                    if logger is not None:
                        logger.warning(
                            "Empty chat content from stage=%s model=%s (finish_reason=%r attempt %d/%d). "
                            "Retrying in %.1fs",
                            stage,
                            model,
                            finish_reason,
                            attempt + 1,
                            int(retries) + 1,
                            sleep_s,
                        )
                except Exception:
                    pass
                time.sleep(sleep_s)
                continue
            if not text.strip():
                try:
                    if logger is not None:
                        logger.warning(
                            "Empty chat content retries exhausted (stage=%s model=%s attempts=%d finish_reason=%r)",
                            stage,
                            model,
                            int(retries) + 1,
                            finish_reason,
                        )
                except Exception:
                    pass
            return text
        except Exception as e:
            last_exc = e
            if attempt >= int(retries) or not _is_retryable(e):
                try:
                    if logger is not None:
                        logger.warning(
                            "Chat call failed without retry (stage=%s model=%s attempt %d/%d retryable=%s): %s",
                            stage,
                            model,
                            attempt + 1,
                            int(retries) + 1,
                            _is_retryable(e),
                            e,
                        )
                except Exception:
                    pass
                raise
            sleep_s = min(60.0, float(retry_backoff_s) * (2**attempt))
            try:
                if logger is not None:
                    logger.warning(
                        "Transient error calling model=%s (attempt %d/%d): %s. Retrying in %.1fs",
                        model,
                        attempt + 1,
                        int(retries) + 1,
                        e,
                        sleep_s,
                    )
            except Exception:
                pass
            time.sleep(sleep_s)
    if last_exc is not None:
        raise last_exc
    return ""


def call_chat_structured_guided_json(
    *,
    base_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    timeout: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    logger: Any,
    retries: int = 3,
    structured_output: str = "auto",
) -> Optional[Dict[str, Any]]:
    """
    Structured output helper for MCQ JSON.

    structured_output:
      - "auto": NIM endpoints -> guided_json, otherwise -> response_format=json_object
      - "nim": guided_json (tries nvext first, then root guided_json)
      - "openai": response_format={"type":"json_object"}
      - "off": do not attempt structured generation (caller should use raw + extraction)

    Returns parsed MCQ dict or None on failure; caller should fall back to raw response +
    extract_json_object_from_llm_text() if this returns None.
    """
    mode, _, _ = build_structured_request_options(
        structured_output=str(structured_output or "auto"),
        base_url=base_url,
        model=model,
        invalid_fallback="auto",
    )
    if mode == "off":
        return None
    is_nim = _is_nim_endpoint(base_url=base_url, model=model)

    cache_key = (ensure_v1(base_url), str(model or ""))

    def _try_response_format_json_object() -> Optional[Dict[str, Any]]:
        if _RESPONSE_FORMAT_SUPPORT_CACHE.get(cache_key) is False:
            return None
        try:
            text = call_chat_raw(
                base_url=base_url,
                model=model,
                messages=messages,
                timeout=timeout,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                response_format={"type": "json_object"},
                logger=logger,
                retries=int(retries or 0),
            ).strip()
            if not text:
                return None
            try:
                obj = json.loads(text)
            except Exception:
                obj = extract_json_object_from_llm_text(text)
            if isinstance(obj, dict) and isinstance(obj.get("mcq"), list):
                _RESPONSE_FORMAT_SUPPORT_CACHE[cache_key] = True
                return obj
            return None
        except Exception as e:
            if _response_format_unsupported(e):
                _RESPONSE_FORMAT_SUPPORT_CACHE[cache_key] = False
            return None

    def _try_guided_json() -> Optional[Dict[str, Any]]:
        if not guided_json_enabled():
            return None
        try:
            # NIM structured generation has multiple wire formats depending on the gateway.
            # - Some deployments accept guided_json at the root level.
            # - NVIDIA integrate gateway requires it under nvext: {"nvext": {"guided_json": schema}}
            #
            # We try nvext first, then fall back to root-level guided_json.
            text = ""
            for extra in (
                {"nvext": {"guided_json": MCQ_GUIDED_JSON_SCHEMA}},
                {"guided_json": MCQ_GUIDED_JSON_SCHEMA},
            ):
                try:
                    text = call_chat_raw(
                        base_url=base_url,
                        model=model,
                        messages=messages,
                        timeout=timeout,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        extra_body=extra,
                        logger=logger,
                        retries=int(retries or 0),
                    ).strip()
                    if text:
                        break
                except Exception:
                    text = ""
                    continue
            if not text:
                return None
            try:
                obj = json.loads(text)
            except Exception:
                obj = extract_json_object_from_llm_text(text)
            if isinstance(obj, dict) and isinstance(obj.get("mcq"), list):
                return obj
            return None
        except Exception:
            return None

    if mode == "openai":
        return _try_response_format_json_object()

    if mode == "nim":
        obj = _try_guided_json()
        if obj is not None:
            return obj
        # Some NVIDIA gateways also accept OpenAI-style response_format=json_object.
        return _try_response_format_json_object() if is_nim else None

    return None


def call_chat_object_with_structured_fallback(
    *,
    base_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    timeout: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    logger: Any,
    retries: int = 3,
    retry_backoff_s: float = 5.0,
    structured_output: str = "auto",
    seed: Optional[int] = None,
    guided_json_schema: Optional[Dict[str, Any]] = None,
    invalid_fallback: str = "auto",
    retry_stage: str = "",
    retry_dump_dir: Optional[str] = None,
    api_key: Optional[str] = None,
    validator: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Generic object call flow:
    1) one structured call (if mode != off)
    2) fallback raw call if structured result is unavailable/unparseable

    When ``validator`` is provided, a parsed dict is only returned if the
    validator accepts it. Otherwise the attempt is treated like a parse
    failure and the retry loop continues. This lets callers reject
    structurally-valid-but-wrong-shape responses (e.g. a model that returns
    a video-summary object when the events schema was requested) without
    duplicating the retry/backoff/token-expansion logic.

    Returns:
      (parsed_obj_or_none, raw_text_from_last_call)
    """
    mode, extra_body, response_format = build_structured_request_options(
        structured_output=str(structured_output or "auto"),
        base_url=base_url,
        model=model,
        guided_json_schema=guided_json_schema,
        invalid_fallback=invalid_fallback,
    )
    stage = str(retry_stage or "").strip() or "unspecified"
    dump_root: Optional[Path] = None
    try:
        dump_dir_s = str(retry_dump_dir or "").strip()
        dump_root = Path(dump_dir_s) if dump_dir_s else None
    except Exception:
        dump_root = None

    def _dump_attempt_raw(*, phase: str, attempt_idx: int, raw_text: str) -> None:
        if dump_root is None:
            return
        try:
            dump_root.mkdir(parents=True, exist_ok=True)
            safe_stage = "".join(c if (c.isalnum() or c in "._-") else "_" for c in stage)[:120]
            safe_phase = "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(phase))[:40]
            p = dump_root / f"{safe_stage}.{safe_phase}.attempt_{attempt_idx:02d}.txt"
            p.write_text(raw_text or "", encoding="utf-8")
        except Exception:
            return

    try:
        retry_max_tokens_cap = int(os.environ.get("MCQ_LENGTH_RETRY_MAX_TOKENS", "16384"))
    except Exception:
        retry_max_tokens_cap = 32768
    retry_max_tokens_cap = max(int(max_tokens), retry_max_tokens_cap)
    structured_max_tokens = int(max_tokens)
    raw_max_tokens = int(max_tokens)

    n_retries = int(retries or 0)

    if mode != "off":
        last_raw_structured = ""
        for attempt in range(n_retries + 1):
            resp_meta: Dict[str, Any] = {}
            attempt_raw_len = 0
            attempt_empty = True
            attempt_finish_reason = ""
            try:
                raw_structured = call_chat_raw(
                    base_url=base_url,
                    model=model,
                    messages=messages,
                    timeout=timeout,
                    max_tokens=structured_max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    extra_body=extra_body,
                    response_format=response_format,
                    seed=seed,
                    logger=logger,
                    retries=0,
                    retry_stage=f"{stage}:structured",
                    response_meta=resp_meta,
                    api_key=api_key,
                ).strip()
                last_raw_structured = raw_structured
                _dump_attempt_raw(phase="structured", attempt_idx=attempt, raw_text=raw_structured)
                attempt_raw_len = len(raw_structured)
                attempt_empty = not bool(raw_structured.strip())
                attempt_finish_reason = str(resp_meta.get("finish_reason") or "").strip().lower()
                obj = parse_strict_json_object(raw_structured)
                if obj is None:
                    obj = extract_json_object_from_llm_text(raw_structured)
                parse_ok = isinstance(obj, dict)
                schema_ok = parse_ok and (validator is None or validator(obj))
                if schema_ok:
                    return obj, raw_structured
                try:
                    if logger is not None:
                        if parse_ok:
                            logger.warning(
                                "Structured JSON schema validation failed (stage=%s model=%s mode=%s attempt %d/%d raw_len=%d finish_reason=%s)",
                                stage,
                                model,
                                mode,
                                attempt + 1,
                                n_retries + 1,
                                attempt_raw_len,
                                (attempt_finish_reason or "unknown"),
                            )
                        else:
                            logger.warning(
                                "Structured JSON parse failed (stage=%s model=%s mode=%s attempt %d/%d reason=%s raw_len=%d finish_reason=%s)",
                                stage,
                                model,
                                mode,
                                attempt + 1,
                                n_retries + 1,
                                ("empty_response" if attempt_empty else "non_empty_but_unparseable"),
                                attempt_raw_len,
                                (attempt_finish_reason or "unknown"),
                            )
                except Exception:
                    pass
            except Exception as e:
                try:
                    if logger is not None:
                        logger.warning(
                            "Structured JSON chat call raised (stage=%s model=%s mode=%s attempt %d/%d): %s: %s",
                            stage,
                            model,
                            mode,
                            attempt + 1,
                            n_retries + 1,
                            e.__class__.__name__,
                            str(e),
                        )
                except Exception:
                    pass
            # If response is truncated by max_tokens, double token budget for the next attempt.
            if (
                str(resp_meta.get("finish_reason") or "").strip().lower() == "length"
                and attempt < n_retries
                and attempt >= 1
                and structured_max_tokens < retry_max_tokens_cap
            ):
                next_tokens = min(retry_max_tokens_cap, max(structured_max_tokens * 2, structured_max_tokens + 1))
                if next_tokens > structured_max_tokens:
                    try:
                        if logger is not None:
                            logger.warning(
                                "Length-triggered token expansion in structured JSON retry "
                                "(stage=%s model=%s mode=%s attempt %d/%d max_tokens=%d->%d)",
                                stage,
                                model,
                                mode,
                                attempt + 1,
                                n_retries + 1,
                                structured_max_tokens,
                                next_tokens,
                            )
                    except Exception:
                        pass
                    structured_max_tokens = int(next_tokens)
            if attempt < n_retries:
                sleep_s = min(60.0, float(retry_backoff_s) * (2**attempt))
                try:
                    if logger is not None:
                        logger.warning(
                            "Structured JSON parse retry scheduled (stage=%s model=%s mode=%s attempt %d/%d raw_len=%d finish_reason=%s); retrying in %.1fs",
                            stage,
                            model,
                            mode,
                            attempt + 1,
                            n_retries + 1,
                            attempt_raw_len,
                            (attempt_finish_reason or "unknown"),
                            sleep_s,
                        )
                except Exception:
                    pass
                time.sleep(sleep_s)
            else:
                try:
                    if logger is not None:
                        logger.warning(
                            "Structured JSON parse retries exhausted (stage=%s model=%s mode=%s attempts=%d)",
                            stage,
                            model,
                            mode,
                            n_retries + 1,
                        )
                except Exception:
                    pass

        raw_text = last_raw_structured
    else:
        raw_text = ""

    last_raw_fallback = raw_text
    for attempt in range(n_retries + 1):
        resp_meta: Dict[str, Any] = {}
        attempt_raw_len = 0
        attempt_empty = True
        attempt_finish_reason = ""
        try:
            raw_fallback = call_chat_raw(
                base_url=base_url,
                model=model,
                messages=messages,
                timeout=timeout,
                max_tokens=raw_max_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                logger=logger,
                retries=0,
                retry_stage=f"{stage}:raw",
                response_meta=resp_meta,
                api_key=api_key,
            ).strip()
            last_raw_fallback = raw_fallback
            _dump_attempt_raw(phase="raw", attempt_idx=attempt, raw_text=raw_fallback)
            attempt_raw_len = len(raw_fallback)
            attempt_empty = not bool(raw_fallback.strip())
            attempt_finish_reason = str(resp_meta.get("finish_reason") or "").strip().lower()
            obj2 = parse_strict_json_object(raw_fallback)
            if obj2 is None:
                obj2 = extract_json_object_from_llm_text(raw_fallback)
            parse_ok = isinstance(obj2, dict)
            schema_ok = parse_ok and (validator is None or validator(obj2))
            if schema_ok:
                return obj2, raw_fallback
            try:
                if logger is not None:
                    if parse_ok:
                        logger.warning(
                            "Raw JSON schema validation failed (stage=%s model=%s attempt %d/%d raw_len=%d finish_reason=%s)",
                            stage,
                            model,
                            attempt + 1,
                            n_retries + 1,
                            attempt_raw_len,
                            (attempt_finish_reason or "unknown"),
                        )
                    else:
                        logger.warning(
                            "Raw JSON parse failed (stage=%s model=%s attempt %d/%d reason=%s raw_len=%d finish_reason=%s)",
                            stage,
                            model,
                            attempt + 1,
                            n_retries + 1,
                            ("empty_response" if attempt_empty else "non_empty_but_unparseable"),
                            attempt_raw_len,
                            (attempt_finish_reason or "unknown"),
                        )
            except Exception:
                pass
        except Exception as e:
            try:
                if logger is not None:
                    logger.warning(
                        "Raw JSON chat call raised (stage=%s model=%s attempt %d/%d): %s: %s",
                        stage,
                        model,
                        attempt + 1,
                        n_retries + 1,
                        e.__class__.__name__,
                        str(e),
                    )
            except Exception:
                pass
        if (
            str(resp_meta.get("finish_reason") or "").strip().lower() == "length"
            and attempt < n_retries
            and attempt >= 1
            and raw_max_tokens < retry_max_tokens_cap
        ):
            next_tokens = min(retry_max_tokens_cap, max(raw_max_tokens * 2, raw_max_tokens + 1))
            if next_tokens > raw_max_tokens:
                try:
                    if logger is not None:
                        logger.warning(
                            "Length-triggered token expansion in raw JSON retry "
                            "(stage=%s model=%s attempt %d/%d max_tokens=%d->%d)",
                            stage,
                            model,
                            attempt + 1,
                            n_retries + 1,
                            raw_max_tokens,
                            next_tokens,
                        )
                except Exception:
                    pass
                raw_max_tokens = int(next_tokens)
        if attempt < n_retries:
            sleep_s = min(60.0, float(retry_backoff_s) * (2**attempt))
            try:
                if logger is not None:
                    logger.warning(
                        "Raw JSON parse retry scheduled (stage=%s model=%s attempt %d/%d raw_len=%d finish_reason=%s); retrying in %.1fs",
                        stage,
                        model,
                        attempt + 1,
                        n_retries + 1,
                        attempt_raw_len,
                        (attempt_finish_reason or "unknown"),
                        sleep_s,
                    )
            except Exception:
                pass
            time.sleep(sleep_s)
        else:
            try:
                if logger is not None:
                    logger.warning(
                        "Raw JSON parse retries exhausted (stage=%s model=%s attempts=%d)",
                        stage,
                        model,
                        n_retries + 1,
                    )
            except Exception:
                pass

    return None, last_raw_fallback


def call_chat_json_with_structured_fallback(
    *,
    base_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    timeout: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    logger: Any,
    retries: int = 3,
    retry_backoff_s: float = 5.0,
    structured_output: str = "auto",
    seed: Optional[int] = None,
    retry_stage: str = "",
    api_key: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Shared JSON call flow used by MCQ runners:
    1) try structured output first (unless mode=off)
    2) if unavailable/unparseable, fallback to raw text + best-effort JSON extraction

    Returns:
      (parsed_obj_or_none, raw_text)
    """
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


def classify_mcq_json_parse_failure(raw_text: str) -> str:
    text = str(raw_text or "").strip()
    if not text:
        return "empty_output"
    strict_obj = parse_strict_json_object(text)
    if isinstance(strict_obj, dict):
        if not isinstance(strict_obj.get("mcq"), list):
            return "missing_mcq_list"
        return "unexpected_parser_rejection"
    extracted_obj = extract_json_object_from_llm_text(text)
    if isinstance(extracted_obj, dict):
        if not isinstance(extracted_obj.get("mcq"), list):
            return "extracted_object_missing_mcq_list"
        return "unexpected_parser_rejection"
    if "```" in text:
        return "fenced_json_not_parseable"
    if "{" in text or "[" in text:
        return "json_not_parseable"
    return "non_json_output"


def parse_strict_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Strict parser: input must be a JSON object string.
    """
    t = str(text or "").strip()
    if not t:
        return None
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def extract_json_object_from_llm_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort extraction for noisy LLM text responses.
    """

    def _parse_candidate(candidate: str) -> Optional[Dict[str, Any]]:
        c = str(candidate or "").strip()
        if not c:
            return None
        try:
            obj = json.loads(c)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
        # Lenient pass for common minor issues from model output:
        # - trailing commas before '}' / ']'
        # - one or more missing closing braces at the end
        repaired = c
        for _ in range(3):
            newer = re.sub(r",\s*([}\]])", r"\1", repaired)
            if newer == repaired:
                break
            repaired = newer
        open_braces = repaired.count("{")
        close_braces = repaired.count("}")
        if open_braces > close_braces:
            repaired = repaired + ("}" * (open_braces - close_braces))
        try:
            obj = json.loads(repaired)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    t = str(text or "").strip()
    if not t:
        return None
    obj = parse_strict_json_object(t)
    if isinstance(obj, dict):
        return obj

    m = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        t,
        re.DOTALL,
    )
    if m:
        obj = _parse_candidate(m.group(1))
        if isinstance(obj, dict):
            return obj

    start = t.find("{")
    if start == -1:
        return None
    brace = 0
    in_string = False
    escaped = False
    for i, ch in enumerate(t[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            brace += 1
        elif ch == "}":
            brace -= 1
            if brace == 0:
                cand = t[start : i + 1]
                obj = _parse_candidate(cand)
                if isinstance(obj, dict):
                    return obj
                return None
    # Incomplete object: try best-effort repair once.
    tail = t[start:]
    obj = _parse_candidate(tail)
    if isinstance(obj, dict):
        return obj
    return None
