# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gemini generateContent compatible multimodal client."""

from __future__ import annotations

import logging
import urllib.parse
from typing import Any

from core.cost_performance import timed_model_call
from core.model_clients.backends import GEMINI_API_KEY_ENV
from core.model_clients.http import post_json
from core.model_clients.openai_compatible import resolve_api_key
from core.model_clients.reasoning import DEFAULT_PARSER, ReasoningParser, strip_think_blocks
from core.model_clients.types import ChatRequest, EndpointError


class GeminiClient:
    """Client for Gemini ``generateContent`` compatible endpoints."""

    def __init__(
        self,
        *,
        endpoint_url: str | None,
        model: str,
        timeout_s: float = 120.0,
        retries: int = 2,
        retry_backoff_s: float = 1.0,
        parser: ReasoningParser = DEFAULT_PARSER,
        error_cls: type[RuntimeError] = EndpointError,
        logger: logging.Logger | None = None,
    ) -> None:
        if parser != DEFAULT_PARSER:
            raise NotImplementedError(
                f"GeminiClient does not support parser={parser!r}; reasoning control is only "
                f"implemented for openai-compatible endpoints. Use {DEFAULT_PARSER!r}."
            )
        self.endpoint_url = endpoint_url
        self.model = model
        self.api_key_env = GEMINI_API_KEY_ENV
        self.parser = parser
        self.api_key = resolve_api_key(GEMINI_API_KEY_ENV, empty_value="")
        self.timeout_s = timeout_s
        self.retries = retries
        self.retry_backoff_s = retry_backoff_s
        self.error_cls = error_cls
        self.logger = logger
        self.last_call_metadata: dict[str, Any] = {}

    def generate(self, request: ChatRequest) -> str:
        """Generate text for ``request``."""
        parts: list[dict[str, Any]] = [{"text": request.prompt}]
        for media in request.media:
            parts.append(
                {
                    "inline_data": {
                        "mime_type": media.mime_type,
                        "data": media.data_base64,
                    }
                }
            )
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "maxOutputTokens": request.max_tokens,
                "temperature": request.temperature,
                "topP": request.top_p,
            },
        }
        if request.system_prompt:
            payload["system_instruction"] = {"parts": [{"text": request.system_prompt}]}
        timer = timed_model_call(
            kind="vlm" if request.media else "llm",
            provider="gemini",
            model=self.model,
            endpoint_url=self.endpoint_url,
        )
        response_meta: dict[str, Any] = {}
        try:
            data = post_json(
                gemini_generate_url(self.endpoint_url, self.model, self.api_key),
                payload,
                headers={"Content-Type": "application/json"},
                timeout_s=self.timeout_s,
                retries=self.retries,
                retry_backoff_s=self.retry_backoff_s,
                error_cls=self.error_cls,
                response_meta=response_meta,
                logger=self.logger,
            )
        except Exception:
            timer.record(success=False, retry_count=int(response_meta.get("retry_count", 0)))
            raise
        try:
            candidate = data["candidates"][0]
            parts_out = candidate["content"]["parts"]
        except (KeyError, IndexError, TypeError) as exc:
            timer.record(
                success=False,
                retry_count=int(response_meta.get("retry_count", 0)),
                usage=data.get("usageMetadata"),
            )
            raise self.error_cls("Gemini response missing candidates[0].content.parts") from exc
        self.last_call_metadata = {
            "provider": "gemini",
            "model": self.model,
            "endpoint_url": self.endpoint_url,
            "finish_reason": candidate.get("finishReason") if isinstance(candidate, dict) else None,
            "usage": data.get("usageMetadata"),
            "safety_ratings": candidate.get("safetyRatings")
            if isinstance(candidate, dict)
            else None,
        }
        timer.record(
            success=True,
            retry_count=int(response_meta.get("retry_count", 0)),
            usage=data.get("usageMetadata"),
        )
        if not isinstance(parts_out, list):
            return strip_think_blocks(str(parts_out))
        text = "".join(str(part.get("text", "")) for part in parts_out if isinstance(part, dict))
        return strip_think_blocks(text)


def gemini_generate_url(endpoint_url: str | None, model: str, api_key: str | None) -> str:
    """Build the Gemini generateContent URL for a model and optional API key."""
    if endpoint_url and "{model}" in endpoint_url:
        url = endpoint_url.format(model=urllib.parse.quote(model, safe=""))
    elif endpoint_url and endpoint_url.rstrip("/").endswith(":generateContent"):
        url = endpoint_url
    else:
        base = (endpoint_url or "https://generativelanguage.googleapis.com").rstrip("/")
        url = f"{base}/v1beta/models/{urllib.parse.quote(model, safe='')}:generateContent"
    if not api_key:
        return url
    separator = "&" if urllib.parse.urlsplit(url).query else "?"
    return f"{url}{separator}key={urllib.parse.quote(api_key)}"
