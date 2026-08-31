# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI chat-completions compatible multimodal client."""

from __future__ import annotations

import logging
import os
import time
import urllib.parse
from typing import Any

from openai import OpenAI

from core.cost_performance import timed_model_call
from core.model_clients.backends import NVIDIA_API_KEY_ENV
from core.model_clients.reasoning import (
    DEFAULT_PARSER,
    ReasoningParser,
    extra_body_for_parser,
    strip_think_blocks,
)
from core.model_clients.types import ChatRequest, EndpointError


class OpenAICompatibleClient:
    """Client for OpenAI chat-completions compatible endpoints."""

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
        if not endpoint_url:
            raise ValueError(
                "openai-compatible endpoints require an explicit endpoint_url "
                "(for example a local NIM base URL ending in /v1)."
            )
        self.endpoint_url = endpoint_url
        self.model = model
        self.api_key_env = NVIDIA_API_KEY_ENV
        self.timeout_s = timeout_s
        self.retries = retries
        self.retry_backoff_s = retry_backoff_s
        self.parser = parser
        self._parser_extra_body = extra_body_for_parser(parser)
        self.error_cls = error_cls
        self.logger = logger
        self.last_call_metadata: dict[str, Any] = {}
        self._client: Any = create_openai_client(
            api_key=resolve_api_key(NVIDIA_API_KEY_ENV),
            endpoint_url=self.endpoint_url,
            timeout_s=self.timeout_s,
        )

    def generate(self, request: ChatRequest) -> str:
        """Generate text for ``request``."""
        last_error: Exception | None = None
        timer = timed_model_call(
            kind="vlm" if request.media else "llm",
            provider="openai-compatible",
            model=self.model,
            endpoint_url=self.endpoint_url,
        )
        for attempt in range(self.retries + 1):
            try:
                response = create_openai_chat_completion(
                    self._client,
                    model=self.model,
                    messages=self._messages(request),
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    extra_body=self._parser_extra_body,
                )
                text = self._response_text(response)
                timer.record(
                    success=True,
                    retry_count=attempt,
                    usage=getattr(response, "usage", None),
                )
                return text
            except Exception as exc:
                if not is_retryable_openai_error(exc) or attempt >= self.retries:
                    timer.record(success=False, retry_count=attempt)
                    raise
                last_error = exc
                sleep_s = self.retry_backoff_s * (attempt + 1)
                if self.logger is not None:
                    self.logger.warning(
                        "Transient LLM endpoint error "
                        "(provider=openai-compatible model=%s attempt %d/%d): %s. "
                        "Retrying in %.1fs",
                        self.model,
                        attempt + 1,
                        self.retries + 1,
                        exc,
                        sleep_s,
                    )
                time.sleep(sleep_s)
        timer.record(success=False, retry_count=self.retries)
        raise self.error_cls("OpenAI-compatible request failed") from last_error

    def _response_text(self, response: Any) -> str:
        choices: Any = getattr(response, "choices", None)
        try:
            choice = choices[0]
            message = choice.message
            content = message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise self.error_cls(
                "OpenAI-compatible response missing choices[0].message.content"
            ) from exc
        usage = getattr(response, "usage", None)
        self.last_call_metadata = {
            "provider": "openai-compatible",
            "model": self.model,
            "endpoint_url": self.endpoint_url,
            "finish_reason": getattr(choice, "finish_reason", None),
            "usage": model_dump_or_value(usage),
        }
        return strip_think_blocks(content_text(content))

    @staticmethod
    def _messages(request: ChatRequest) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        content: list[dict[str, Any]] = []
        for media in request.media:
            if media.mime_type.startswith("video/"):
                content.append({"type": "video_url", "video_url": {"url": media.data_url}})
            else:
                content.append({"type": "image_url", "image_url": {"url": media.data_url}})
        content.append({"type": "text", "text": request.prompt})
        messages.append({"role": "user", "content": content})
        return messages


def openai_base_url(endpoint_url: str) -> str:
    """Normalize full chat-completions URLs to OpenAI SDK base URLs."""
    stripped = endpoint_url.rstrip("/")
    if stripped.endswith("/chat/completions"):
        stripped = stripped[: -len("/chat/completions")]
    return validate_openai_endpoint_url(stripped)


def validate_openai_endpoint_url(endpoint_url: str) -> str:
    """Validate an OpenAI-compatible endpoint URL before passing it to the SDK."""
    parsed_url = urllib.parse.urlparse(endpoint_url)
    if parsed_url.scheme.lower() not in {"http", "https"}:
        raise ValueError(f"Unsupported endpoint URL scheme: {parsed_url.scheme or '<missing>'}")
    if not parsed_url.netloc:
        raise ValueError("OpenAI-compatible endpoint URL requires a host")
    return endpoint_url


def create_openai_client(
    *,
    api_key: str,
    endpoint_url: str,
    timeout_s: float,
) -> Any:
    """Create an OpenAI SDK client with shared endpoint normalization."""
    return OpenAI(
        api_key=api_key,
        base_url=openai_base_url(endpoint_url),
        timeout=timeout_s,
        max_retries=0,
    )


def create_openai_chat_completion(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    extra_body: dict[str, Any] | None = None,
    response_format: dict[str, Any] | None = None,
    seed: int | None = None,
) -> Any:
    """Call OpenAI chat completions with shared option assembly."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
    }
    if extra_body is not None:
        kwargs["extra_body"] = extra_body
    if response_format is not None:
        kwargs["response_format"] = response_format
    if seed is not None:
        kwargs["seed"] = int(seed)
    return client.chat.completions.create(**kwargs)


def resolve_api_key(api_key_env: str, *, empty_value: str = "EMPTY") -> str:
    """Resolve an API key from ``api_key_env``, or return ``empty_value`` when unset."""
    value = os.getenv(api_key_env)
    return value if value else empty_value


def is_retryable_openai_error(exc: Exception) -> bool:
    """Return whether an OpenAI SDK exception is transient."""
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code in {408, 429} or 500 <= status_code <= 599
    if exc.__class__.__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "InternalServerError",
        "RateLimitError",
        "ReadTimeout",
        "RemoteProtocolError",
        "ServiceUnavailableError",
    }:
        return True
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
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


def content_text(content: object, *, list_separator: str = "") -> str:
    """Extract text from OpenAI-compatible message content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return list_separator.join(parts)
    return str(content)


def model_dump_or_value(value: object) -> object:
    """Return a pydantic-style model dump when available."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value
