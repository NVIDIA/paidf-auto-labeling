# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral endpoint client factory."""

from __future__ import annotations

import logging

from core.model_clients.gemini import GeminiClient
from core.model_clients.openai_compatible import OpenAICompatibleClient
from core.model_clients.reasoning import DEFAULT_PARSER, ReasoningParser
from core.model_clients.types import EndpointClient, EndpointError, EndpointProvider


def create_endpoint_client(
    *,
    provider: EndpointProvider,
    endpoint_url: str | None,
    model: str,
    timeout_s: float,
    retries: int,
    retry_backoff_s: float,
    parser: ReasoningParser = DEFAULT_PARSER,
    error_cls: type[RuntimeError] = EndpointError,
    logger: logging.Logger | None = None,
) -> EndpointClient:
    """Create a shared endpoint client for ``provider``.

    ``parser`` controls hybrid-model reasoning: ``"instruct"`` (default) disables
    thinking for directly parseable responses; ``"reasoning"`` leaves it enabled.

    Authentication is fixed by provider:
    - ``openai-compatible`` reads ``NVIDIA_API_KEY``
    - ``gemini`` reads ``GEMINI_API_KEY``
    """
    if provider == "openai-compatible":
        return OpenAICompatibleClient(
            endpoint_url=endpoint_url,
            model=model,
            timeout_s=timeout_s,
            retries=retries,
            retry_backoff_s=retry_backoff_s,
            parser=parser,
            error_cls=error_cls,
            logger=logger,
        )
    if provider == "gemini":
        return GeminiClient(
            endpoint_url=endpoint_url,
            model=model,
            timeout_s=timeout_s,
            retries=retries,
            retry_backoff_s=retry_backoff_s,
            parser=parser,
            error_cls=error_cls,
            logger=logger,
        )
    raise ValueError(f"Unsupported endpoint provider: {provider}")


__all__ = ["create_endpoint_client"]
