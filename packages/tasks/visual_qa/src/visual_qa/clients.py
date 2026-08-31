# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Endpoint client adapters used by visual QA generation."""

from __future__ import annotations

from core.model_clients import (
    DEFAULT_PARSER,
    ChatRequest,
    EndpointClient,
    EndpointProvider,
    MediaPayload,
    ReasoningParser,
)
from core.model_clients import (
    EndpointError as _SharedEndpointError,
)
from core.model_clients import (
    create_endpoint_client as _create_endpoint_client,
)


class EndpointError(_SharedEndpointError):
    """Raised when a visual QA model endpoint call fails."""


def create_endpoint_client(
    *,
    provider: EndpointProvider,
    endpoint_url: str | None,
    model: str,
    timeout_s: float,
    retries: int,
    retry_backoff_s: float,
    parser: ReasoningParser = DEFAULT_PARSER,
) -> EndpointClient:
    """Create an endpoint client for ``provider``."""
    return _create_endpoint_client(
        provider=provider,
        endpoint_url=endpoint_url,
        model=model,
        timeout_s=timeout_s,
        retries=retries,
        retry_backoff_s=retry_backoff_s,
        parser=parser,
        error_cls=EndpointError,
    )


__all__ = [
    "ChatRequest",
    "EndpointClient",
    "EndpointError",
    "EndpointProvider",
    "MediaPayload",
    "ReasoningParser",
    "create_endpoint_client",
]
