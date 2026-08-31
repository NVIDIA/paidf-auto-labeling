# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Endpoint client adapters used by grounding (thin wrap of core)."""

from __future__ import annotations

from core.model_clients import (
    ChatRequest,
    EndpointClient,
    EndpointError,
    MediaPayload,
)
from core.model_clients import (
    create_endpoint_client as _create_endpoint_client,
)

from grounding_2d.config import EndpointProvider


class GroundingEndpointError(EndpointError):
    """Raised when a grounding VLM endpoint call fails."""


GroundingEndpointClient = EndpointClient


def create_endpoint_client(
    *,
    provider: EndpointProvider,
    endpoint_url: str | None,
    model: str,
    timeout_s: float,
    retries: int,
    retry_backoff_s: float,
) -> GroundingEndpointClient:
    """Create a shared core endpoint client for ``provider``."""
    return _create_endpoint_client(
        provider=provider,
        endpoint_url=endpoint_url,
        model=model,
        timeout_s=timeout_s,
        retries=retries,
        retry_backoff_s=retry_backoff_s,
        error_cls=GroundingEndpointError,
    )


__all__ = [
    "ChatRequest",
    "GroundingEndpointClient",
    "GroundingEndpointError",
    "MediaPayload",
    "create_endpoint_client",
]
