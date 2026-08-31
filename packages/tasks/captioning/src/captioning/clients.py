# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Endpoint client adapters used by captioning backends."""

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

from captioning.config import EndpointProvider


class CaptionEndpointError(EndpointError):
    """Raised when a caption endpoint call fails."""


CaptionEndpointClient = EndpointClient


def create_endpoint_client(
    *,
    provider: EndpointProvider,
    endpoint_url: str | None,
    model: str,
    timeout_s: float,
    retries: int,
    retry_backoff_s: float,
) -> CaptionEndpointClient:
    """Create an endpoint client for ``provider``."""
    return _create_endpoint_client(
        provider=provider,
        endpoint_url=endpoint_url,
        model=model,
        timeout_s=timeout_s,
        retries=retries,
        retry_backoff_s=retry_backoff_s,
        error_cls=CaptionEndpointError,
    )


__all__ = [
    "CaptionEndpointClient",
    "CaptionEndpointError",
    "ChatRequest",
    "MediaPayload",
    "create_endpoint_client",
]
