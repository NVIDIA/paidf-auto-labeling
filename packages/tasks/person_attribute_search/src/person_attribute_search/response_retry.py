# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retry helpers for unusable PAS LLM responses."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from core.model_clients import ChatRequest, EndpointClient, EndpointError

ResponseValidator = Callable[[str], str | None]


def generate_with_response_retries(
    client: EndpointClient,
    request: ChatRequest,
    *,
    validate_response: ResponseValidator,
    stage: str,
    retries: int,
    retry_backoff_s: float,
    logger: logging.Logger | None = None,
) -> str:
    """Generate text and retry responses rejected by ``validate_response``.

    Transport exceptions are left to the endpoint client, which already classifies
    retryable failures. This layer retries successful responses that are empty,
    malformed, or violate the PAS output contract, plus shared ``EndpointError``
    response-shape failures. After exhaustion, the last text response is returned so
    each caller retains its existing strict or fail-soft parsing behavior.
    """
    retry_count = max(0, int(retries))
    attempt = 0
    while True:
        validation_error: str | None
        try:
            raw = client.generate(request)
        except EndpointError as exc:
            if attempt >= retry_count:
                raise
            validation_error = f"invalid endpoint response: {exc}"
            raw = ""
        else:
            validation_error = validate_response(raw)
        if validation_error is None:
            return raw
        if attempt >= retry_count:
            if logger is not None:
                logger.warning(
                    "Invalid LLM response retries exhausted (stage=%s attempts=%d): %s",
                    stage,
                    retry_count + 1,
                    validation_error,
                )
            return raw
        sleep_s = min(60.0, max(0.0, float(retry_backoff_s)) * (2**attempt))
        if logger is not None:
            logger.warning(
                "Invalid LLM response (stage=%s attempt %d/%d): %s. Retrying in %.1fs",
                stage,
                attempt + 1,
                retry_count + 1,
                validation_error,
                sleep_s,
            )
        time.sleep(sleep_s)
        attempt += 1


__all__ = ["ResponseValidator", "generate_with_response_retries"]
