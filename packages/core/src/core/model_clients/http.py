# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP helpers shared by model endpoint clients."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from core.model_clients.types import EndpointError


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str],
    timeout_s: float,
    retries: int,
    retry_backoff_s: float,
    error_cls: type[RuntimeError] = EndpointError,
    response_meta: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """POST JSON and retry transient transport errors."""
    parsed_url = urllib.parse.urlparse(url)
    if parsed_url.scheme.lower() not in {"http", "https"}:
        raise ValueError(f"Unsupported endpoint URL scheme: {parsed_url.scheme or '<missing>'}")

    body = json.dumps(payload).encode("utf-8")
    last_error: BaseException | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                decoded = response.read().decode("utf-8")
            parsed = json.loads(decoded)
            if not isinstance(parsed, dict):
                raise error_cls("Endpoint returned non-object JSON")
            if response_meta is not None:
                response_meta["retry_count"] = attempt
            return parsed
        except urllib.error.HTTPError as exc:
            if not _is_retryable_http_error(exc):
                raise
            last_error = exc
        except json.JSONDecodeError:
            raise
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
        if last_error is not None:
            if attempt >= retries:
                if response_meta is not None:
                    response_meta["retry_count"] = attempt
                raise last_error
            sleep_s = retry_backoff_s * (attempt + 1)
            if logger is not None:
                logger.warning(
                    "Transient LLM endpoint error "
                    "(provider=http attempt %d/%d): %s. Retrying in %.1fs",
                    attempt + 1,
                    retries + 1,
                    last_error,
                    sleep_s,
                )
            time.sleep(sleep_s)
    raise error_cls("Endpoint request failed without an exception")


def _is_retryable_http_error(exc: urllib.error.HTTPError) -> bool:
    return exc.code in {408, 429} or 500 <= exc.code <= 599
