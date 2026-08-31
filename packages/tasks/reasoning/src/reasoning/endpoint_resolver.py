# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Endpoint resolution — constructed once before the sample loop.

Resolution order:
  1. Env vars   VLM_ENDPOINT_URL / VLM_BASE_URL / VLM_MODEL
                LLM_ENDPOINT_URL / LLM_BASE_URL / LLM_MODEL
  2. config.endpoints.vlm / config.endpoints.llm

OpenAI-compatible LLM authentication always uses ``NVIDIA_API_KEY``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from core.model_clients import NVIDIA_API_KEY_ENV

_MISSING = object()


class EndpointResolver:
    """Resolved endpoint config, constructed once before the sample loop.

    Args:
        endpoints: Parsed ``config.endpoints`` Pydantic model (may be None).
        logger: Logger instance.

    """

    def __init__(
        self,
        endpoints: Any | None,
        *,
        logger: logging.Logger,
    ) -> None:
        ep = endpoints

        # VLM — env vars take precedence, then config
        self._vlm_url: str = (
            os.getenv("VLM_ENDPOINT_URL")
            or os.getenv("VLM_BASE_URL")
            or str(_endpoint_value(ep, "vlm", "url", ""))
        )
        self._vlm_model: str = os.getenv("VLM_MODEL") or str(
            _endpoint_value(ep, "vlm", "model", "")
        )
        self._vlm_retries: int = int(_endpoint_value(ep, "vlm", "retries", 3))
        self._vlm_retry_backoff_s: float = float(_endpoint_value(ep, "vlm", "retry_backoff_s", 5.0))

        # LLM — env vars take precedence, then config
        self._llm_url: str = (
            os.getenv("LLM_ENDPOINT_URL")
            or os.getenv("LLM_BASE_URL")
            or str(_endpoint_value(ep, "llm", "url", ""))
        )
        self._llm_model: str = os.getenv("LLM_MODEL") or str(
            _endpoint_value(ep, "llm", "model", "")
        )
        self._llm_retries: int = int(_endpoint_value(ep, "llm", "retries", 3))
        self._llm_retry_backoff_s: float = float(_endpoint_value(ep, "llm", "retry_backoff_s", 5.0))

        self._endpoints = endpoints
        self._logger = logger

    # ------------------------------------------------------------------
    # Resolution methods
    # ------------------------------------------------------------------

    def resolve_vlm(self, *, required: bool = False) -> tuple[str, str]:
        """Return (url, model) for the VLM endpoint.

        Args:
            required: If True, raises ValueError when url or model is missing.

        Returns:
            (url, model) strings — may be empty if not configured and not required.
        """
        if required and (not self._vlm_url or not self._vlm_model):
            raise ValueError(
                "VLM endpoint not configured. "
                "Set endpoints.vlm.{url,model} in the config "
                "or the VLM_ENDPOINT_URL / VLM_BASE_URL / VLM_MODEL environment variables."
            )
        return self._vlm_url, self._vlm_model

    def resolve_llm(self, *, required: bool = False) -> tuple[str, str]:
        """Return (url, model) for the LLM endpoint.

        Args:
            required: If True, raises ValueError when url or model is missing.

        Returns:
            (url, model) strings — may be empty if not configured and not required.
        """
        if required and (not self._llm_url or not self._llm_model):
            raise ValueError(
                "LLM endpoint not configured. "
                "Set endpoints.llm.{url,model} in the config "
                "or the LLM_ENDPOINT_URL / LLM_BASE_URL / LLM_MODEL environment variables."
            )
        return self._llm_url, self._llm_model

    def resolve_llm_api_key(self) -> str:
        """Return the LLM API key from ``NVIDIA_API_KEY``, or ``EMPTY`` when unset."""
        return os.getenv(NVIDIA_API_KEY_ENV) or "EMPTY"

    # ------------------------------------------------------------------
    # Best-effort overrides (used by NVCF auto-detection)
    # ------------------------------------------------------------------

    def apply_vlm_defaults(self, *, url: str = "", model: str = "") -> None:
        """Best-effort: set missing VLM url/model only.

        Does not override explicitly configured values.
        """
        u = str(url or "").strip()
        m = str(model or "").strip()
        if u and not self._vlm_url:
            self._vlm_url = u
        if m and not self._vlm_model:
            self._vlm_model = m

    def apply_llm_defaults(
        self,
        *,
        url: str = "",
        model: str = "",
    ) -> None:
        """Best-effort: set missing LLM url/model only.

        Does not override explicitly configured values.
        """
        u = str(url or "").strip()
        m = str(model or "").strip()
        if u and not self._llm_url:
            self._llm_url = u
        if m and not self._llm_model:
            self._llm_model = m

    def apply_llm_overrides(
        self,
        *,
        url: str = "",
        model: str = "",
    ) -> None:
        """Set explicit LLM CLI overrides when values are provided."""
        u = str(url or "").strip()
        m = str(model or "").strip()
        if u:
            self._llm_url = u
        if m:
            self._llm_model = m

    # ------------------------------------------------------------------
    # Retry / backoff accessors (used by factories to pass to runners)
    # ------------------------------------------------------------------

    @property
    def vlm_retries(self) -> int:
        return self._vlm_retries

    @property
    def vlm_retry_backoff_s(self) -> float:
        return self._vlm_retry_backoff_s

    @property
    def llm_retries(self) -> int:
        return self._llm_retries

    @property
    def llm_retry_backoff_s(self) -> float:
        return self._llm_retry_backoff_s

    @property
    def endpoints(self) -> Any | None:
        """Raw resolved EndpointsConfig (may be None if no endpoints configured)."""
        return self._endpoints


def _endpoint_value(endpoints: Any | None, family: str, key: str, default: Any) -> Any:
    if endpoints is None:
        return default
    if isinstance(endpoints, dict):
        group = endpoints.get(family, _MISSING)
    else:
        group = getattr(endpoints, family, _MISSING)
    if group is _MISSING or group is None:
        return default
    if isinstance(group, dict):
        value = group.get(key, _MISSING)
    else:
        value = getattr(group, key, _MISSING)
    if value is _MISSING or value is None:
        return default
    return value
