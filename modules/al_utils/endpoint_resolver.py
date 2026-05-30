# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Endpoint resolution — constructed ONCE in cli.py before the sample loop.

Resolution order:
  1. Env vars   VLM_BASE_URL / VLM_MODEL / LLM_BASE_URL / LLM_MODEL
  2. config.endpoints.vlm / config.endpoints.llm
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from al_utils.schema.endpoints import EndpointsConfig


class EndpointResolver:
    """Resolved endpoint config, constructed once before the sample loop.

    Args:
        endpoints: Parsed ``config.endpoints`` Pydantic model (may be None).
        logger: Logger instance.

    """

    def __init__(
        self,
        endpoints: Optional[EndpointsConfig],
        *,
        logger: logging.Logger,
    ) -> None:
        ep = endpoints

        # VLM — env vars take precedence, then config
        self._vlm_url: str = os.getenv("VLM_BASE_URL") or (ep.vlm.url if ep and ep.vlm and ep.vlm.url else "") or ""
        self._vlm_model: str = os.getenv("VLM_MODEL") or (ep.vlm.model if ep and ep.vlm and ep.vlm.model else "") or ""
        self._vlm_retries: int = ep.vlm.retries if ep and ep.vlm else 3
        self._vlm_retry_backoff_s: float = ep.vlm.retry_backoff_s if ep and ep.vlm else 5.0

        # LLM — env vars take precedence, then config
        self._llm_url: str = os.getenv("LLM_BASE_URL") or (ep.llm.url if ep and ep.llm and ep.llm.url else "") or ""
        self._llm_model: str = os.getenv("LLM_MODEL") or (ep.llm.model if ep and ep.llm and ep.llm.model else "") or ""
        self._llm_retries: int = ep.llm.retries if ep and ep.llm else 3
        self._llm_retry_backoff_s: float = ep.llm.retry_backoff_s if ep and ep.llm else 5.0

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
                "or the VLM_BASE_URL / VLM_MODEL environment variables."
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
                "or the LLM_BASE_URL / LLM_MODEL environment variables."
            )
        return self._llm_url, self._llm_model

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

    def apply_llm_defaults(self, *, url: str = "", model: str = "") -> None:
        """Best-effort: set missing LLM url/model only.

        Does not override explicitly configured values.
        """
        u = str(url or "").strip()
        m = str(model or "").strip()
        if u and not self._llm_url:
            self._llm_url = u
        if m and not self._llm_model:
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
    def endpoints(self) -> Optional[EndpointsConfig]:
        """Raw resolved EndpointsConfig (may be None if no endpoints configured)."""
        return self._endpoints
