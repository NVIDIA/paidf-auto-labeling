# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

from al_utils.schema.base import _clean_optional_str_allow_required_sentinel
from pydantic import BaseModel, ConfigDict, Field, model_validator


class VlmEndpointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: Optional[str] = None
    model: Optional[str] = None
    retries: int = Field(default=3, ge=0, description="Number of retries with exponential backoff (5s, 10s, 20s).")
    retry_backoff_s: float = Field(
        default=5.0,
        ge=0.1,
        le=300.0,
        description="Base delay in seconds for exponential backoff; delays are base * 2^attempt (capped at 60s).",
    )

    @model_validator(mode="after")
    def _sanitize_required(self) -> "VlmEndpointConfig":
        # Endpoints are conditionally required depending on which stages are enabled.
        # Treat "__REQUIRED__" placeholders as unset here; PipelineConfig enforces requirements.
        self.url = _clean_optional_str_allow_required_sentinel(self.url, field="endpoints.vlm.url")
        self.model = _clean_optional_str_allow_required_sentinel(self.model, field="endpoints.vlm.model")
        return self


class LlmEndpointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: Optional[str] = None
    model: Optional[str] = None
    retries: int = Field(default=3, ge=0, description="Number of retries with exponential backoff (5s, 10s, 20s).")
    retry_backoff_s: float = Field(
        default=5.0,
        ge=0.1,
        le=300.0,
        description="Base delay in seconds for exponential backoff; delays are base * 2^attempt (capped at 60s).",
    )

    @model_validator(mode="after")
    def _sanitize_required(self) -> "LlmEndpointConfig":
        self.url = _clean_optional_str_allow_required_sentinel(self.url, field="endpoints.llm.url")
        self.model = _clean_optional_str_allow_required_sentinel(self.model, field="endpoints.llm.model")
        return self


class EndpointsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vlm: Optional[VlmEndpointConfig] = None
    llm: Optional[LlmEndpointConfig] = None
