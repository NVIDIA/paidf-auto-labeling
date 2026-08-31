# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the referring-expressions task."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EndpointProvider = Literal["openai-compatible"]


class ReferringExpressionsConfig(BaseModel):
    """Static configuration for ``ReferringExpressionsTask``.

    Flow (MVP): read DAFT instances/boxes, ask a VLM for per-object region
    phrases. Auth for OpenAI-compatible VLMs uses ``NVIDIA_API_KEY``.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    force_reprocess: bool = False

    vlm_provider: EndpointProvider = "openai-compatible"
    vlm_endpoint_url: str | None = None
    vlm_model: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    system_prompt: str | None = None
    max_tokens: int = Field(default=4096, ge=1)
    temperature: float = Field(default=0.2, ge=0.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    timeout_s: float = Field(default=120.0, gt=0.0)
    retries: int = Field(default=2, ge=0)
    retry_backoff_s: float = Field(default=1.0, ge=0.0)

    # Planned follow-ups (hooks only in MVP).
    enable_grouped_expressions: bool = False
    enable_double_check: bool = False
    frame_number: int = Field(default=0, ge=0)
    draw_box_overlay: bool = True
    min_match_iou: float = Field(default=0.3, ge=0.0, le=1.0)


__all__ = ["EndpointProvider", "ReferringExpressionsConfig"]
