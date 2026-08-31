# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the 2D grounding task."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EndpointProvider = Literal["openai-compatible"]


class Grounding2DConfig(BaseModel):
    """Static configuration for ``Grounding2DTask``.

    Flow: VLM expression extraction, then SAM3 text-prompted boxes/masks via
    ``detection_and_tracking``. Auth for OpenAI-compatible VLMs uses
    ``NVIDIA_API_KEY`` (core contract).
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    force_reprocess: bool = False
    caption: str | None = None
    input_metadata_filename: str = "input.json"

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

    max_instances_per_expression: int = Field(default=10, ge=1)
    filter_ungroundable_expressions: bool = True
    expression_filter_policy_path: str | None = None
    min_instance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    min_bbox_area: int = Field(default=64, ge=0)

    sam3_model_cache_path: str | None = None
    sam3_gpu_ids: str | int | None = "all"
    sam3_version: Literal["sam3", "sam3.1"] = "sam3"
    sam3_runtime: Literal["auto", "transformers", "native"] = "auto"
    sam3_target_fps: float = Field(default=10.0, gt=0.0)
    sam3_session_reset_s: float = Field(default=10.0, gt=0.0)
    sam3_max_duration_s: float = Field(default=30.0, gt=0.0)
    sam3_write_annotated_media: bool = False
    sam3_annotated_media_label_style: Literal["id", "name", "none"] = "name"
    sam3_annotated_media_mask_opacity: int = Field(default=0, ge=0, le=100)


__all__ = ["EndpointProvider", "Grounding2DConfig"]
