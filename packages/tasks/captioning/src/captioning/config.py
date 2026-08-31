# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration models for the captioning task."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

EndpointProvider = Literal["openai-compatible", "gemini"]
CaptionInputSource = Literal["auto", "original", "enhanced", "tracking"]
CaptionMediaMode = Literal["auto", "video", "frames"]


class CaptioningConfig(BaseModel):
    """Static configuration for ``CaptioningTask``."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    input_source: CaptionInputSource = "auto"
    image_group_dir: str | None = None
    max_group_images: int = Field(default=0, ge=0)

    vlm_provider: EndpointProvider = "openai-compatible"
    vlm_endpoint_url: str | None = None
    vlm_model: str = "default"

    enable_llm_summary: bool = False
    llm_provider: EndpointProvider | None = None
    llm_endpoint_url: str | None = None
    llm_model: str | None = None

    prompt_text: str | None = None
    prompt_file: str | None = None
    image_prompt_text: str | None = None
    image_prompt_file: str | None = None
    summary_prompt_text: str | None = None
    summary_prompt_file: str | None = None
    system_prompt: str | None = None

    window_seconds: float = Field(default=10.0, gt=0.0)
    window_frames: int = Field(default=256, ge=0)
    remainder_threshold: int = Field(default=128, ge=0)
    single_window: bool = False
    sampling_fps: float = Field(default=2.0, gt=0.0)
    max_frames: int = Field(default=8, ge=1)
    resolution: int = Field(default=768, ge=64)
    media_mode: CaptionMediaMode = "auto"

    max_tokens: int = Field(default=1024, ge=1)
    summary_input_token_budget: int = Field(default=6000, ge=1)
    temperature: float = Field(default=0.2, ge=0.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    timeout_s: float = Field(default=120.0, gt=0.0)
    retries: int = Field(default=2, ge=0)
    retry_backoff_s: float = Field(default=1.0, ge=0.0)

    preserve_raw_model_output: bool = False
    write_contextual: bool = True
    sidecar_filename: str = "metadata_chunk.json"

    @model_validator(mode="after")
    def _validate_combinations(self) -> Self:
        for text_field, file_field in (
            ("prompt_text", "prompt_file"),
            ("image_prompt_text", "image_prompt_file"),
            ("summary_prompt_text", "summary_prompt_file"),
        ):
            if getattr(self, text_field) is not None and getattr(self, file_field) is not None:
                raise ValueError(f"{text_field} and {file_field} cannot both be set")

        if self.enable_llm_summary:
            if not self.llm_model:
                raise ValueError("enable_llm_summary requires llm_model")
            if self.llm_provider is None and not self.llm_endpoint_url:
                raise ValueError("enable_llm_summary requires llm_provider or llm_endpoint_url")
        return self


__all__ = [
    "CaptionInputSource",
    "CaptionMediaMode",
    "CaptioningConfig",
    "EndpointProvider",
]
