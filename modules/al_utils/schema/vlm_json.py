# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class VlmJsonConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    model: Literal["vlm"] = "vlm"
    # Optional prompt overrides. When unset, shipped defaults are used.
    scene_prompt_file: Optional[str] = None
    events_prompt_file: Optional[str] = None
    default_video_json_prompt_file: str = "cookbooks/shared/prompts/vlm_json/video_json_prompt.md"
    default_video_events_prompt_file: str = "cookbooks/shared/prompts/vlm_json/video_events_prompt.md"
    default_image_json_prompt_file: str = "cookbooks/shared/prompts/vlm_json/image_caption_prompt.md"
    # Run VLM twice per video (video.json then events.json). More stable, ~2x cost.
    split_json_calls: bool = True
    # Structured output mode for VLM JSON generation:
    # - auto: NIM endpoints -> guided_json; otherwise -> response_format=json_object
    # - nim: force guided_json (permissive object schema)
    # - openai: force response_format=json_object
    # - off: keep raw text output (prompt/extraction driven)
    structured_output: Literal["auto", "nim", "openai", "off"] = "openai"

    # Sampling temperature for VLM JSON generation.
    # Keep default deterministic for stability (downstream parsing expects consistent schema).
    temperature: float = Field(default=0.0, ge=0.0)
    frame_fps: float = Field(default=1.0, gt=0.0)
    resolution: int = Field(default=360, gt=0)
    max_frames: int = Field(default=24, gt=0)
    max_tokens: int = Field(default=8192, gt=0)
    timeout: int = Field(default=600, gt=0)
    rate_limit: float = Field(default=0.0, ge=0.0)
