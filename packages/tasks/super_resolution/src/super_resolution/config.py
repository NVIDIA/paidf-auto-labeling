# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration models for the super-resolution task."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SeedVR2Variant = Literal["seedvr2_3b", "seedvr2_7b"]
EmptyOutputPolicyValue = Literal["warn", "fail"]
ResolutionPolicyValue = Literal["always", "auto"]


class SeedVR2Config(BaseModel):
    """SeedVR2 runtime, cache, and inference settings."""

    model_config = ConfigDict(extra="forbid")

    variant: SeedVR2Variant = "seedvr2_3b"
    seed: int = 42
    res_h: int = Field(default=720, gt=0)
    res_w: int = Field(default=1280, gt=0)
    window_frames: int = Field(default=128, gt=0)
    overlap_frames: int = Field(default=64, ge=0)
    out_fps: float | None = Field(default=None, gt=0)
    gpu_ids: str | int | None = "all"
    use_multi_gpu: bool = False
    model_cache_path: str | None = None
    seedvr_root: str | None = None
    allow_checkpoint_download: bool = False
    keep_intermediates: bool = False
    empty_output_policy: EmptyOutputPolicyValue = "warn"
    command_timeout_s: float | None = Field(default=None, gt=0)


class SuperResolutionConfig(BaseModel):
    """Static configuration for ``SuperResolutionTask``."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    resolver: str = "seedvr2"
    resolution_policy: ResolutionPolicyValue = "always"
    min_input_short_side: int = Field(default=720, gt=0)
    min_input_long_side: int = Field(default=1280, gt=0)
    seedvr2: SeedVR2Config = Field(default_factory=SeedVR2Config)


__all__ = [
    "EmptyOutputPolicyValue",
    "ResolutionPolicyValue",
    "SeedVR2Config",
    "SeedVR2Variant",
    "SuperResolutionConfig",
]
