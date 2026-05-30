# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict


class SuperResolutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    model: Literal["seedvr2"] = "seedvr2"
    variant: str = "seedvr2_3b"
    seed: int = 42
    res_h: int = 720
    res_w: int = 1280
    window_frames: int = 128
    overlap_frames: int = 64
    window_timeout: int = 3600
    out_fps: Optional[float] = None
