# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Captioning task package."""

from captioning.artifacts import CAPTION_ARTIFACTS_KEY, CaptionArtifactsState
from captioning.captioner import CaptionResult, DenseCaptioner
from captioning.config import CaptioningConfig
from captioning.task import CaptioningTask

__all__ = [
    "CAPTION_ARTIFACTS_KEY",
    "CaptionArtifactsState",
    "CaptionResult",
    "CaptioningConfig",
    "CaptioningTask",
    "DenseCaptioner",
]
