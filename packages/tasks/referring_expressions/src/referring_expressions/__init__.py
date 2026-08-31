# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Referring expressions: known boxes → VLM phrases."""

from referring_expressions.artifacts import (
    REFERRING_EXPRESSIONS_ARTIFACTS_KEY,
    ReferringExpressionsArtifactsState,
)
from referring_expressions.config import ReferringExpressionsConfig
from referring_expressions.task import ReferringExpressionsTask

__all__ = [
    "REFERRING_EXPRESSIONS_ARTIFACTS_KEY",
    "ReferringExpressionsArtifactsState",
    "ReferringExpressionsConfig",
    "ReferringExpressionsTask",
]
