# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visual question-answering sidecar task."""

from visual_qa.artifacts import VISUAL_QA_ARTIFACTS_KEY, VisualQaArtifactsState
from visual_qa.task import (
    VisualQaConfig,
    VisualQaGenerationMode,
    VisualQaInputSource,
    VisualQaMediaMode,
    VisualQaTask,
)

__all__ = [
    "VISUAL_QA_ARTIFACTS_KEY",
    "VisualQaArtifactsState",
    "VisualQaConfig",
    "VisualQaGenerationMode",
    "VisualQaInputSource",
    "VisualQaMediaMode",
    "VisualQaTask",
]
