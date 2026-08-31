# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared scaffolding for opt-in DAFT-export pipeline stages.

The ten LLM-driven emitters in :mod:`pipeline` (``msted``,
``temporal_localization``, ``causal_linkage``, ``open_qa``,
``mcq_openended``, ``bcq_openended``, ``temporal_description``, plus
the chunks/prose-tasks pure pivots) all repeated the same eight-step
scaffold:

1. early-return on image scenes (when applicable)
2. resolve LLM endpoint, return on missing
3. read ``sidecars/metadata.json``
4. read ``contextual/video.json`` for scene prose / duration
5. resolve optional ``prompt_dir`` against ``config_dir`` and build
   a :class:`PromptRegistry`
6. call the LLM adapter (typed errors logged, others re-raised)
7. call the pure DAFT converter (``DaftConvertError`` logged)
8. ``write_daft_json`` and log success

This package factors all of (3)–(5), (7), and (8) into shared
primitives. The per-stage implementations under :mod:`reasoning.stages`
are typically ~25 lines each: pull the slices of ``StageInputs`` they
need, call the LLM adapter, return the converted payload. The
emitter scaffolding handles the rest, including the image-skip
decision (declarative ``requires_temporal_axis = True`` flag).
"""

from __future__ import annotations

from reasoning.stage_inputs import StageInputs, safe_read_json
from reasoning.stages.base import LlmStageEmitter, build_prompt_registry, emit_stage

__all__ = [
    "LlmStageEmitter",
    "StageInputs",
    "build_prompt_registry",
    "emit_stage",
    "safe_read_json",
]
