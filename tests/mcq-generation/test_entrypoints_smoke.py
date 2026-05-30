# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib


def test_mcq_runners_importable() -> None:
    m1 = importlib.import_module("mcq_generation.mcq.runners.window_vlm_llm")
    m2 = importlib.import_module("mcq_generation.mcq.runners.window_direct_vlm")
    m4 = importlib.import_module("mcq_generation.mcq.runners.metadata_llm")
    m5 = importlib.import_module("mcq_generation.question_driven_vlm_llm")

    assert hasattr(m1, "WindowVlmLlmRunner")
    assert hasattr(m2, "WindowDirectVlmRunner")
    assert hasattr(m4, "MetadataLlmRunner")
    assert hasattr(m5, "QuestionDrivenVlmLlmGenerator")
