# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Response parsing helpers for VLM/LLM caption outputs.

The JSON-object, window-description, and visual-QA-item extractors are shared with
``visual_qa`` via ``core.llm`` (re-exported here for the captioning call sites).
``extract_inner_chunks`` is captioning-specific and stays local.
"""

from __future__ import annotations

from typing import Any

from core.llm.json_extract import extract_json_object
from core.llm.vlm_response import extract_visual_qa_items, extract_window_description


def extract_inner_chunks(parsed: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Extract model-emitted dense chunks, preserving the original chunk fields."""
    if parsed is None:
        return []
    chunks = parsed.get("chunks")
    if not isinstance(chunks, list):
        return []
    return [dict(chunk) for chunk in chunks if isinstance(chunk, dict)]


__all__ = [
    "extract_inner_chunks",
    "extract_json_object",
    "extract_visual_qa_items",
    "extract_window_description",
]
