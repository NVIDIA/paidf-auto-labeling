# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Field extractors for parsed VLM/LLM response objects.

These operate on an already-parsed response dict (see
:func:`core.llm.json_extract.extract_json_object`) and are shared by the
``captioning`` and ``visual_qa`` tasks, which consume the same VLM output shapes.
Keeping them here avoids the previous per-task copies drifting apart.
"""

from __future__ import annotations

from typing import Any


def extract_window_description(raw_text: str, parsed: dict[str, Any] | None) -> str:
    """Extract a human-readable description from a parsed model response.

    Prefers the first non-empty of ``event_summary``/``scene_description``/
    ``description``/``summary``/``caption``; otherwise joins per-chunk
    ``description`` fields; otherwise falls back to the stripped raw text.
    """
    if parsed is None:
        return raw_text.strip()
    for key in ("event_summary", "scene_description", "description", "summary", "caption"):
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    chunks = parsed.get("chunks")
    if isinstance(chunks, list):
        descriptions = [
            str(chunk.get("description", "")).strip()
            for chunk in chunks
            if isinstance(chunk, dict) and str(chunk.get("description", "")).strip()
        ]
        if descriptions:
            return " ".join(descriptions)
    return raw_text.strip()


def extract_visual_qa_items(parsed: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Extract visual-QA item objects from a parsed model response.

    Returns the first present of ``items`` or ``mcq`` as a list of dicts. This
    helper deliberately does no answer validation; the ``visual_qa`` task owns
    bank validation, ``include_if`` gating, and aggregation.
    """
    if parsed is None:
        return []
    for key in ("items", "mcq"):
        raw_items = parsed.get(key)
        if isinstance(raw_items, list):
            return [dict(item) for item in raw_items if isinstance(item, dict)]
    return []


__all__ = [
    "extract_visual_qa_items",
    "extract_window_description",
]
