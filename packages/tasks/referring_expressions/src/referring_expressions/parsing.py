# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parsing helpers for referring-expression VLM responses."""

from __future__ import annotations

import ast
import json
import re
from typing import Any


def parse_json_array(raw: str) -> list[Any]:
    """Parse a JSON array from a model response."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        text = text.strip()
    bracket = text.find("[")
    if bracket > 0:
        text = text[bracket:]
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    end = text.rfind("]") + 1
    if end > 1:
        try:
            parsed = json.loads(text[:end])
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return parsed
    # Salvage complete object dicts from truncated arrays.
    pattern = re.compile(r"\{[^{}]*\}", re.DOTALL)
    salvaged: list[Any] = []
    for match in pattern.finditer(text):
        try:
            item = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            salvaged.append(item)
    if salvaged:
        return salvaged
    raise ValueError("Model response did not contain a JSON array.")


def normalize_bbox(bbox: object, *, width: int, height: int) -> list[int] | None:
    """Clamp a pixel-space bbox to image bounds."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    left = max(0, min(width, int(round(min(x1, x2)))))
    right = max(0, min(width, int(round(max(x1, x2)))))
    top = max(0, min(height, int(round(min(y1, y2)))))
    bottom = max(0, min(height, int(round(max(y1, y2)))))
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


__all__ = ["normalize_bbox", "parse_json_array"]
