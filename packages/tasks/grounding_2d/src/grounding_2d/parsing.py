# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parsing and normalization helpers for 2D grounding model responses."""

from __future__ import annotations

import ast
import json
import re
from typing import Any


def parse_json_object(raw: str) -> dict[str, Any]:
    """Parse a JSON-like model response into a dictionary."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        text = text.strip()
    brace = text.find("{")
    if brace > 0:
        text = text[brace:]
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    end = text.rfind("}") + 1
    if 0 <= brace < end:
        parsed = json.loads(text[:end])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Model response did not contain a JSON object.")


def parse_grounding_response(raw: str) -> dict[str, Any]:
    """Parse grounding JSON, salvaging complete entries from truncated responses."""
    try:
        return parse_json_object(raw)
    except Exception:
        pass

    text = raw.strip()
    pattern = re.compile(
        r'"([^"]+)"\s*:\s*\{\s*"bboxes"\s*:\s*(\[[^\]]*(?:\[[^\]]*\][^\]]*)*\])'
        r'\s*,\s*"scores"\s*:\s*(\[[^\]]*\])\s*\}',
        re.DOTALL,
    )
    result: dict[str, Any] = {}
    for match in pattern.finditer(text):
        try:
            result[match.group(1)] = {
                "bboxes": json.loads(match.group(2)),
                "scores": json.loads(match.group(3)),
            }
        except json.JSONDecodeError:
            continue
    if result:
        return result
    raise ValueError("Model response did not contain grounding JSON.")


def normalize_bbox(
    bbox: object,
    *,
    width: int,
    height: int,
    coordinate_mode: str,
) -> list[int] | None:
    """Return a clamped pixel-space bbox or ``None`` when invalid."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    if coordinate_mode == "normalized_1000":
        x1 = x1 / 1000.0 * width
        x2 = x2 / 1000.0 * width
        y1 = y1 / 1000.0 * height
        y2 = y2 / 1000.0 * height
    left = max(0, min(width, int(round(min(x1, x2)))))
    right = max(0, min(width, int(round(max(x1, x2)))))
    top = max(0, min(height, int(round(min(y1, y2)))))
    bottom = max(0, min(height, int(round(max(y1, y2)))))
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def rle_area(counts: list[int]) -> int:
    """Compute foreground pixel count for integer RLE starting with background."""
    return sum(int(value) for value in counts[1::2])


__all__ = ["normalize_bbox", "parse_grounding_response", "parse_json_object", "rle_area"]
