# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompts and box↔object matching for referring expressions."""

from __future__ import annotations

from typing import Any

REGION_EXPRESSION_PROMPT_TEMPLATE = """\
You are an AI visual assistant. The image has numbered colored boxes (marks).
Describe each marked object in one short, discriminative phrase.

Marked objects (mark id → pixel bbox [x1, y1, x2, y2]):
{marks_block}

Rules:
1. Return exactly one JSON object per mark id listed above.
2. Lead with color (if visible) and object type.
3. Add ONE distinguishing detail (position, relation to a neighbor, or action/state).
4. Keep each description under 15 words. No guessing beyond what is visible.
5. "type" is a short lowercase object noun (specific > generic); no closed enum.
6. "color" is a simple color word or "unknown".
7. Use the provided bbox values for "bbox_2d" (do not invent new boxes).

Good examples:
- "The white chair nearest the window"
- "A red package on the lower shelf"
- "Person in a dark jacket crossing the crosswalk"

Bad examples:
- "Object on surface" (too vague)
- Long multi-clause sentences

Respond ONLY with a JSON array:
[
  {{"mark": 1, "bbox_2d": [x1,y1,x2,y2], "type": "chair", "color": "white", "description": "..."}},
  ...
]
"""


def region_expression_prompt(box_records: list[dict[str, Any]]) -> str:
    """Build the Step 0 region-expression prompt for marked boxes."""
    lines: list[str] = []
    for index, record in enumerate(box_records, start=1):
        bbox = record["bbox"]
        lines.append(f"  [{index}] {bbox}")
    return REGION_EXPRESSION_PROMPT_TEMPLATE.format(
        marks_block="\n".join(lines),
    )


def bbox_iou(a: list[int], b: list[int]) -> float:
    """Intersection-over-union for two [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter) / float(denom) if denom > 0 else 0.0


def match_region_to_candidate(
    *,
    bbox: list[int],
    candidates: list[dict[str, Any]],
    min_iou: float = 0.3,
    used_ids: set[str] | None = None,
) -> dict[str, Any] | None:
    """Return the best unused overlapping candidate, if IoU >= ``min_iou``."""
    best: dict[str, Any] | None = None
    best_iou = 0.0
    blocked = used_ids or set()
    for candidate in candidates:
        oid = candidate.get("object_id")
        if oid is not None and str(oid) in blocked:
            continue
        cand_bbox = list(candidate.get("bbox") or [])
        if len(cand_bbox) != 4:
            continue
        iou = bbox_iou(bbox, [int(v) for v in cand_bbox])
        if iou > best_iou:
            best_iou = iou
            best = candidate
    if best is None or best_iou < min_iou:
        return None
    return best


def greedy_match_by_iou(
    *,
    predicted: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    min_iou: float = 0.3,
) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    """Greedy IoU matching of predicted regions to candidate DAFT boxes."""
    pairs: list[tuple[float, int, int]] = []
    for pi, pred in enumerate(predicted):
        pb = pred.get("bbox")
        if not isinstance(pb, list) or len(pb) != 4:
            continue
        for ci, cand in enumerate(candidates):
            cb = cand.get("bbox")
            if not isinstance(cb, list) or len(cb) != 4:
                continue
            iou = bbox_iou([int(v) for v in pb], [int(v) for v in cb])
            if iou >= min_iou:
                pairs.append((iou, pi, ci))
    pairs.sort(reverse=True)
    used_p: set[int] = set()
    used_c: set[int] = set()
    assignment: dict[int, int] = {}
    for _iou, pi, ci in pairs:
        if pi in used_p or ci in used_c:
            continue
        used_p.add(pi)
        used_c.add(ci)
        assignment[pi] = ci
    result: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    for pred_idx, pred in enumerate(predicted):
        cand_idx = assignment.get(pred_idx)
        result.append((pred, candidates[cand_idx] if cand_idx is not None else None))
    return result


# Back-compat alias used by older tests/callers.
def match_region_to_object_id(
    *,
    bbox: list[int],
    candidates: list[dict[str, Any]],
) -> str | None:
    """Return object_id of the best IoU match."""
    matched = match_region_to_candidate(bbox=bbox, candidates=candidates)
    if matched is None:
        return None
    object_id = matched.get("object_id")
    return str(object_id) if object_id is not None else None


__all__ = [
    "bbox_iou",
    "greedy_match_by_iou",
    "match_region_to_candidate",
    "match_region_to_object_id",
    "region_expression_prompt",
]
