# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Object-type normalization for referring expressions.

Types are open vocabulary (any short noun the VLM returns). Soft aliases only
collapse common spelling variants — there is no domain allowlist.
"""

from __future__ import annotations

import re

_WHITESPACE_RE = re.compile(r"\s+")
_NON_SLUG_RE = re.compile(r"[^a-z0-9_]+")

# Optional spelling / synonym collapses. Not a closed taxonomy.
_TYPE_ALIASES: dict[str, str] = {
    "semi": "truck",
    "semi-truck": "truck",
    "semi_truck": "truck",
    "lorry": "truck",
    "auto": "car",
    "automobile": "car",
    "human": "person",
    "people": "person",
    "pedestrian": "person",
    "bike": "bicycle",
    "motorbike": "motorcycle",
    "unknown": "other",
}


def normalize_object_type(raw: object) -> str:
    """Normalize a free-form type to a lowercase slug (default ``other``)."""
    text = _WHITESPACE_RE.sub(" ", str(raw or "").strip().lower())
    if not text:
        return "other"
    dashed = text.replace(" ", "_").replace("-", "_")
    if dashed in _TYPE_ALIASES:
        return _TYPE_ALIASES[dashed]
    if text in _TYPE_ALIASES:
        return _TYPE_ALIASES[text]
    slug = _NON_SLUG_RE.sub("_", dashed).strip("_")
    return slug or "other"


__all__ = ["normalize_object_type"]
