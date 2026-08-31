# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expression filtering for SAM3 grounding.

Domain semantics are not hardcoded here. Prefer:

1. VLM ``groundable`` on each expression (set during Step 0), and
2. Optional JSON policy files for deploy-time deny lists, and
3. Detector physics only (SAM3 CLIP text length / CJK density).

Empty / missing policy => no keyword assumptions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_WHITESPACE_RE = re.compile(r"\s+")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

# SAM3 CLIP text encoder: max_position_embeddings=32 (incl. specials).
_SAM3_MAX_PROMPT_CHARS = 64
_SAM3_MAX_PROMPT_WORDS = 8
_SAM3_MAX_CJK_CHARS = 4


@dataclass(frozen=True)
class ExpressionFilterPolicy:
    """Optional deploy-time keyword policy (usually empty)."""

    deny_phrases: frozenset[str] = field(default_factory=frozenset)
    deny_nouns: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def empty(cls) -> ExpressionFilterPolicy:
        return cls()

    @classmethod
    def from_json_file(cls, path: str | Path) -> ExpressionFilterPolicy:
        """Load an optional policy JSON.

        Expected shape::

            {
              "deny_phrases": ["the scene", ...],
              "deny_nouns": ["background", ...]
            }

        Phrases/nouns are matched after lowercasing and whitespace collapse.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expression filter policy must be a JSON object: {path}")
        phrases = payload.get("deny_phrases") or []
        nouns = payload.get("deny_nouns") or []
        if not isinstance(phrases, list) or not isinstance(nouns, list):
            raise ValueError(f"deny_phrases and deny_nouns must be lists in policy file: {path}")
        return cls(
            deny_phrases=frozenset(
                normalize_expression_text(str(p)) for p in phrases if str(p).strip()
            ),
            deny_nouns=frozenset(
                normalize_expression_text(str(n)) for n in nouns if str(n).strip()
            ),
        )


def normalize_expression_text(text: str) -> str:
    """Lowercase and collapse whitespace for phrase comparisons."""
    return _WHITESPACE_RE.sub(" ", text.strip().lower())


def fits_sam3_text_encoder(text: str) -> bool:
    """Return True when ``text`` is unlikely to exceed SAM3's CLIP length cap."""
    stripped = _WHITESPACE_RE.sub(" ", text.strip())
    if not stripped:
        return False
    if len(stripped) > _SAM3_MAX_PROMPT_CHARS:
        return False
    if len(stripped.split()) > _SAM3_MAX_PROMPT_WORDS:
        return False
    if len(_CJK_RE.findall(stripped)) > _SAM3_MAX_CJK_CHARS:
        return False
    return True


def _vlm_marked_ungroundable(expression: dict[str, Any]) -> bool:
    """True when the VLM explicitly marked this expression as not groundable."""
    if "groundable" not in expression:
        return False
    value = expression.get("groundable")
    if isinstance(value, bool):
        return not value
    if isinstance(value, str):
        return value.strip().lower() in {"false", "0", "no"}
    return False


def _blocked_by_policy(expression: dict[str, Any], *, policy: ExpressionFilterPolicy) -> bool:
    text = normalize_expression_text(str(expression.get("text") or ""))
    if not text:
        return True
    if text in policy.deny_phrases:
        return True
    noun = normalize_expression_text(str(expression.get("noun_chunk") or ""))
    if noun and noun in policy.deny_nouns:
        return True
    tokens = text.split()
    if len(tokens) == 1 and tokens[0] in policy.deny_nouns:
        return True
    if tokens and tokens[-1] in policy.deny_nouns:
        return True
    return False


def is_groundable_expression(
    expression: dict[str, Any],
    *,
    policy: ExpressionFilterPolicy | None = None,
) -> bool:
    """Return True when an expression may be sent to SAM3.

    Decision order (no domain assumptions in code):

    1. Reject empty text / SAM3-unsafe length.
    2. Reject when VLM set ``groundable: false``.
    3. Reject when optional JSON policy deny-lists match.
    """
    raw_text = str(expression.get("text") or "").strip()
    if not raw_text:
        return False
    if not fits_sam3_text_encoder(raw_text):
        return False
    if _vlm_marked_ungroundable(expression):
        return False
    active = policy or ExpressionFilterPolicy.empty()
    if _blocked_by_policy(expression, policy=active):
        return False
    return True


def filter_groundable_expressions(
    expressions: list[dict[str, Any]],
    *,
    policy: ExpressionFilterPolicy | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split expressions into (groundable, skipped)."""
    active = policy or ExpressionFilterPolicy.empty()
    groundable: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for expression in expressions:
        if is_groundable_expression(expression, policy=active):
            groundable.append(expression)
        else:
            skipped.append(expression)
    return groundable, skipped


__all__ = [
    "ExpressionFilterPolicy",
    "filter_groundable_expressions",
    "fits_sam3_text_encoder",
    "is_groundable_expression",
    "normalize_expression_text",
]
