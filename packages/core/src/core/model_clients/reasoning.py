# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reasoning-mode controls for chat-completions model clients.

Hybrid models can either emit chain-of-thought ``<think>`` traces ("reasoning"
mode) or answer directly ("instruct" mode). This module centralizes the
request- and response-side handling so every client exposes a single ``parser``
switch with consistent semantics.
"""

from __future__ import annotations

import re
from typing import Any, Literal

ReasoningParser = Literal["instruct", "reasoning"]
"""Reasoning mode for hybrid models.

``"instruct"`` (default) disables model reasoning so responses are directly
parseable; ``"reasoning"`` leaves it enabled.
"""

DEFAULT_PARSER: ReasoningParser = "instruct"

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def extra_body_for_parser(parser: ReasoningParser) -> dict[str, Any] | None:
    """Return endpoint ``extra_body`` that enforces ``parser`` mode.

    ``"instruct"`` disables chat-template thinking for hybrid models so the
    response is directly parseable; ``"reasoning"`` returns ``None`` to leave the
    server defaults untouched.
    """
    if parser == "instruct":
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return None


def merge_extra_body(
    base: dict[str, Any] | None,
    override: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Recursively merge two ``extra_body`` mappings.

    ``override`` values win on conflict, but nested dicts are merged rather than
    replaced so reasoning controls and structured-output options can coexist in
    the same request body. Inputs are not mutated.
    """
    if base is None:
        return dict(override) if override is not None else None
    if override is None:
        return dict(base)
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = merge_extra_body(existing, value)
        else:
            merged[key] = value
    return merged


def strip_think_blocks(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning traces from ``text``.

    Hybrid models in "reasoning" mode can leak think blocks into the content
    channel; stripping them keeps downstream JSON/text parsing robust.
    """
    if not text or "<think>" not in text.lower():
        return text
    return _THINK_BLOCK_RE.sub("", text).strip()


__all__ = [
    "DEFAULT_PARSER",
    "ReasoningParser",
    "extra_body_for_parser",
    "merge_extra_body",
    "strip_think_blocks",
]
