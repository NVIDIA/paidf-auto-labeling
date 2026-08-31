# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared LLM-context helpers for ``reasoning`` LLM adapters.

Every LLM-driven DAFT-export stage (msted, temporal_localization,
reasoning, open_qa, mcq_openended, bcq_openended, causal_linkage)
renders a window block plus optional context blocks before the LLM
call. The rendering primitives — picking seconds / strings / captions
out of sidecar window dicts, building a header+body block, formatting
the per-window list with timecodes — were previously copy-pasted
across four LLM-adapter modules. That duplication was a drift trap:
the moment any one of them learned about a new sidecar key alias the
others would silently disagree on what counts as a "window with a
caption".

This module is the single source of truth. Callers parameterize the
log tag and (optionally) the key tuples; everything else is shared.

Note on the converter-side variants under ``reasoning/temporal.py``
and ``reasoning/chunks.py``: those have a different contract
(raise ``DaftConvertError`` on missing data rather than returning
``None``) so they are intentionally not folded in here. They would
require either a ``strict=`` flag (extra knob for a one-line
behavior swap) or callers to add their own ``if v is None: raise``
wrapper, both of which would shift duplication rather than remove it.

Addresses review feedback problem area #4.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

#: Sidecar keys checked, in order, when extracting a window's start
#: time in seconds. Lifted to a module constant so a new alias only
#: needs to be added in one place.
WINDOW_START_KEYS: tuple[str, ...] = (
    "start_s",
    "start",
    "start_seconds",
    "start_time",
)

#: Sidecar keys checked, in order, when extracting a window's end
#: time in seconds.
WINDOW_END_KEYS: tuple[str, ...] = (
    "end_s",
    "end",
    "end_seconds",
    "end_time",
)


def pick_seconds(win: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    """Return the first numeric value among ``keys`` as a float, else ``None``.

    Booleans are skipped on purpose — they are technically ``int`` in
    Python and would otherwise round-trip as ``0.0`` / ``1.0`` from a
    sidecar that mistakenly stored a flag under a temporal key.
    """
    for k in keys:
        if k in win:
            v = win[k]
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                return float(v)
    return None


def pick_string(win: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    """Return the first non-empty string among ``keys``, stripped, else ``None``."""
    for k in keys:
        v = win.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def block(header: str, body: str | None, *, fallback: str) -> str:
    """Compose an optional prompt block.

    - ``"<header>\\n<body>"`` when ``body`` is a non-empty string.
    - ``"<header>\\n<fallback>"`` when ``body`` is empty/None and
      ``fallback`` is non-empty.
    - ``""`` when both are empty (so callers can ``"\\n\\n".join([...])``
      without producing dangling headers).
    """
    if body is None or not isinstance(body, str) or not body.strip():
        if not fallback:
            return ""
        return f"{header}\n{fallback}"
    return f"{header}\n{body.strip()}"


def format_windows_block(
    windows: Iterable[Any],
    *,
    description_keys: tuple[str, ...],
    log: logging.Logger,
    tag: str,
    start_keys: tuple[str, ...] = WINDOW_START_KEYS,
    end_keys: tuple[str, ...] = WINDOW_END_KEYS,
) -> list[str]:
    """Render the per-window block as ``"MM:SS-MM:SS: <caption>"`` lines.

    Skips windows that are not dicts, have no resolvable start/end
    seconds, or carry no caption-like field. Each skip emits a debug
    log scoped to ``tag`` so per-stage logs stay distinguishable.

    ``seconds_to_timecode`` is imported lazily so this module stays
    importable without dragging in the rest of ``reasoning`` at
    test-collection time.
    """
    from reasoning.timecodes import seconds_to_timecode  # noqa: PLC0415

    out: list[str] = []
    for idx, win in enumerate(windows):
        if not isinstance(win, dict):
            log.debug("[%s] skipping window[%d]: not a dict", tag, idx)
            continue
        start = pick_seconds(win, start_keys)
        end = pick_seconds(win, end_keys)
        if start is None or end is None:
            log.debug("[%s] skipping window[%d]: missing start/end", tag, idx)
            continue
        caption = pick_string(win, description_keys)
        if caption is None:
            log.debug("[%s] skipping window[%d]: no caption-like field", tag, idx)
            continue
        out.append(f"{seconds_to_timecode(start)}-{seconds_to_timecode(end)}: {caption}")
    return out


__all__ = [
    "WINDOW_END_KEYS",
    "WINDOW_START_KEYS",
    "block",
    "format_windows_block",
    "pick_seconds",
    "pick_string",
]
