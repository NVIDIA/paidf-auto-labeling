# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSON parsing helpers for LLM text responses."""

from __future__ import annotations

import json
import os
from typing import Any

import regex as re


def wrap_json_fence(obj: dict[str, Any]) -> str:
    """Render ``obj`` in a Markdown JSON fence."""
    return "```json\n" + json.dumps(obj, ensure_ascii=False, indent=2) + "\n```"


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from a raw model response (no repair).

    Collects fenced ```json blocks, then the text from its first ``{``, and
    returns the first candidate that ``json.JSONDecoder.raw_decode`` parses into a
    dict. Unlike :func:`extract_json_object_from_llm_text` this performs no
    trailing-comma/brace repair; prefer it for strict VLM JSON outputs (the
    ``captioning`` and ``visual_qa`` tasks share this single implementation).
    """
    candidates = [match.group(1).strip() for match in _FENCED_JSON_RE.finditer(text)]
    stripped = text.strip()
    if stripped.startswith("{"):
        candidates.append(stripped)
    else:
        object_start = stripped.find("{")
        if object_start >= 0:
            candidates.append(stripped[object_start:])
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            parsed, _idx = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_strict_json_object(text: str) -> dict[str, Any] | None:
    """Parse ``text`` only when it is exactly one JSON object."""
    candidate = str(text or "").strip()
    if not candidate:
        return None
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def extract_json_object_from_llm_text(text: str) -> dict[str, Any] | None:
    """Best-effort extraction for noisy LLM responses.

    Handles strict JSON, fenced JSON, and the first balanced object found
    in surrounding prose. A small repair pass removes trailing commas and
    adds missing closing braces for common truncated outputs.
    """

    def parse_candidate(candidate: str) -> dict[str, Any] | None:
        raw = str(candidate or "").strip()
        if not raw:
            return None
        strict = parse_strict_json_object(raw)
        if strict is not None:
            return strict

        repaired = raw
        for _ in range(3):
            newer = re.sub(r",\s*([}\]])", r"\1", repaired)
            if newer == repaired:
                break
            repaired = newer
        open_braces = repaired.count("{")
        close_braces = repaired.count("}")
        if open_braces > close_braces:
            repaired += "}" * (open_braces - close_braces)
        return parse_strict_json_object(repaired)

    payload = str(text or "").strip()
    if not payload:
        return None

    strict = parse_strict_json_object(payload)
    if strict is not None:
        return strict

    try:
        timeout_s: float | None = float(os.environ.get("RE_SEARCH_TIMEOUT", "60"))
    except (TypeError, ValueError):
        timeout_s = 60.0
    if timeout_s is not None and timeout_s <= 0:
        timeout_s = None

    try:
        match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            payload,
            re.DOTALL,
            timeout=timeout_s,
        )
    except TimeoutError:
        match = None
    if match is not None:
        parsed = parse_candidate(match.group(1))
        if parsed is not None:
            return parsed

    start = payload.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(payload)):
        ch = payload[idx]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                parsed = parse_candidate(payload[start : idx + 1])
                if parsed is not None:
                    return parsed
                break

    return parse_candidate(payload[start:])
