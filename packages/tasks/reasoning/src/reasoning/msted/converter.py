# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured-event input -> DAFT ``contextual/msted.json``.

DAFT v3 ``msted.json`` is the Multi-Scale Spatio-Temporal Event Description:
a single per-scene file that captures (1) a scene-level overview, (2) a
non-empty list of temporal-spatial segments, and (3) a flexible
string-keyed object characterizing the most salient event.

This module is a *pure transformation* — it takes a structured dict
(typically produced by an LLM aggregation pass over per-window VLM
captions, but the source is irrelevant) and returns a DAFT-compliant
payload. It makes no network calls, knows nothing about the LLM, and is
fully test-friendly.

The split between "pure converter" and "LLM caller" mirrors the prior
writer-stage separation between payload-shaping and LLM aggregation.
Callers that want to hand-write MSTED data, replay a cached aggregation,
or use a non-LLM source can import :func:`to_daft_msted` directly without
dragging in any LLM infrastructure.

Use-case agnosticism: the converter never invents fields, never injects
domain vocabulary, never enforces a specific category taxonomy. The
schema's ``event_description`` is intentionally a free-form
``Dict[str, str]`` (the schema only requires ``minProperties: 1`` and
``additionalProperties: { "type": "string" }``); we honor that as the
caller's flexibility hook.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from reasoning.common import DaftConvertError, SceneContext, daft_envelope
from reasoning.converter_utils import coerce_timecode, require_nonempty_string
from reasoning.timecodes import timecode_to_seconds

# Maximum number of localization items we'll accept from an upstream
# aggregator. Generous on purpose — a long video legitimately has many
# segments — but capped so a runaway LLM hallucination doesn't write a
# 100k-item file. Tuneable via ``max_segments`` in :func:`to_daft_msted`.
_DEFAULT_MAX_SEGMENTS: int = 10_000


def to_daft_msted(
    structured: dict[str, Any],
    *,
    ctx: SceneContext,
    sources: list[str] | None = None,
    duration: float | None = None,
    max_segments: int = _DEFAULT_MAX_SEGMENTS,
) -> dict[str, Any]:
    """Convert a structured event-description dict into a DAFT MSTED payload.

    ``structured`` must be a mapping with the following keys (the same
    shape the bundled ``msted_default`` prompt asks the LLM to produce,
    so the LLM caller can pass its parsed JSON straight through):

    - ``scene_description`` (str, non-empty)
    - ``temporal_spatial_localization`` (list of dicts, non-empty)
        - each item: ``start``, ``end`` (numeric seconds *or* DAFT
          timecode string), ``description`` (str), optional
          ``spatial_region`` (str)
    - ``event_description`` (dict[str, str], non-empty)

    Numeric ``start``/``end`` values are converted to DAFT timecodes via
    :func:`seconds_to_timecode`; string values are validated against the
    schema regex. ``duration``, when provided, clamps both ends so a
    rounding overshoot in the LLM doesn't trip the validator.

    ``sources`` lands in the optional top-level ``sources`` field — used
    when the caller wants to credit specific upstream files (e.g.
    ``["sidecars/metadata.json"]``).

    Returns a complete DAFT payload ready for :func:`write_daft_json`.

    Raises :class:`DaftConvertError` for image contexts (MSTED has no
    image flavor), missing/empty required fields, malformed segments,
    non-string values in ``event_description``, or more than
    ``max_segments`` segments.
    """
    if ctx.is_image:
        raise DaftConvertError(
            "msted.json is not defined for image scenes; MSTED requires a temporal axis"
        )

    if not isinstance(structured, dict):
        raise DaftConvertError(f"to_daft_msted expects a dict, got {type(structured).__name__}")

    scene_description = _require_nonempty_string(
        structured.get("scene_description"),
        field="scene_description",
    )

    raw_segments = structured.get("temporal_spatial_localization")
    if not isinstance(raw_segments, list):
        raise DaftConvertError(
            "msted 'temporal_spatial_localization' must be a list, "
            f"got {type(raw_segments).__name__}"
        )
    if not raw_segments:
        raise DaftConvertError(
            "msted 'temporal_spatial_localization' must be non-empty (schema minItems: 1)"
        )
    if len(raw_segments) > max_segments:
        raise DaftConvertError(
            f"msted 'temporal_spatial_localization' has {len(raw_segments)} segments, "
            f"exceeds max_segments={max_segments}"
        )

    segments_out = [
        _to_segment(seg, idx, duration=duration) for idx, seg in enumerate(raw_segments)
    ]

    event_description = _to_event_description(structured.get("event_description"))

    out = daft_envelope("msted", ctx, include_scene_id=True)
    if sources is not None:
        out["sources"] = _coerce_sources(sources)
    out["scene_description"] = scene_description
    out["temporal_spatial_localization"] = segments_out
    out["event_description"] = event_description
    return out


def _to_segment(seg: Any, idx: int, *, duration: float | None) -> dict[str, Any]:
    """Validate and normalize one ``temporal_spatial_localization`` item.

    Splitting this out keeps :func:`to_daft_msted`'s body readable and
    gives error messages a stable ``segment[i]`` prefix that's easy to
    grep for in pipeline logs."""
    if not isinstance(seg, dict):
        raise DaftConvertError(f"msted segment[{idx}] must be a dict, got {type(seg).__name__}")

    start = _coerce_timecode(seg.get("start"), idx, "start", duration=duration)
    end = _coerce_timecode(seg.get("end"), idx, "end", duration=duration)

    # Monotonicity: DAFT msted.schema.json doesn't enforce ``end >= start``
    # on temporal_spatial_localization items, so an LLM that swaps the
    # fields (or emits the regression "00:00-00:04.839" pattern that the
    # default prompt now warns against) would otherwise ship a reversed
    # range. Catch it here so the segment is dropped at the converter
    # boundary with a stable, greppable error rather than silently
    # corrupting downstream MSTED consumers.
    start_s = timecode_to_seconds(start)
    end_s = timecode_to_seconds(end)
    if end_s < start_s:
        raise DaftConvertError(
            f"msted segment[{idx}] has end ({end!r}, {end_s:.3f}s) earlier "
            f"than start ({start!r}, {start_s:.3f}s); reversed time ranges "
            "are rejected to keep downstream temporal consumers well-defined"
        )

    description = _require_nonempty_string(
        seg.get("description"),
        field=f"segment[{idx}].description",
    )

    item: dict[str, Any] = {
        "start": start,
        "end": end,
        "description": description,
    }

    spatial_region = seg.get("spatial_region")
    if spatial_region is not None:
        if not isinstance(spatial_region, str):
            raise DaftConvertError(
                f"msted segment[{idx}].spatial_region must be a string, "
                f"got {type(spatial_region).__name__}"
            )
        spatial_region = spatial_region.strip()
        if spatial_region:
            item["spatial_region"] = spatial_region

    return item


def _to_event_description(raw: Any) -> dict[str, str]:
    """Validate and normalize the free-form ``event_description`` dict.

    The DAFT schema is liberal here (``additionalProperties: {type:
    "string"}``, ``minProperties: 1``) — the only invariant is that
    every value is a string and the dict is non-empty. We honor that
    exactly: no key whitelist, no value transformations beyond strip().
    Empty values are dropped (an empty string passes the type check but
    is semantically garbage)."""
    if not isinstance(raw, dict):
        raise DaftConvertError(
            f"msted 'event_description' must be a dict, got {type(raw).__name__}"
        )
    if not raw:
        raise DaftConvertError(
            "msted 'event_description' must have at least one key (schema minProperties: 1)"
        )

    out: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key.strip():
            raise DaftConvertError(
                f"msted 'event_description' keys must be non-empty strings, got {key!r}"
            )
        if not isinstance(value, str):
            raise DaftConvertError(
                f"msted 'event_description'[{key!r}] must be a string "
                f"(schema requires string-typed values), got {type(value).__name__}"
            )
        v = value.strip()
        if v:
            out[key.strip()] = v

    if not out:
        raise DaftConvertError(
            "msted 'event_description' is empty after dropping blank values "
            "(at least one key must have a non-empty string value)"
        )
    return out


def _coerce_sources(sources: Any) -> list[str]:
    """Normalize the optional top-level ``sources`` field.

    The schema declares ``sources`` as a list of strings; we accept any
    iterable of strings, strip whitespace, and reject empty entries
    (silently producing an empty list would mask a config bug)."""
    if isinstance(sources, str):
        raise DaftConvertError("msted 'sources' must be a list of strings, not a single string")
    if not isinstance(sources, Iterable):
        raise DaftConvertError(f"msted 'sources' must be iterable, got {type(sources).__name__}")
    out: list[str] = []
    for i, s in enumerate(sources):
        if not isinstance(s, str):
            raise DaftConvertError(f"msted 'sources'[{i}] must be a string, got {type(s).__name__}")
        s = s.strip()
        if not s:
            raise DaftConvertError(f"msted 'sources'[{i}] is empty after strip()")
        out.append(s)
    if not out:
        raise DaftConvertError("msted 'sources' is empty (omit the field instead)")
    return out


def _coerce_timecode(
    value: Any,
    idx: int,
    label: str,
    *,
    duration: float | None,
) -> str:
    return coerce_timecode(
        value,
        field=f"msted segment[{idx}].{label}",
        duration=duration,
        duration_field="msted duration",
    )


def _require_nonempty_string(value: Any, *, field: str) -> str:
    return require_nonempty_string(value, field=field, prefix="msted", quote_field=True)


__all__ = ["to_daft_msted"]
