# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured event-list -> DAFT ``contextual/events.json``.

DAFT v3 ``events.json`` is the temporal event-detection file: a list of
noteworthy events plus optional ``groups`` for organizing related
events into causal/temporal sequences.

This module is a *pure transformation* — it takes a structured dict
(typically produced by an LLM aggregation pass over per-window dense
captions, but the source is irrelevant) and returns a DAFT-compliant
payload. It makes no network calls, knows nothing about the LLM, and is
fully test-friendly.

The split between "pure converter" and "LLM caller" mirrors the
``msted`` package; see ``reasoning/msted/converter.py`` for the same
design rationale. Callers that want to hand-write events data, replay a
cached aggregation, or use a non-LLM source can import
:func:`to_daft_events` directly without importing :mod:`events.llm`.

Schema constraints enforced by this converter (see
``schemas/contextual/events.schema.json``):

- Top-level: ``additionalProperties: false`` — only ``version``,
  ``video_id``, ``events``, ``groups``, ``metadata``, ``instances_source``
  are allowed.
- Each event: ``additionalProperties: false`` — only ``event_id``,
  ``start_time``, ``end_time``, ``category``, ``sub_category``,
  ``instances``, ``event_caption``, ``severity``, ``group_id`` are
  allowed.
- ``severity`` is enum-restricted to ``low | medium | high | critical``.
- ``start_time`` / ``end_time`` regex: ``^(\\d{2}:)?\\d{2}:\\d{2}(\\.\\d+)?$``.
- Each group: required ``group_id``; optional ``description``.

Use-case agnosticism: the converter never invents fields, never injects
domain vocabulary, never enforces a specific category taxonomy. The
schema is liberal on ``category`` (free-form string); we honor that as
the caller's flexibility hook.
"""

from __future__ import annotations

from typing import Any

from reasoning.common import DaftConvertError, SceneContext, daft_envelope
from reasoning.converter_utils import coerce_timecode, require_nonempty_string
from reasoning.timecodes import timecode_to_seconds

_SEVERITY_ENUM = frozenset({"low", "medium", "high", "critical"})

_DEFAULT_MAX_EVENTS: int = 1_000


def to_daft_events(
    structured: dict[str, Any],
    *,
    ctx: SceneContext,
    duration: float | None = None,
    max_events: int = _DEFAULT_MAX_EVENTS,
    valid_object_ids: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Convert a structured events dict into a DAFT events.json payload.

    ``structured`` is the LLM's parsed JSON (or any equivalent source) and
    must have at minimum::

        {"events": [ {<event>, ...}, ... ]}

    The optional top-level ``groups`` is preserved when present and
    consistent with the events that reference it.

    ``duration`` (when supplied) clamps the high end of every timecode so
    a rounding overshoot in the LLM doesn't trip the schema regex.

    ``valid_object_ids`` is the catalogue of allowed ``instances[]``
    values, typically built from ``contextual/instances.json``. When
    supplied, every ``instances[]`` entry that is not in the catalogue is
    silently dropped from the output (defensive: the prompt forbids
    inventing IDs but a hallucinating LLM should not break the artifact).
    Pass ``None`` to disable the catalogue check entirely.

    Returns a complete DAFT payload ready for :func:`write_daft_json`.

    Raises :class:`DaftConvertError` for image contexts (events.json has
    no image flavor in the v3 schema), missing ``events`` key, malformed
    timecodes, invalid severity values, or more than ``max_events``
    events.
    """
    if ctx.is_image:
        raise DaftConvertError(
            "events.json is not defined for image scenes; events require a temporal axis"
        )

    if not isinstance(structured, dict):
        raise DaftConvertError(f"to_daft_events expects a dict, got {type(structured).__name__}")

    raw_events = structured.get("events")
    if raw_events is None:
        raise DaftConvertError("events 'events' key is required (use [] for an empty list)")
    if not isinstance(raw_events, list):
        raise DaftConvertError(f"events 'events' must be a list, got {type(raw_events).__name__}")
    if len(raw_events) > max_events:
        raise DaftConvertError(
            f"events 'events' has {len(raw_events)} items, exceeds max_events={max_events}"
        )

    seen_ids: set[str] = set()
    used_group_ids: set[str] = set()
    events_out: list[dict[str, Any]] = []
    for idx, ev in enumerate(raw_events):
        item = _to_event(
            ev,
            idx,
            duration=duration,
            valid_object_ids=valid_object_ids,
        )
        if item["event_id"] in seen_ids:
            raise DaftConvertError(
                f"events item[{idx}] has duplicate event_id {item['event_id']!r}; "
                "every event_id must be unique"
            )
        seen_ids.add(item["event_id"])
        gid = item.get("group_id")
        if isinstance(gid, str) and gid:
            used_group_ids.add(gid)
        events_out.append(item)

    groups_out = _to_groups(
        structured.get("groups"),
        used_group_ids=used_group_ids,
    )

    out = daft_envelope("events", ctx, include_scene_id=True)
    if ctx.instances_source:
        out["instances_source"] = ctx.instances_source
    if groups_out is not None:
        out["groups"] = groups_out
    out["events"] = events_out
    return out


def _to_event(
    ev: Any,
    idx: int,
    *,
    duration: float | None,
    valid_object_ids: frozenset[str] | None,
) -> dict[str, Any]:
    """Validate and normalize one event item.

    Splits out the per-event work so error messages carry a stable
    ``event[i]`` prefix that's easy to grep for.
    """
    if not isinstance(ev, dict):
        raise DaftConvertError(f"events item[{idx}] must be a dict, got {type(ev).__name__}")

    event_id = _require_nonempty_string(ev.get("event_id"), field=f"event[{idx}].event_id")

    start_time = _coerce_timecode(ev.get("start_time"), idx, "start_time", duration=duration)
    end_time = _coerce_timecode(ev.get("end_time"), idx, "end_time", duration=duration)

    start_s = timecode_to_seconds(start_time)
    end_s = timecode_to_seconds(end_time)
    if end_s < start_s:
        raise DaftConvertError(
            f"events item[{idx}] has end_time ({end_time!r}, {end_s:.3f}s) "
            f"earlier than start_time ({start_time!r}, {start_s:.3f}s); "
            "reversed time ranges are rejected"
        )

    item: dict[str, Any] = {
        "event_id": event_id,
        "start_time": start_time,
        "end_time": end_time,
    }

    category = ev.get("category")
    if category is not None:
        if not isinstance(category, str) or not category.strip():
            raise DaftConvertError(
                f"event[{idx}].category must be a non-empty string when present, got {category!r}"
            )
        item["category"] = category.strip()

    sub_category = ev.get("sub_category")
    if sub_category is not None:
        if not isinstance(sub_category, list):
            raise DaftConvertError(
                f"event[{idx}].sub_category must be a list when present, "
                f"got {type(sub_category).__name__}"
            )
        sub_out: list[str] = []
        for j, s in enumerate(sub_category):
            if not isinstance(s, str) or not s.strip():
                raise DaftConvertError(
                    f"event[{idx}].sub_category[{j}] must be a non-empty string, got {s!r}"
                )
            sub_out.append(s.strip())
        if sub_out:
            item["sub_category"] = sub_out

    instances = ev.get("instances")
    if instances is not None:
        if not isinstance(instances, list):
            raise DaftConvertError(
                f"event[{idx}].instances must be a list when present, "
                f"got {type(instances).__name__}"
            )
        inst_out: list[str] = []
        for j, oid in enumerate(instances):
            if not isinstance(oid, str) or not oid.strip():
                raise DaftConvertError(
                    f"event[{idx}].instances[{j}] must be a non-empty string, got {oid!r}"
                )
            oid_stripped = oid.strip()
            if valid_object_ids is not None and oid_stripped not in valid_object_ids:
                continue
            inst_out.append(oid_stripped)
        if inst_out:
            item["instances"] = inst_out

    event_caption = ev.get("event_caption")
    if event_caption is not None:
        if not isinstance(event_caption, str) or not event_caption.strip():
            raise DaftConvertError(
                f"event[{idx}].event_caption must be a non-empty string when present, "
                f"got {event_caption!r}"
            )
        item["event_caption"] = event_caption.strip()

    severity = ev.get("severity")
    if severity is not None:
        if not isinstance(severity, str):
            raise DaftConvertError(
                f"event[{idx}].severity must be a string when present, "
                f"got {type(severity).__name__}"
            )
        sev = severity.strip().lower()
        if sev not in _SEVERITY_ENUM:
            raise DaftConvertError(
                f"event[{idx}].severity must be one of {sorted(_SEVERITY_ENUM)}, got {severity!r}"
            )
        item["severity"] = sev

    group_id = ev.get("group_id")
    if group_id is not None:
        if not isinstance(group_id, str) or not group_id.strip():
            raise DaftConvertError(
                f"event[{idx}].group_id must be a non-empty string when present, got {group_id!r}"
            )
        item["group_id"] = group_id.strip()

    return item


def _to_groups(
    raw: Any,
    *,
    used_group_ids: set[str],
) -> list[dict[str, Any]] | None:
    """Validate and normalize the optional top-level ``groups`` list.

    Returns ``None`` to omit the field entirely, or a non-empty list.
    Group entries are kept only when their ``group_id`` is referenced by
    at least one event (silently drop unused groups; the prompt asks for
    this discipline but a generous LLM might over-emit).
    """
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise DaftConvertError(
            f"events 'groups' must be a list when present, got {type(raw).__name__}"
        )
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, g in enumerate(raw):
        if not isinstance(g, dict):
            raise DaftConvertError(f"events groups[{idx}] must be a dict, got {type(g).__name__}")
        group_id = _require_nonempty_string(g.get("group_id"), field=f"groups[{idx}].group_id")
        if group_id in seen:
            raise DaftConvertError(
                f"events groups[{idx}] duplicates group_id {group_id!r}; "
                "every group_id must be unique"
            )
        if group_id not in used_group_ids:
            continue
        seen.add(group_id)
        item: dict[str, Any] = {"group_id": group_id}
        description = g.get("description")
        if description is not None:
            if not isinstance(description, str) or not description.strip():
                raise DaftConvertError(
                    f"groups[{idx}].description must be a non-empty string when present, "
                    f"got {description!r}"
                )
            item["description"] = description.strip()
        out.append(item)
    if not out:
        return None
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
        field=f"event[{idx}].{label}",
        duration=duration,
        duration_field=f"event[{idx}].{label} duration",
    )


def _require_nonempty_string(value: Any, *, field: str) -> str:
    return require_nonempty_string(value, field=field, prefix="events", quote_field=True)


def build_object_id_catalogue(instances_payload: Any) -> frozenset[str] | None:
    """Build the catalogue of valid object_ids from a parsed ``instances.json``.

    The DAFT v3 ``instances.json`` puts the per-instance entries under the
    top-level ``instances`` key as a mapping ``{object_id -> {...}}``;
    every key in that mapping is a valid ``object_id`` for cross-reference
    in events.json. Returns a frozen set for valid payloads so the converter
    can treat it as an immutable lookup table.

    Returns ``None`` when ``instances_payload`` is missing or malformed;
    :func:`to_daft_events` treats ``None`` as "disable catalogue filtering."
    """
    if not isinstance(instances_payload, dict):
        return None
    inst = instances_payload.get("instances")
    if not isinstance(inst, dict):
        return None
    return frozenset(k for k in inst.keys() if isinstance(k, str) and k.strip())


__all__ = ["build_object_id_catalogue", "to_daft_events"]
