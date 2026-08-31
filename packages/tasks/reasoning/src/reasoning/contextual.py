# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PL VLM outputs -> DAFT ``contextual/{video,events,image}.json``.

PL's VLM emits bookkeeping extras (``source_video``, ``generated_at``, and
historically ``description``) that DAFT's ``additionalProperties: false``
rejects, and float fps values that need rounding. The converters whitelist
DAFT-allowed fields, convert timestamps to ``HH:MM:SS.ms`` timecodes, and
clamp events to the video duration.

For image-input scenes (a single frame), DAFT v3.0 uses a separate
``image.json`` schema instead of ``video.json`` (no fps/duration, no events).
:func:`to_daft_image` mirrors :func:`to_daft_video` for that path. Events are
video-only in the VLM JSON stage.

Every converter takes a :class:`~reasoning.common.SceneContext` so the
DAFT envelope (version, scene id, metadata block) is built once per scene
in :func:`~reasoning.common.daft_envelope`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from reasoning.common import DaftConvertError, SceneContext, daft_envelope
from reasoning.converter_utils import clamp_nonnegative_seconds
from reasoning.id_translator import (
    ground_vlm_ids,
    index_instances_by_suffix,
    strip_ungrounded_id_annotations,
)
from reasoning.timecodes import seconds_to_timecode, timecode_to_seconds

_VIDEO_FORMAT_ENUM: frozenset[str] = frozenset({"mp4", "avi", "mov", "mkv", "webm"})
_IMAGE_FORMAT_ENUM: frozenset[str] = frozenset({"png", "jpg", "jpeg", "bmp", "tiff", "webp"})
_MAX_FPS: int = 240

# Optional video-level keys that pass through from PL verbatim.
_VIDEO_PASSTHROUGH_KEYS: tuple[str, ...] = (
    "rectified",
    "scenario_info",
    "scene_description",
    "event_summary",
)

# Optional image-level keys that pass through from PL verbatim. Mirrors
# ``_VIDEO_PASSTHROUGH_KEYS`` except DAFT image.json has a single ``caption``
# field rather than ``scene_description`` + ``event_summary``; the runner is
# responsible for collapsing those when producing image scenes.
_IMAGE_PASSTHROUGH_KEYS: tuple[str, ...] = (
    "rectified",
    "scenario_info",
    "caption",
    "timestamp",
)

# Optional per-event keys that pass through verbatim (event_id, start_time,
# end_time, event_caption are handled explicitly elsewhere).
_EVENT_PASSTHROUGH_KEYS: tuple[str, ...] = (
    "category",
    "sub_category",
    "instances",
    "severity",
    "group_id",
)


def to_daft_video(
    obj: dict[str, Any],
    *,
    ctx: SceneContext,
    instances_keys: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Convert PL's in-memory video dict into a DAFT ``video.json`` payload.

    Required fields (``format``, ``fps``, ``duration``, ``height``, ``width``)
    must be present; missing ones raise ``DaftConvertError``. ``fps`` is rounded
    to the integer DAFT expects. All PL-internal extras are stripped.

    When ``instances_keys`` is provided, ungrounded ``{id: <n>}`` annotations
    are stripped from ``scene_description`` and ``event_summary`` (same
    policy applied to per-event ``event_caption`` in :func:`to_daft_events`).
    """
    if ctx.is_image:
        raise DaftConvertError("to_daft_video called with an image SceneContext; use to_daft_image")

    fmt = obj.get("format")
    if fmt not in _VIDEO_FORMAT_ENUM:
        raise DaftConvertError(
            f"video.json format {fmt!r} not in DAFT enum {sorted(_VIDEO_FORMAT_ENUM)}"
        )

    out = daft_envelope("video", ctx)
    out["format"] = fmt
    out["fps"] = _coerce_fps(obj.get("fps"))
    out["duration"] = _coerce_duration(obj.get("duration"))
    out["height"] = _coerce_positive_int(obj.get("height"), "height")
    out["width"] = _coerce_positive_int(obj.get("width"), "width")

    for k in _VIDEO_PASSTHROUGH_KEYS:
        if k in obj:
            out[k] = obj[k]

    if instances_keys is not None:
        suffix_to_key = index_instances_by_suffix(instances_keys)
        for k in ("scene_description", "event_summary"):
            v = out.get(k)
            if isinstance(v, str):
                out[k] = strip_ungrounded_id_annotations(v, suffix_to_key)

    # ``daft_envelope`` puts metadata immediately after version+scene_id; rotate
    # it to the end so on-disk diffs read top-down (anchor → fields → metadata).
    metadata = out.pop("metadata")
    out["metadata"] = metadata
    return out


def to_daft_image(obj: dict[str, Any], *, ctx: SceneContext) -> dict[str, Any]:
    """Convert PL's in-memory image dict into a DAFT ``image.json`` payload.

    The image counterpart to :func:`to_daft_video`. Required fields
    (``format``, ``height``, ``width``) must be present; missing ones raise
    ``DaftConvertError``. There is no ``fps``/``duration`` (still frame) and
    Image scenes do not produce ``events.json`` in the VLM JSON stage.

    The ``format`` enum (png/jpg/jpeg/bmp/tiff/webp) differs from video; the
    runner derives it from the input file's suffix. ``image_id`` (instead of
    ``video_id``) is the DAFT scene-anchor field for image scenes.
    """
    if not ctx.is_image:
        raise DaftConvertError("to_daft_image called with a video SceneContext; use to_daft_video")

    fmt = obj.get("format")
    if fmt not in _IMAGE_FORMAT_ENUM:
        raise DaftConvertError(
            f"image.json format {fmt!r} not in DAFT enum {sorted(_IMAGE_FORMAT_ENUM)}"
        )

    out = daft_envelope("image", ctx)
    out["format"] = fmt
    out["height"] = _coerce_positive_int(obj.get("height"), "height")
    out["width"] = _coerce_positive_int(obj.get("width"), "width")
    for k in _IMAGE_PASSTHROUGH_KEYS:
        if k in obj:
            out[k] = obj[k]

    metadata = out.pop("metadata")
    out["metadata"] = metadata
    return out


def to_daft_events(
    obj: dict[str, Any] | list[dict[str, Any]],
    *,
    ctx: SceneContext,
    duration: float | None = None,
    instances_keys: Iterable[str] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Convert PL's in-memory events data into a DAFT ``events.json`` payload.

    Accepts either the full PL dict (``{"events": [...]}``) or a bare list of
    event dicts. Per-event ``start_time`` / ``end_time`` are required and
    converted to ``HH:MM:SS.ms`` timecodes. For compatibility with older
    emitters, ``start_time_sec`` / ``end_time_sec`` are accepted as seconds
    aliases, for example ``{"start_time_sec": 2.0, "end_time_sec": 5.0}``
    becomes ``"00:02"`` / ``"00:05"``. If both forms are present, the canonical
    ``start_time`` / ``end_time`` values take precedence over the ``*_sec``
    aliases. If ``duration`` is given, times are clamped to ``[0, duration]``.
    ``description`` is remapped to ``event_caption`` when the event doesn't
    already carry one. All other per-event extras are stripped.

    When ``instances_keys`` is provided (e.g. the keys of the scene's
    ``instances.json``), each event's ``instances`` list is translated from
    the VLM's ``id_<n>`` format to the tracker's ``<class>_<n>`` format via
    :func:`reasoning.id_translator.ground_vlm_ids`. Ungrounded IDs (VLM
    OCR errors / hallucinations) are dropped with a warning. Pass ``None``
    (the default) to skip translation entirely — useful for tests and for
    callers that have no tracker output to ground against.

    Image scenes have no temporal axis; calling this with an image
    ``SceneContext`` raises ``DaftConvertError``.
    """
    if ctx.is_image:
        raise DaftConvertError("events.json is not produced for image scenes")

    if isinstance(obj, dict):
        events_in = obj.get("events", [])
        if not isinstance(events_in, list):
            raise DaftConvertError(
                f"events.json 'events' must be a list, got {type(events_in).__name__}"
            )
    else:
        events_in = obj
    if not isinstance(events_in, list):
        raise DaftConvertError(f"events.json input must be a list, got {type(events_in).__name__}")

    log = logger or logging.getLogger(__name__)
    events_out: list[dict[str, Any]] = []
    for idx, ev in enumerate(events_in):
        try:
            events_out.append(_to_daft_event(ev, duration=duration))
        except (AttributeError, DaftConvertError, TypeError, ValueError) as exc:
            log.warning("[events] skipping malformed event[%d]: %s", idx, exc)

    if instances_keys is not None:
        suffix_to_key = index_instances_by_suffix(instances_keys)
        # When the catalogue is empty, every VLM id in the scene is
        # ungrounded; emit a single aggregate warning here so the per-id
        # warning in ground_vlm_ids() doesn't spam.
        if not suffix_to_key and any(ev.get("instances") for ev in events_out):
            log.warning(
                "[id_translator] instances.json missing/empty; dropping all "
                "event instance refs (det/track likely disabled)."
            )
        total_in = total_kept = 0
        for ev in events_out:
            ids = ev.get("instances")
            if isinstance(ids, list):
                total_in += len(ids)
                ev["instances"] = ground_vlm_ids(ids, suffix_to_key, logger=log)
                total_kept += len(ev["instances"])
            # Same catalogue used for the structured side, so prose and the
            # ``instances`` array stay consistent: a number dropped from
            # ``instances`` is also dropped from any ``{id: <n>}`` clause
            # in the caption.
            cap = ev.get("event_caption")
            if isinstance(cap, str):
                ev["event_caption"] = strip_ungrounded_id_annotations(cap, suffix_to_key)
        if total_in > 0:
            log.info(
                "[id_translator] events.json: %d/%d instance refs translated, %d dropped",
                total_kept,
                total_in,
                total_in - total_kept,
            )

    out = daft_envelope("events", ctx)
    if ctx.instances_source is not None:
        out["instances_source"] = ctx.instances_source
    out["events"] = events_out
    metadata = out.pop("metadata")
    out["metadata"] = metadata
    return out


def _as_number(v: Any, field: str) -> float:
    # Rejects bool (``isinstance(True, int)`` is True in Python), None, and
    # strings. Used by every numeric scene-field coercion (video + image) for
    # a consistent "not a number" error shape.
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise DaftConvertError(f"{field} {v!r} is not a number")
    return float(v)


def _coerce_fps(fps: Any) -> int:
    ifps = round(_as_number(fps, "fps"))
    if ifps < 1 or ifps > _MAX_FPS:
        raise DaftConvertError(f"fps {ifps} out of DAFT range 1..{_MAX_FPS}")
    return ifps


def _coerce_duration(dur: Any) -> float:
    val = _as_number(dur, "duration")
    if val <= 0:
        raise DaftConvertError(f"duration {dur!r} must be positive")
    return val


def _coerce_positive_int(val: Any, field: str) -> int:
    # Accept ints and whole-number floats (VLM occasionally emits 720.0).
    num = _as_number(val, field)
    ival = int(num)
    if ival < 1 or ival != num:
        raise DaftConvertError(f"{field} {val!r} must be a positive integer")
    return ival


def _as_seconds(v: Any, eid: str, field: str) -> float:
    if isinstance(v, bool):
        raise DaftConvertError(f"event {eid!r} {field} {v!r} is not a number or timecode")
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return timecode_to_seconds(v)
        except ValueError as e:
            raise DaftConvertError(
                f"event {eid!r} {field} {v!r} is not a number or valid timecode"
            ) from e
    raise DaftConvertError(f"event {eid!r} {field} {v!r} is not a number or timecode")


def _clamp(s: float, duration: float | None) -> float:
    return clamp_nonnegative_seconds(s, duration)


def _to_daft_event(ev: dict[str, Any], *, duration: float | None) -> dict[str, Any]:
    if "event_id" not in ev:
        raise DaftConvertError("event missing required field 'event_id'")
    eid = str(ev["event_id"])
    # Prefer canonical time keys, falling back to older *_sec aliases. _as_seconds accepts
    # numbers or timecodes for either form; missing values raise DaftConvertError for eid,
    # and parsed seconds are clamped to the video duration below.
    start_value = ev.get("start_time", ev.get("start_time_sec"))
    end_value = ev.get("end_time", ev.get("end_time_sec"))
    for k, v in (("start_time", start_value), ("end_time", end_value)):
        if v is None:
            raise DaftConvertError(f"event {eid!r} missing required field {k!r}")

    start_s = _clamp(_as_seconds(start_value, eid, "start_time"), duration)
    end_s = _clamp(_as_seconds(end_value, eid, "end_time"), duration)

    # Monotonicity: DAFT events.json doesn't enforce ``end >= start``
    # at the schema layer (both fields are independent timecodes), so
    # detector quirks (clamp-to-zero on truncated tails, swapped fields)
    # would otherwise propagate downstream and quietly poison MSTED /
    # causal-linkage stages that consume this file. Fail fast at the
    # converter boundary with a stable error code so a single bad event
    # is easy to triage in pipeline logs (other valid events in the same
    # file still flow through; the per-event raise is caught upstream
    # by ``to_daft_events`` and surfaced as a structured warning).
    if end_s < start_s:
        raise DaftConvertError(
            f"event {eid!r} has end_time ({end_s:.3f}s) earlier than "
            f"start_time ({start_s:.3f}s); reversed time ranges are "
            "rejected to keep downstream temporal stages well-defined"
        )

    out: dict[str, Any] = {
        "event_id": ev["event_id"],
        "start_time": seconds_to_timecode(start_s),
        "end_time": seconds_to_timecode(end_s),
    }
    for k in _EVENT_PASSTHROUGH_KEYS:
        if k in ev:
            out[k] = ev[k]

    # event_caption wins if present; fall back to older `description` key.
    caption = ev.get("event_caption")
    if caption is None:
        caption = ev.get("description")
    if caption is not None:
        out["event_caption"] = caption
    return out


__all__ = ["to_daft_events", "to_daft_image", "to_daft_video"]
