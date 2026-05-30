# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auto-labeling tracking outputs -> DAFT ``contextual/instances.json`` + ``contextual/objects.json``.

The auto-labeling ``rfdetr_tracking.py`` emits rich tracker bookkeeping (``track_id``,
``first_frame``, ``last_frame``, ``confidence_avg``, ``frame_count``) and
per-frame bookkeeping (``width``, ``height``, ``detection_count``) that DAFT's
``additionalProperties: false`` rejects. The converters here whitelist only
DAFT-allowed keys and enforce the stricter per-frame ``format`` enum.
"""

from __future__ import annotations

from typing import Any

from daft_export.common import DAFT_VERSION, DaftConvertError, metadata_block

_INSTANCE_REQUIRED: tuple[str, ...] = ("object_type", "instance_id", "semantic_id")
_INSTANCE_PASSTHROUGH_KEYS: tuple[str, ...] = ("color", "caption", "images", "videos")

# DAFT's per-frame format enum is stricter than video.json's (no mp4/etc).
_FRAME_FORMAT_ENUM: frozenset[str] = frozenset({"png", "jpg", "jpeg", "bmp"})
_FRAME_REQUIRED: tuple[str, ...] = ("format", "frame_number", "instances")

_DETECTION_REQUIRED: tuple[str, ...] = ("object_id", "bounding_box_2d_tight")
_DETECTION_PASSTHROUGH_KEYS: tuple[str, ...] = ("bounding_box_2d_loose",)


def to_daft_instances(obj: dict[str, Any]) -> dict[str, Any]:
    """Convert the auto-labeling in-memory instances dict into a DAFT ``instances.json`` payload.

    The auto-labeling tracker keys ``instances`` by ``object_id`` (e.g. ``"car_7"``). Each
    entry carries tracker bookkeeping (``track_id``, ``first_frame``, ...)
    which is stripped — only ``object_type`` / ``instance_id`` / ``semantic_id``
    (required) and ``color`` / ``caption`` / ``images`` / ``videos`` (optional)
    are kept.

    No ``video_id`` is emitted at the top level: the DAFT schema treats
    ``instances.json`` as a scene-level catalog referenced via the scene tree
    and ``additionalProperties: false`` would reject it.
    """
    instances_in = obj.get("instances")
    if not isinstance(instances_in, dict):
        raise DaftConvertError(f"instances.json 'instances' must be a dict (got {type(instances_in).__name__})")
    return {
        "version": DAFT_VERSION,
        "instances": {k: _to_daft_instance(k, v) for k, v in instances_in.items()},
        "metadata": metadata_block("instances"),
    }


def to_daft_objects(obj: dict[str, Any], *, video_id: str) -> dict[str, Any]:
    """Convert the auto-labeling in-memory per-frame detections dict into a DAFT ``objects.json`` payload.

    Strips per-frame bookkeeping (``width``, ``height``, ``detection_count``)
    and per-detection bookkeeping (``instance_id``, ``semantic_id``,
    ``confidence``) since DAFT mirrors those fields via the instances.json
    cross-reference instead of duplicating them per frame.
    """
    frames_in = obj.get("frames")
    if not isinstance(frames_in, dict):
        raise DaftConvertError(f"objects.json 'frames' must be a dict (got {type(frames_in).__name__})")
    return {
        "version": DAFT_VERSION,
        "video_id": video_id,
        "frames": {k: _to_daft_frame(k, v) for k, v in frames_in.items()},
        "metadata": metadata_block("objects"),
    }


def _to_daft_instance(key: str, entry: dict[str, Any]) -> dict[str, Any]:
    for f in _INSTANCE_REQUIRED:
        if f not in entry:
            raise DaftConvertError(f"instance {key!r} missing required field {f!r}")
    if not isinstance(entry["object_type"], str):
        raise DaftConvertError(f"instance {key!r} object_type must be a string")

    out: dict[str, Any] = {
        "object_type": entry["object_type"],
        "instance_id": _coerce_nonneg_int(entry["instance_id"], key, "instance_id"),
        "semantic_id": _coerce_nonneg_int(entry["semantic_id"], key, "semantic_id"),
    }
    for f in _INSTANCE_PASSTHROUGH_KEYS:
        if f in entry:
            out[f] = entry[f]
    return out


def _to_daft_frame(key: str, frame: dict[str, Any]) -> dict[str, Any]:
    for f in _FRAME_REQUIRED:
        if f not in frame:
            raise DaftConvertError(f"frame {key!r} missing required field {f!r}")

    fmt = frame["format"]
    if fmt not in _FRAME_FORMAT_ENUM:
        raise DaftConvertError(f"frame {key!r} format {fmt!r} not in DAFT enum {sorted(_FRAME_FORMAT_ENUM)}")

    dets_in = frame["instances"]
    if not isinstance(dets_in, list):
        raise DaftConvertError(f"frame {key!r} 'instances' must be a list")

    return {
        "format": fmt,
        "frame_number": frame["frame_number"],
        "instances": [_to_daft_detection(key, d) for d in dets_in],
    }


def _to_daft_detection(frame_key: str, det: dict[str, Any]) -> dict[str, Any]:
    for f in _DETECTION_REQUIRED:
        if f not in det:
            raise DaftConvertError(f"frame {frame_key!r} detection missing required field {f!r}")
    out: dict[str, Any] = {
        "object_id": det["object_id"],
        "bounding_box_2d_tight": det["bounding_box_2d_tight"],
    }
    for f in _DETECTION_PASSTHROUGH_KEYS:
        if f in det:
            out[f] = det[f]
    return out


def _coerce_nonneg_int(val: Any, entry_key: str, field: str) -> int:
    # Rejects bool (``isinstance(True, int)`` is True in Python) and negatives.
    if isinstance(val, bool) or not isinstance(val, int) or val < 0:
        raise DaftConvertError(f"instance {entry_key!r} {field} {val!r} must be a non-negative integer")
    return val


__all__ = ["to_daft_instances", "to_daft_objects"]
