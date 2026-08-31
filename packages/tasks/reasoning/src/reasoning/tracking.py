# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PL tracking outputs -> DAFT ``contextual/{instances,objects,tracking}.json``.

PL's ``rfdetr_tracking.py`` emits rich tracker bookkeeping (``track_id``,
``first_frame``, ``last_frame``, ``confidence_avg``, ``frame_count``) and
per-frame bookkeeping (``width``, ``height``, ``detection_count``) that DAFT's
``additionalProperties: false`` rejects. The converters here whitelist only
DAFT-allowed keys and enforce the stricter per-frame ``format`` enum.

``tracking.json`` (4D MOT — per-frame 3D pose plus optional per-camera 2D
projections) requires ``3d_location`` per detection, which PL's pure-2D
tracker doesn't produce. :func:`to_daft_tracking` is exposed for callers that
*do* have 3D data (external monocular-depth estimator, simulator output,
multi-camera reconstruction). The reasoning service does not auto-emit
``tracking.json`` from pure 2D tracker data; defaulting to ``[0, 0, 0]`` would
silently misrepresent the data.
"""

from __future__ import annotations

from typing import Any

from reasoning.common import DaftConvertError, SceneContext, daft_envelope

_INSTANCE_REQUIRED: tuple[str, ...] = ("object_type", "instance_id", "semantic_id")
_INSTANCE_PASSTHROUGH_KEYS: tuple[str, ...] = ("color", "caption", "images", "videos")

# DAFT's per-frame format enum is stricter than video.json's (no mp4/etc).
_FRAME_FORMAT_ENUM: frozenset[str] = frozenset({"png", "jpg", "jpeg", "bmp"})
_FRAME_REQUIRED: tuple[str, ...] = ("format", "frame_number", "instances")

_DETECTION_REQUIRED: tuple[str, ...] = ("object_id", "bounding_box_2d_tight")
_DETECTION_PASSTHROUGH_KEYS: tuple[str, ...] = ("bounding_box_2d_loose",)

# tracking.json (4D MOT) is a separate schema from objects.json. Required per
# detection: ``object_id`` + ``3d_location`` (3-vector). Optional: 3D bbox
# scale/rotation (3-vectors) and a per-camera ``2d_bounding_box_visible``
# mapping. ``additionalProperties: false`` at every level — keep these
# whitelists tight or strict validation will reject the output.
_TRACK_DET_REQUIRED: tuple[str, ...] = ("object_id", "3d_location")
_TRACK_DET_OPTIONAL_VEC3: tuple[str, ...] = ("3d_bounding_box_scale", "3d_bounding_box_rotation")


def to_daft_instances(obj: dict[str, Any], *, ctx: SceneContext) -> dict[str, Any]:
    """Convert PL's in-memory instances dict into a DAFT ``instances.json`` payload.

    PL's tracker keys ``instances`` by ``object_id`` (e.g. ``"car_7"``). Each
    entry carries tracker bookkeeping (``track_id``, ``first_frame``, ...)
    which is stripped — only ``object_type`` / ``instance_id`` / ``semantic_id``
    (required) and ``color`` / ``caption`` / ``images`` / ``videos`` (optional)
    are kept.

    No ``video_id`` / ``image_id`` is emitted at the top level: the DAFT schema
    treats ``instances.json`` as a scene-level catalog referenced via the
    scene tree and ``additionalProperties: false`` would reject one. ``ctx``
    is still required so date/license/tags carry through the metadata block.

    Image-scene support: schema-valid. ``instances.schema.json`` defines a
    per-instance ``images`` array precisely for this case (parallel to
    ``videos``). When ``ctx.is_image=True`` and the input entry carries
    neither ``images`` nor ``videos``, the converter auto-binds
    ``images=[ctx.media_id]`` so the DAFT cross-reference is explicit.
    Existing ``images`` / ``videos`` lists on the input are preserved
    verbatim (caller may already know about multi-image scenes).

    Validator nuance (verified against ``tao-daft`` v2.9.1): default
    ``tao-daft validate --raw image --strict`` does NOT track
    ``instances`` in its default contextual set (``[objects, tracking]``),
    so a scene with ``instances.json`` present validates cleanly — the
    file is simply ignored by the default flow. Only the explicit
    ``--contextual instances`` invocation rejects it ("Contextual type
    'instances' not valid for raw type 'image'"). The same is true for
    ``--raw video``: ``instances`` is not in the default video contextual
    set either. We treat the file as a downstream-consumer artifact in
    both cases — the schema permits it; the default validator ignores it.
    """
    instances_in = obj.get("instances")
    if not isinstance(instances_in, dict):
        raise DaftConvertError(
            f"instances.json 'instances' must be a dict (got {type(instances_in).__name__})"
        )

    out = daft_envelope("instances", ctx, include_scene_id=False)
    out["instances"] = {k: _to_daft_instance(k, v, scene_ctx=ctx) for k, v in instances_in.items()}
    metadata = out.pop("metadata")
    out["metadata"] = metadata
    return out


def to_daft_objects(obj: dict[str, Any], *, ctx: SceneContext) -> dict[str, Any]:
    """Convert PL's in-memory per-frame detections dict into a DAFT ``objects.json`` payload.

    Strips per-frame bookkeeping (``width``, ``height``, ``detection_count``)
    and per-detection bookkeeping (``instance_id``, ``semantic_id``,
    ``confidence``) since DAFT mirrors those fields via the instances.json
    cross-reference instead of duplicating them per frame.

    Both video AND image scenes are supported. Empirically verified
    against ``tao-daft validate --raw image --strict``: ``objects`` is
    in the *default* contextual set for image scenes
    (``['objects', 'tracking']``), and ``video_id`` is optional in the
    schema. For image scenes we omit the scene-id field at the top
    level (``additionalProperties: false`` would reject ``image_id``,
    and ``video_id`` would be a misleading reference); the cross-link
    to ``image.json`` is implicit via ``metadata.type`` and the flat
    ``contextual/`` directory layout.

    ``instances_source`` is opt-in via ``ctx.instances_source`` for
    both video and image scenes (the field is only emitted when the
    ctx actually carries one, ``is not None``). Image scenes that
    pair an ``instances.json`` companion can set this just like video
    scenes — see :func:`to_daft_instances` for the image cross-ref
    contract.
    """
    frames_in = obj.get("frames")
    if not isinstance(frames_in, dict):
        raise DaftConvertError(
            f"objects.json 'frames' must be a dict (got {type(frames_in).__name__})"
        )

    # ``include_scene_id=not ctx.is_image`` keeps the historic video
    # behavior (top-level ``video_id``) while letting image scenes
    # produce a schema-valid ``objects.json`` with no scene-id field.
    # The schema permits this because ``video_id`` is in ``properties``
    # but NOT in the ``required`` list.
    out = daft_envelope("objects", ctx, include_scene_id=not ctx.is_image)
    if ctx.instances_source is not None:
        out["instances_source"] = ctx.instances_source
    out["frames"] = {k: _to_daft_frame(k, v) for k, v in frames_in.items()}
    metadata = out.pop("metadata")
    out["metadata"] = metadata
    return out


def to_daft_tracking(obj: dict[str, Any], *, ctx: SceneContext) -> dict[str, Any]:
    """Convert per-frame 4D-MOT data into a DAFT ``tracking.json`` payload.

    Input shape mirrors the on-disk schema closely::

        {
            "frames": {
                "frame_000042": [
                    {
                        "object_id": "car_1",
                        "3d_location": [5.2, 10.3, 0.0],
                        "3d_bounding_box_scale": [2.0, 3.5, 2.0],     # optional
                        "3d_bounding_box_rotation": [0, 0, 1.57],     # optional
                        "2d_bounding_box_visible": {                  # optional
                            "cam_001": [100, 200, 300, 500],
                        },
                    },
                    ...
                ],
                ...
            }
        }

    PL's stock ``rfdetr_tracking.py`` is 2D-only and does not produce
    ``3d_location``; the converter therefore enforces it strictly rather than
    defaulting to ``[0, 0, 0]`` (silently lying about data is worse than
    failing loudly). External callers with 3D pose — depth estimators,
    simulator output, multi-camera reconstruction — can drive this directly.

    Image scenes have no temporal axis; calling this with an image
    ``SceneContext`` raises ``DaftConvertError``.
    """
    if ctx.is_image:
        raise DaftConvertError("tracking.json is not produced for image scenes")

    frames_in = obj.get("frames")
    if not isinstance(frames_in, dict):
        raise DaftConvertError(
            f"tracking.json 'frames' must be a dict (got {type(frames_in).__name__})"
        )

    out = daft_envelope("tracking", ctx, include_scene_id=False)
    out["frames"] = {k: _to_daft_track_frame(k, v) for k, v in frames_in.items()}
    metadata = out.pop("metadata")
    out["metadata"] = metadata
    return out


def _to_daft_track_frame(key: str, dets: Any) -> list[dict[str, Any]]:
    if not isinstance(dets, list):
        raise DaftConvertError(f"tracking.json frame {key!r} must be a list of detections")
    return [_to_daft_track_detection(key, d) for d in dets]


def _to_daft_track_detection(frame_key: str, det: Any) -> dict[str, Any]:
    if not isinstance(det, dict):
        raise DaftConvertError(f"tracking.json frame {frame_key!r} detection must be a dict")
    for f in _TRACK_DET_REQUIRED:
        if f not in det:
            raise DaftConvertError(
                f"tracking.json frame {frame_key!r} detection missing required field {f!r}"
            )

    object_id = det["object_id"]
    if not isinstance(object_id, str) or not object_id:
        raise DaftConvertError(
            f"tracking.json frame {frame_key!r} object_id must be a non-empty string"
        )

    out: dict[str, Any] = {
        "object_id": object_id,
        "3d_location": _coerce_vec3(det["3d_location"], frame_key, "3d_location"),
    }
    for f in _TRACK_DET_OPTIONAL_VEC3:
        if f in det:
            out[f] = _coerce_vec3(
                det[f],
                frame_key,
                f,
                nonneg=(f == "3d_bounding_box_scale"),
            )
    if "2d_bounding_box_visible" in det:
        out["2d_bounding_box_visible"] = _coerce_2d_visible(
            det["2d_bounding_box_visible"], frame_key
        )
    return out


def _coerce_vec3(v: Any, frame_key: str, field: str, *, nonneg: bool = False) -> list[float]:
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        raise DaftConvertError(
            f"tracking.json frame {frame_key!r} {field} must be a 3-element list, got {v!r}"
        )
    out: list[float] = []
    for i, x in enumerate(v):
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise DaftConvertError(
                f"tracking.json frame {frame_key!r} {field}[{i}] {x!r} is not a number"
            )
        if nonneg and x < 0:
            raise DaftConvertError(
                f"tracking.json frame {frame_key!r} {field}[{i}] {x!r} must be >= 0"
            )
        out.append(float(x))
    return out


def _coerce_2d_visible(v: Any, frame_key: str) -> dict[str, list[float]]:
    if not isinstance(v, dict):
        raise DaftConvertError(
            f"tracking.json frame {frame_key!r} 2d_bounding_box_visible must be a dict "
            "(cam_id -> bbox)"
        )
    out: dict[str, list[float]] = {}
    for cam_id, bbox in v.items():
        if not isinstance(cam_id, str) or not cam_id:
            raise DaftConvertError(
                f"tracking.json frame {frame_key!r} 2d_bounding_box_visible camera id "
                "must be a non-empty string"
            )
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise DaftConvertError(
                f"tracking.json frame {frame_key!r} cam {cam_id!r} bbox must be a "
                f"4-element list, got {bbox!r}"
            )
        coerced: list[float] = []
        for i, x in enumerate(bbox):
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                raise DaftConvertError(
                    f"tracking.json frame {frame_key!r} cam {cam_id!r} bbox[{i}] "
                    f"{x!r} is not a number"
                )
            coerced.append(float(x))
        out[cam_id] = coerced
    return out


def _to_daft_instance(
    key: str, entry: dict[str, Any], *, scene_ctx: SceneContext
) -> dict[str, Any]:
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

    # Image-scene cross-reference: ``instances.schema.json`` defines a
    # per-instance ``images`` array precisely so an image scene's
    # instances.json can bind each instance back to the originating
    # image(s). Without it, the file would float scene-less inside the
    # contextual/ directory. Auto-bind only when the caller didn't
    # supply ``images`` / ``videos`` already (multi-image scenes or
    # callers that pre-compute bindings stay in control). Video scenes
    # are NOT auto-bound — they have ``video_id`` at the scene-tree
    # level via ``video.json`` and the existing on-disk video contract
    # is already exercised by long-standing pipelines we don't want
    # to perturb here.
    if scene_ctx.is_image and "images" not in out and "videos" not in out:
        out["images"] = [scene_ctx.media_id]
    return out


def _to_daft_frame(key: str, frame: dict[str, Any]) -> dict[str, Any]:
    for f in _FRAME_REQUIRED:
        if f not in frame:
            raise DaftConvertError(f"frame {key!r} missing required field {f!r}")

    fmt = frame["format"]
    if fmt not in _FRAME_FORMAT_ENUM:
        raise DaftConvertError(
            f"frame {key!r} format {fmt!r} not in DAFT enum {sorted(_FRAME_FORMAT_ENUM)}"
        )

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
        raise DaftConvertError(
            f"instance {entry_key!r} {field} {val!r} must be a non-negative integer"
        )
    return int(val)


__all__ = ["to_daft_instances", "to_daft_objects", "to_daft_tracking"]
