# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Backend-agnostic per-track crop extraction (PAS video flow).

Runs as a post-step over the canonical ``contextual/objects.json`` +
``contextual/instances.json`` that any tracker backend writes, so it does not
depend on SAM3 internals. For each kept track it samples up to ``crops_per_track``
frames, crops the (padded, clamped) person box, and writes
``sidecars/<crop_subdir>/track_XXXX/crop_*.jpg`` plus a ``tracks.json`` seam that
the Visual QA fan-out / PAS task consume.

The frame-selection and box math are pure and unit-tested; only ``write_crops``
and ``extract_track_crops`` touch the filesystem / video decoder.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from core import ScenePaths, is_image_path, read_json, resolve_sidecar, write_json
from pydantic import BaseModel, ConfigDict, Field

from detection_and_tracking.backends.runtime import as_float, import_optional
from detection_and_tracking.config import DetectionAndTrackingConfig
from detection_and_tracking.media import iter_selected_video_frames_bgr

Box = tuple[int, int, int, int]


class SelectedFrame(BaseModel):
    """One sampled frame for a track, with its padded/clamped integer box."""

    model_config = ConfigDict(extra="forbid")

    frame_number: int
    box: Box


class TrackCropPlan(BaseModel):
    """A track's selected frames and metadata, ready for cropping."""

    model_config = ConfigDict(extra="forbid")

    track_id: int
    object_id: str
    object_type: str
    detection_score: float | None = None
    first_frame: int
    last_frame: int
    duration_sec: float | None = None
    frames: list[SelectedFrame] = Field(default_factory=list)


class CropRecord(BaseModel):
    """Per-track crop output, mergeable into the PAS ``track_inputs.json`` seam."""

    model_config = ConfigDict(extra="forbid")

    track_id: int
    object_id: str
    object_type: str
    detection_score: float | None = None
    first_frame: int
    last_frame: int
    duration_sec: float | None = None
    n_crops: int
    crop_dir: str
    crops: list[str] = Field(default_factory=list)


def sample_frame_numbers(available: list[int], max_count: int) -> list[int]:
    """Evenly sample up to ``max_count`` frame numbers, preserving order."""
    ordered = sorted(set(available))
    n = len(ordered)
    if max_count <= 0 or n == 0:
        return []
    if n <= max_count:
        return ordered
    if max_count == 1:
        return [ordered[n // 2]]
    positions = sorted({round(i * (n - 1) / (max_count - 1)) for i in range(max_count)})
    return [ordered[pos] for pos in positions]


def pad_and_clamp_box(
    box: list[float] | tuple[float, ...],
    *,
    width: int,
    height: int,
    padding: float,
) -> Box | None:
    """Expand a box by ``padding`` (fraction) and clamp to the frame, or ``None``."""
    if len(box) < 4 or width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    dx = (x2 - x1) * padding / 2.0
    dy = (y2 - y1) * padding / 2.0
    nx1 = max(0, math.floor(x1 - dx))
    ny1 = max(0, math.floor(y1 - dy))
    nx2 = min(width, math.ceil(x2 + dx))
    ny2 = min(height, math.ceil(y2 + dy))
    if nx2 - nx1 <= 0 or ny2 - ny1 <= 0:
        return None
    return (nx1, ny1, nx2, ny2)


def plan_track_crops(
    objects_payload: dict[str, Any],
    instances_payload: dict[str, Any],
    *,
    crop_classes: tuple[str, ...] = (),
    crops_per_track: int = 16,
    crop_padding: float = 0.0,
    min_crop_size: int = 0,
    min_detection_score: float = 0.0,
    min_track_seconds: float = 0.0,
) -> list[TrackCropPlan]:
    """
    Build per-track crop plans from canonical tracking artifacts (pure).

    Args:
        objects_payload: Parsed ``objects.json`` (``{"frames": [...]}``).
        instances_payload: Parsed ``instances.json`` (``{"instances": [...]}``).
        crop_classes: Restrict to these ``object_type`` values (empty = all).
        crops_per_track: Max crops sampled per track.
        crop_padding: Box expansion fraction before clamping.
        min_crop_size: Drop crops whose shorter side is below this many pixels.
        min_detection_score: Drop tracks whose detection score is below this
            (``0`` disables the gate). Mirrors the legacy v1 ``min_score`` gate
            and prunes low-confidence spurious tracks before PAS.
        min_track_seconds: Drop tracks shorter than this duration (``0`` disables
            the gate). Mirrors the legacy v1 ``min_track_seconds`` gate. Apply
            only when planning over full videos; leave disabled for already-short
            pre-chunked clips where it would drop legitimate people.

    Returns:
        One ``TrackCropPlan`` per kept track that has at least one valid crop.
    """
    wanted = set(crop_classes)
    instances = {
        str(item["object_id"]): item
        for item in instances_payload.get("instances", [])
        if isinstance(item, dict) and "object_id" in item
    }

    # object_id -> {frame_number: (box, width, height)}
    boxes_by_object: dict[str, dict[int, tuple[list[float], int, int]]] = {}
    for frame in objects_payload.get("frames", []):
        if not isinstance(frame, dict):
            continue
        frame_number = int(frame.get("frame_number", 0))
        width = int(frame.get("width", 0))
        height = int(frame.get("height", 0))
        for inst in frame.get("instances", []):
            if not isinstance(inst, dict):
                continue
            object_id = str(inst.get("object_id", ""))
            box = inst.get("bounding_box_2d_tight")
            if not object_id or not isinstance(box, list):
                continue
            boxes_by_object.setdefault(object_id, {})[frame_number] = (box, width, height)

    plans: list[TrackCropPlan] = []
    for object_id, summary in instances.items():
        object_type = str(summary.get("object_type", ""))
        if wanted and object_type not in wanted:
            continue
        if min_detection_score > 0.0:
            score = _optional_score(summary.get("detection_score"))
            if score is None or score < min_detection_score:
                continue
        if min_track_seconds > 0.0:
            duration = _duration(summary)
            if duration is None or duration < min_track_seconds:
                continue
        per_frame = boxes_by_object.get(object_id)
        if not per_frame:
            continue
        selected = _select_frames(
            per_frame,
            crops_per_track=crops_per_track,
            crop_padding=crop_padding,
            min_crop_size=min_crop_size,
        )
        if not selected:
            continue
        plans.append(
            TrackCropPlan(
                track_id=int(summary.get("track_id", 0)),
                object_id=object_id,
                object_type=object_type,
                detection_score=_optional_score(summary.get("detection_score")),
                first_frame=int(summary.get("first_frame", selected[0].frame_number)),
                last_frame=int(summary.get("last_frame", selected[-1].frame_number)),
                duration_sec=_duration(summary),
                frames=selected,
            )
        )
    return plans


def write_crops(
    media_path: Path,
    plans: list[TrackCropPlan],
    *,
    output_root: Path,
    crop_subdir: str,
    crop_format: str,
    cv2: Any,
) -> list[CropRecord]:
    """Decode the media once and write each plan's sampled crops (I/O)."""
    needed: dict[int, list[tuple[int, Box]]] = {}
    for plan_idx, plan in enumerate(plans):
        for frame in plan.frames:
            needed.setdefault(frame.frame_number, []).append((plan_idx, frame.box))
    if not needed:
        return []

    crops_by_plan: dict[int, list[str]] = {idx: [] for idx in range(len(plans))}
    # crop_subdir is operator-supplied (config/CLI); contain it under the scene
    # sidecars root so an absolute path or ``..`` cannot write outside the scene.
    crops_root = resolve_sidecar(output_root, crop_subdir)
    # Resolve a unique on-disk folder per plan. track_id is the readable key, but
    # different plans can share a track_id (e.g. a missing id defaulting to 0),
    # which would otherwise collide; disambiguate those with the plan index.
    plan_dirnames: list[str] = []
    used_dirnames: set[str] = set()
    for plan_idx, plan in enumerate(plans):
        dirname = f"track_{plan.track_id:04d}"
        if dirname in used_dirnames:
            dirname = f"track_{plan.track_id:04d}_plan_{plan_idx:04d}"
        used_dirnames.add(dirname)
        plan_dirnames.append(dirname)
        (crops_root / dirname).mkdir(parents=True, exist_ok=True)

    for frame_number, frame_bgr in _iter_media_frames(media_path, sorted(needed), cv2=cv2):
        height, width = frame_bgr.shape[:2]
        for plan_idx, box in needed.get(frame_number, []):
            x1, y1, x2, y2 = box
            x2 = min(x2, width)
            y2 = min(y2, height)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame_bgr[y1:y2, x1:x2]
            rel = f"{crop_subdir}/{plan_dirnames[plan_idx]}/crop_{frame_number:06d}.{crop_format}"
            out_path = output_root / rel
            if not cv2.imwrite(str(out_path), crop):
                raise RuntimeError(f"Failed to write crop: {out_path}")
            crops_by_plan[plan_idx].append(rel)

    records: list[CropRecord] = []
    for plan_idx, plan in enumerate(plans):
        crops = sorted(crops_by_plan[plan_idx])
        if not crops:
            continue
        records.append(
            CropRecord(
                track_id=plan.track_id,
                object_id=plan.object_id,
                object_type=plan.object_type,
                detection_score=plan.detection_score,
                first_frame=plan.first_frame,
                last_frame=plan.last_frame,
                duration_sec=plan.duration_sec,
                n_crops=len(crops),
                crop_dir=f"{crop_subdir}/{plan_dirnames[plan_idx]}",
                crops=crops,
            )
        )
    return records


def extract_track_crops(
    media_path: Path,
    scene_paths: ScenePaths,
    config: DetectionAndTrackingConfig,
    *,
    objects_json: Path,
    instances_json: Path,
    logger: logging.Logger,
) -> Path | None:
    """
    Plan + write per-track crops and a ``tracks.json`` seam.

    Returns:
        Path to the written ``tracks.json`` sidecar, or ``None`` when no crops
        were produced.
    """
    if not objects_json.is_file() or not instances_json.is_file():
        logger.warning("Crop extraction skipped: tracking artifacts missing.")
        return None

    plans = plan_track_crops(
        read_json(objects_json),
        read_json(instances_json),
        crop_classes=config.crop_classes,
        crops_per_track=config.crops_per_track,
        crop_padding=config.crop_padding,
        min_crop_size=config.min_crop_size,
        min_detection_score=config.min_detection_score,
        min_track_seconds=config.min_track_seconds,
    )
    if not plans:
        logger.info("Crop extraction produced no track plans for %s", media_path)
        return None

    cv2 = import_optional("cv2", backend="sam3")
    records = write_crops(
        Path(media_path),
        plans,
        output_root=scene_paths.sidecars_dir,
        crop_subdir=config.crop_subdir,
        crop_format=config.crop_format,
        cv2=cv2,
    )
    if not records:
        logger.info("Crop extraction wrote no crops for %s", media_path)
        return None

    tracks_path = resolve_sidecar(scene_paths.sidecars_dir, config.tracks_sidecar)
    write_json(
        tracks_path,
        {
            "crop_root": config.crop_subdir,
            "n_tracks": len(records),
            "tracks": [record.model_dump() for record in records],
        },
    )
    logger.info("Crop extraction wrote %d tracks to %s", len(records), tracks_path)
    return tracks_path


def _select_frames(
    per_frame: dict[int, tuple[list[float], int, int]],
    *,
    crops_per_track: int,
    crop_padding: float,
    min_crop_size: int,
) -> list[SelectedFrame]:
    selected: list[SelectedFrame] = []
    for frame_number in sample_frame_numbers(list(per_frame), crops_per_track):
        box, width, height = per_frame[frame_number]
        clamped = pad_and_clamp_box(box, width=width, height=height, padding=crop_padding)
        if clamped is None:
            continue
        if min(clamped[2] - clamped[0], clamped[3] - clamped[1]) < min_crop_size:
            continue
        selected.append(SelectedFrame(frame_number=frame_number, box=clamped))
    return selected


def _iter_media_frames(
    media_path: Path,
    frame_numbers: list[int],
    *,
    cv2: Any,
) -> Any:
    if is_image_path(media_path):
        bgr = cv2.imread(str(media_path))
        if bgr is None:
            raise RuntimeError(f"Could not read image: {media_path}")
        yield 0, bgr
        return

    # Service images disable OpenCV video backends; decode via core FFmpeg.
    np = import_optional("numpy", backend="sam3")
    yield from iter_selected_video_frames_bgr(
        media_path,
        frame_numbers,
        cv2=cv2,
        np=np,
    )


def _optional_score(value: Any) -> float | None:
    if value is None:
        return None
    return round(as_float(value, 0.0), 4)


def _duration(summary: dict[str, Any]) -> float | None:
    start = summary.get("start_time_s")
    end = summary.get("end_time_s")
    if start is None or end is None:
        return None
    return round(as_float(end, 0.0) - as_float(start, 0.0), 3)


__all__ = [
    "CropRecord",
    "SelectedFrame",
    "TrackCropPlan",
    "extract_track_crops",
    "pad_and_clamp_box",
    "plan_track_crops",
    "sample_frame_numbers",
    "write_crops",
]
