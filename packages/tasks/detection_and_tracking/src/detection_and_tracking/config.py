# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration models for the detection-and-tracking task."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

TrackerKindValue = Literal[
    "stub",
    "rfdetr-bytetrack",
    "rfdetr-boosttrack",
    "rfdetr-deepocsort",
    "sam3",
]


class DetectionAndTrackingConfig(BaseModel):
    """Static configuration for ``DetectionAndTrackingTask``."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    tracker: TrackerKindValue = "rfdetr-boosttrack"
    model_cache_path: str | None = None
    gpu_ids: str | int | None = "all"
    classes: tuple[str, ...] = ()
    # RF-DETR / BoostTrack defaults match legacy pseudo-labeling tracking schema.
    threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    per_class: bool = True
    min_hits: int = Field(default=3, ge=1)
    max_age: int = Field(default=60, ge=0)
    min_track_frames: int = Field(default=5, ge=1)
    bbox_expansion_ratio: float = Field(
        default=0.1,
        ge=0.0,
        description=(
            "Fraction of box width/height used to expand RF-DETR tight boxes "
            "into DAFT bounding_box_2d_loose values. Matches legacy "
            "bbox_expansion_ratio."
        ),
    )
    save_video: bool = False
    save_red_id_overlay: bool = Field(
        default=False,
        description=(
            "When True, RF-DETR writes a red-id overlay. SAM3 treats this as "
            "a request to write its annotated id-label video."
        ),
    )
    save_rgb: bool = False
    copy_media: bool = False
    allow_model_download: bool = False
    sam3_prompts: tuple[str, ...] = ()
    sam3_version: Literal["sam3", "sam3.1"] = "sam3"
    sam3_runtime: Literal["auto", "transformers", "native"] = "auto"
    sam3_tracking_mode: Literal["chunked", "continuous"] = Field(
        default="chunked",
        description=(
            "chunked resets the SAM3 video session every sam3_session_reset_s; "
            "continuous keeps one session for the whole allowed clip."
        ),
    )
    sam3_target_fps: float = Field(default=10.0, gt=0.0)
    sam3_session_reset_s: float = Field(default=10.0, gt=0.0)
    sam3_max_duration_s: float = Field(
        default=30.0,
        gt=0.0,
        validation_alias=AliasChoices("sam3_max_duration_s", "sam3_max_clip_duration_s"),
    )
    sam3_write_annotated_video: bool = False
    sam3_annotated_video_trails: bool = False
    sam3_annotated_video_label_style: Literal["id", "name", "track", "none"] = "id"
    sam3_annotated_video_mask_opacity: int = Field(default=0, ge=0, le=100)
    sam3_annotated_video_shape: Literal["box", "contour"] = "contour"
    sam3_write_masks: bool = Field(
        default=False,
        description=(
            "When True, write per-detection mask PNGs under sidecars/sam3/masks/ "
            "for SoM-style inspection."
        ),
    )
    sam3_score_threshold_detection: float | None = Field(default=None, ge=0.0, le=1.0)
    sam3_det_nms_thresh: float | None = Field(default=None, ge=0.0, le=1.0)
    sam3_new_det_thresh: float | None = Field(default=None, ge=0.0, le=1.0)
    sam3_fill_hole_area: int | None = Field(default=None, ge=0)
    sam3_recondition_every_nth_frame: int | None = Field(default=None, ge=1)
    sam3_recondition_on_trk_masks: bool | None = None
    sam3_high_conf_thresh: float | None = Field(default=None, ge=0.0, le=1.0)
    sam3_high_iou_thresh: float | None = Field(default=None, ge=0.0, le=1.0)
    sam3_multiplex_count: int = Field(
        default=16,
        ge=1,
        description="Objects per Object Multiplex bucket (native SAM 3.1 only).",
    )
    sam3_max_num_objects: int = Field(
        default=16,
        ge=1,
        description="Maximum tracked objects for native SAM 3.1 multiplex.",
    )
    sam3_compile: bool = Field(
        default=False,
        description="Enable torch.compile for native SAM 3.1 when supported.",
    )

    # Per-track crop extraction (PAS video flow). Backend-agnostic: runs as a
    # post-step over the written ``objects.json`` so it works for any tracker.
    extract_crops: bool = False
    crop_classes: tuple[str, ...] = ()
    crops_per_track: int = Field(default=16, ge=1)
    crop_padding: float = Field(
        default=0.0,
        ge=0.0,
        description="Box expansion as a fraction of width/height before cropping.",
    )
    min_crop_size: int = Field(
        default=0,
        ge=0,
        description="Drop crops whose shorter side is below this many pixels.",
    )
    min_detection_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Drop tracks whose detection score is below this before cropping "
            "(0 disables). Mirrors legacy v1 min_score; prunes spurious tracks."
        ),
    )
    min_track_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Drop tracks shorter than this many seconds before cropping "
            "(0 disables). Mirrors legacy v1 min_track_seconds. Use only for "
            "full-video planning, not already-short pre-chunked clips."
        ),
    )
    crop_format: Literal["jpg", "png"] = "jpg"
    crop_subdir: str = "tracks/crops"
    tracks_sidecar: str = "detection_and_tracking/tracks.json"


def pas_tracking_config(**overrides: Any) -> DetectionAndTrackingConfig:
    """
    Build a Person-Attribute-Search tracking profile.

    Presets SAM3 person tracking with per-track crop extraction enabled, which is
    what the PAS video flow needs (person identities + sampled crops the
    Visual QA fan-out consumes). Any field can be overridden by keyword.

    Args:
        **overrides: Keyword overrides for any ``DetectionAndTrackingConfig`` field.

    Returns:
        A configured ``DetectionAndTrackingConfig`` for the PAS video flow.
    """
    defaults: dict[str, Any] = {
        "tracker": "sam3",
        "sam3_prompts": ("person",),
        "classes": ("person",),
        "min_track_frames": 4,
        "extract_crops": True,
        "crop_classes": ("person",),
        "crops_per_track": 16,
        "crop_padding": 0.1,
        "min_crop_size": 32,
    }
    defaults.update(overrides)
    return DetectionAndTrackingConfig.model_validate(defaults)


__all__ = ["DetectionAndTrackingConfig", "TrackerKindValue", "pas_tracking_config"]
