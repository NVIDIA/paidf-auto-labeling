# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DetectionAndTrackingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    model: Literal["rfdetr"] = "rfdetr"
    threshold: float = 0.2
    iou_threshold: float = 0.3
    classes: Any = Field(default_factory=lambda: ["car", "truck", "bus", "motorcycle", "bicycle"])

    @field_validator("classes", mode="before")
    @classmethod
    def _check_classes_nonempty(cls, v: Any) -> Any:
        if isinstance(v, (list, tuple)) and len(v) == 0:
            raise ValueError("detection_and_tracking.classes must not be empty")
        return v

    tracker: str = "boosttrack"
    # Whether to enable appearance-based ReID (BoostTrack only).
    # When false, pipeline will force-disable ReID even if weights exist.
    use_reid: bool = True
    # Optional explicit ReID weights path. If None/empty, pipeline may fall back to default.
    reid_weights: Optional[str] = None
    # Tracker behavior knobs (kept explicit; pipeline maps these to rfdetr_tracking.py argv).
    per_class: bool = True
    asso_func: str = "diou"
    min_hits: int = 3
    max_age: int = 60
    min_track_frames: int = 5
    deepocsort_stage2_off: bool = True
    deepocsort_min_hits_nonconsecutive: bool = True
    # Whether to copy the source input media into sidecars/.
    # Default to False to avoid duplicating the (potentially large) input media in outputs.
    # Set to True if you want a copy colocated with the tracking artifacts.
    copy_video: bool = False
    save_vis: bool = False
    # Whether to save overlay media (*_detection.<ext>, *_tracking.<ext>).
    # Enable by default; this is usually the primary artifact users want.
    save_video: bool = True
    # Whether to save the extra visualization-only red-id overlay media.
    # Enable by default (requested).
    save_video_red_id: bool = True
    # Whether to save extracted per-frame RGB JPEGs under <out_dir>/sidecars/rgb/.
    # Disable this to keep only mp4 videos + JSON outputs (much smaller output footprint).
    save_rgb: bool = False
    # Post-processing / output cleanup knobs.
    cross_class_iou_threshold: float = 0.9
    dedup_iou_threshold: float = 0.3
    dedup_priority: str = "prev_iou"
