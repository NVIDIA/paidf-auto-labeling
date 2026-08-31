# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI service for the standalone detection-and-tracking container."""

from __future__ import annotations

import argparse
from typing import cast

from core import DataEntry
from core.interfaces import ServiceInterface
from core.pipelines import LinearPipeline
from core.policy import EmptyOutputPolicy
from core.utils.logging import apply_log_level
from daft_validation import DaftValidationTask
from detection_and_tracking.config import DetectionAndTrackingConfig, TrackerKindValue
from detection_and_tracking.factory import list_trackers
from detection_and_tracking.task import DetectionAndTrackingTask


class DetectionAndTrackingService(ServiceInterface):
    """Run only the detection-and-tracking task over supplied ``DataEntry`` records."""

    def __init__(self) -> None:
        super().__init__(
            name="detection_and_tracking_service",
            description="Run the standalone detection-and-tracking service.",
        )

    def add_service_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--tracker",
            default="rfdetr-boosttrack",
            help="Detection-and-tracking backend to run.",
        )
        parser.add_argument("--disabled", action="store_true", help="Skip tracking and exit.")
        parser.add_argument(
            "--model-cache-path",
            default=None,
            help=(
                "Root directory for model assets. RF-DETR expects rfdetr/rf-detr-base.pth; "
                "SAM3 expects a sam3/ directory unless backend-specific env vars override it."
            ),
        )
        parser.add_argument(
            "--gpu-ids",
            default="all",
            help='GPU ids for model runtime, e.g. "0" or "0,1". Use "all" for visible GPUs.',
        )
        parser.add_argument(
            "--classes",
            nargs="*",
            default=[],
            help=(
                "Class names to keep for RF-DETR. SAM3 uses these as text prompts when "
                "--sam3-prompts is omitted."
            ),
        )
        parser.add_argument(
            "--threshold",
            type=float,
            default=0.2,
            help=(
                "RF-DETR detection confidence threshold. "
                "Default 0.2 matches legacy pseudo-labeling."
            ),
        )
        parser.add_argument(
            "--iou-threshold",
            type=float,
            default=0.3,
            help="IoU association threshold for tracker matching where the backend supports it.",
        )
        parser.add_argument(
            "--per-class",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=(
                "Run tracker association independently per detected class. "
                "Default on matches legacy pseudo-labeling; disable with --no-per-class."
            ),
        )
        parser.add_argument(
            "--min-hits",
            type=_int_ge_one,
            default=3,
            help="Minimum consecutive hits before a BoostTrack track is emitted.",
        )
        parser.add_argument(
            "--max-age",
            type=_int_ge_zero,
            default=60,
            help="Maximum missed frames before a BoostTrack track is removed.",
        )
        parser.add_argument(
            "--min-track-frames",
            type=int,
            default=5,
            help=(
                "Drop RF-DETR tracks shorter than this many frames. "
                "Default 5 matches legacy pseudo-labeling."
            ),
        )
        parser.add_argument(
            "--bbox-expansion-ratio",
            type=float,
            default=0.1,
            help=(
                "Expand RF-DETR tight boxes by this fraction when writing "
                "bounding_box_2d_loose. Default 0.1 matches legacy."
            ),
        )
        parser.add_argument(
            "--save-video",
            action="store_true",
            help="Write detection/tracking overlay videos under sidecars/.",
        )
        parser.add_argument(
            "--save-red-id-overlay",
            action="store_true",
            help="Write an RF-DETR red-id overlay; SAM3 treats this as an annotated video request.",
        )
        parser.add_argument(
            "--save-rgb",
            action="store_true",
            help="Write sampled RGB frames for RF-DETR diagnostics.",
        )
        parser.add_argument(
            "--copy-media",
            action="store_true",
            help="Copy input media into raw/ instead of creating a symlink.",
        )
        parser.add_argument(
            "--allow-model-download",
            action="store_true",
            help="Allow RF-DETR to download its checkpoint when it is missing locally.",
        )
        parser.add_argument(
            "--sam3-prompts",
            nargs="*",
            default=[],
            help="Text prompts to track with SAM3, e.g. person vehicle.",
        )
        parser.add_argument(
            "--sam3-version",
            choices=["sam3", "sam3.1"],
            default="sam3",
            help=(
                "SAM3 model family. sam3 uses the Transformers runtime by default; "
                "sam3.1 requires the native Meta package (Object Multiplex)."
            ),
        )
        parser.add_argument(
            "--sam3-runtime",
            choices=["auto", "transformers", "native"],
            default="auto",
            help=(
                "SAM3 inference stack. auto selects transformers for sam3 and native for sam3.1."
            ),
        )
        parser.add_argument(
            "--sam3-tracking-mode",
            choices=["chunked", "continuous"],
            default="chunked",
            help=(
                "chunked resets the SAM3 session every --sam3-session-reset-s; "
                "continuous keeps one session for the whole allowed clip."
            ),
        )
        parser.add_argument(
            "--sam3-target-fps",
            type=float,
            default=10.0,
            help="Frame rate used when sampling video for SAM3.",
        )
        parser.add_argument(
            "--sam3-session-reset-s",
            type=float,
            default=10.0,
            help="SAM3 chunk duration before resetting the video session.",
        )
        parser.add_argument(
            "--sam3-max-duration-s",
            "--sam3-max-clip-duration-s",
            dest="sam3_max_duration_s",
            type=float,
            default=30.0,
            help="Maximum SAM3 clip duration to process per input.",
        )
        parser.add_argument(
            "--sam3-write-annotated-video",
            action="store_true",
            help="Write a SAM3 mask-contour annotated video under sidecars/sam3/.",
        )
        parser.add_argument(
            "--sam3-annotated-video-trails",
            action="store_true",
            help="Draw object trails in the SAM3 annotated video.",
        )
        parser.add_argument(
            "--sam3-annotated-video-label-style",
            choices=["id", "name", "track", "none"],
            default="id",
            help=(
                "Label style for SAM3 annotated video overlays: chunk id (#0:3), "
                "prompt name, stable track label (person_T001), or none."
            ),
        )
        parser.add_argument(
            "--sam3-annotated-video-mask-opacity",
            type=int,
            default=0,
            help="Mask fill opacity for SAM3 annotated videos, from 0 to 100.",
        )
        parser.add_argument(
            "--sam3-annotated-video-shape",
            choices=["box", "contour"],
            default="contour",
            help="Draw a tight box or the mask contour outline in the SAM3 annotated video.",
        )
        parser.add_argument(
            "--sam3-write-masks",
            action="store_true",
            help="Write per-detection mask PNGs under sidecars/sam3/masks/.",
        )
        parser.add_argument(
            "--sam3-score-threshold-detection",
            type=_float_0_to_1,
            default=None,
            help="Override SAM3 detection score threshold.",
        )
        parser.add_argument(
            "--sam3-det-nms-thresh",
            type=_float_0_to_1,
            default=None,
            help="Override SAM3 detection NMS threshold.",
        )
        parser.add_argument(
            "--sam3-new-det-thresh",
            type=_float_0_to_1,
            default=None,
            help="Override SAM3 new-object detection threshold.",
        )
        parser.add_argument(
            "--sam3-fill-hole-area",
            type=_int_ge_zero,
            default=None,
            help="Override SAM3 mask hole filling area.",
        )
        parser.add_argument(
            "--sam3-recondition-every-nth-frame",
            type=_int_ge_one,
            default=None,
            help="Override how often SAM3 reconditions tracking state.",
        )
        parser.add_argument(
            "--sam3-recondition-on-trk-masks",
            nargs="?",
            const=True,
            default=None,
            type=_optional_bool,
            help="Override whether SAM3 reconditions from tracked masks.",
        )
        parser.add_argument(
            "--sam3-high-conf-thresh",
            type=_float_0_to_1,
            default=None,
            help="Override SAM3 high-confidence threshold.",
        )
        parser.add_argument(
            "--sam3-high-iou-thresh",
            type=_float_0_to_1,
            default=None,
            help="Override SAM3 high-IoU threshold.",
        )
        parser.add_argument(
            "--sam3-multiplex-count",
            type=_int_ge_one,
            default=16,
            help="Objects per Object Multiplex bucket (native SAM 3.1 only).",
        )
        parser.add_argument(
            "--sam3-max-num-objects",
            type=_int_ge_one,
            default=16,
            help="Maximum tracked objects for native SAM 3.1 multiplex.",
        )
        parser.add_argument(
            "--sam3-compile",
            action="store_true",
            help="Enable torch.compile for native SAM 3.1 when supported.",
        )
        self._add_crop_args(parser)

    def _add_crop_args(self, parser: argparse.ArgumentParser) -> None:
        """Per-track crop extraction flags (PAS video flow)."""
        parser.add_argument(
            "--extract-crops",
            action="store_true",
            help=(
                "Write per-track crops and a tracks.json seam over the tracker's "
                "objects.json (backend-agnostic). Required by the PAS video flow."
            ),
        )
        parser.add_argument(
            "--crop-classes",
            nargs="*",
            default=[],
            help="Class names to crop. Empty keeps every tracked class.",
        )
        parser.add_argument(
            "--crops-per-track",
            type=_int_ge_one,
            default=16,
            help="Maximum crops sampled per track.",
        )
        parser.add_argument(
            "--crop-padding",
            type=_float_ge_zero,
            default=0.0,
            help="Box expansion as a fraction of width/height before cropping.",
        )
        parser.add_argument(
            "--min-crop-size",
            type=_int_ge_zero,
            default=0,
            help="Drop crops whose shorter side is below this many pixels.",
        )
        parser.add_argument(
            "--min-detection-score",
            type=_float_0_to_1,
            default=0.0,
            help=(
                "Drop tracks whose detection score is below this before cropping "
                "(0.0-1.0; 0 disables). Mirrors legacy v1 min_score."
            ),
        )
        parser.add_argument(
            "--min-track-seconds",
            type=_float_ge_zero,
            default=0.0,
            help=(
                "Drop tracks shorter than this many seconds before cropping "
                "(0 disables). Mirrors legacy v1 min_track_seconds; use only for "
                "full-video planning, not already-short pre-chunked clips."
            ),
        )
        parser.add_argument(
            "--crop-format",
            choices=["jpg", "png"],
            default="jpg",
            help="Image format for written crops.",
        )
        parser.add_argument(
            "--crop-subdir",
            default="tracks/crops",
            help="Sidecar subdirectory for per-track crops.",
        )
        parser.add_argument(
            "--tracks-sidecar",
            default="detection_and_tracking/tracks.json",
            help="Sidecar-relative path for the tracks.json seam.",
        )

    def run(self) -> None:
        parser = argparse.ArgumentParser(description=self.description)
        self._add_common_args(parser)
        self.add_service_args(parser)
        args = parser.parse_args()
        if not bool(getattr(args, "disabled", False)):
            args.tracker = _validate_tracker_arg(parser, args.tracker)
        apply_log_level(level=args.log_level)
        self.logger.info(f"Starting Service {self.name} with log level {args.log_level}.")
        if bool(getattr(args, "disabled", False)):
            self.logger.info("Detection-and-tracking disabled; skipping input loading and run.")
            return
        data_entries = self._get_data_entries(args)
        if args.dev_data_root is not None:
            data_entries = self._copy_data_entries_to_dev_root(
                data_entries,
                args.dev_data_root,
            )
        self.execute(args, data_entries)

    def execute(self, args: argparse.Namespace, data_entries: list[DataEntry]) -> None:
        if bool(getattr(args, "disabled", False)):
            self.logger.info("Detection-and-tracking disabled; skipping task creation and run.")
            return

        if not data_entries:
            raise SystemExit("Pass --input or --input-file with at least one DataEntry.")

        pipeline = LinearPipeline(
            tasks=[DetectionAndTrackingTask(config=build_config(args)), DaftValidationTask()],
            name="detection_and_tracking_pipeline",
            policy=EmptyOutputPolicy.FAIL,
        )
        processed = pipeline.run(data_entries)
        self.logger.info("Processed %d data entries.", len(processed))


def build_config(args: argparse.Namespace) -> DetectionAndTrackingConfig:
    """Build task config from parsed CLI args."""
    return DetectionAndTrackingConfig(
        enabled=not bool(args.disabled),
        tracker=cast(TrackerKindValue, args.tracker),
        model_cache_path=args.model_cache_path,
        gpu_ids=args.gpu_ids,
        classes=tuple(args.classes),
        threshold=args.threshold,
        iou_threshold=args.iou_threshold,
        per_class=args.per_class,
        min_hits=args.min_hits,
        max_age=args.max_age,
        min_track_frames=args.min_track_frames,
        bbox_expansion_ratio=args.bbox_expansion_ratio,
        save_video=args.save_video,
        save_red_id_overlay=args.save_red_id_overlay,
        save_rgb=args.save_rgb,
        copy_media=args.copy_media,
        allow_model_download=args.allow_model_download,
        sam3_prompts=tuple(args.sam3_prompts),
        sam3_version=args.sam3_version,
        sam3_runtime=args.sam3_runtime,
        sam3_tracking_mode=args.sam3_tracking_mode,
        sam3_target_fps=args.sam3_target_fps,
        sam3_session_reset_s=args.sam3_session_reset_s,
        sam3_max_duration_s=args.sam3_max_duration_s,
        sam3_write_annotated_video=args.sam3_write_annotated_video,
        sam3_annotated_video_trails=args.sam3_annotated_video_trails,
        sam3_annotated_video_label_style=args.sam3_annotated_video_label_style,
        sam3_annotated_video_mask_opacity=args.sam3_annotated_video_mask_opacity,
        sam3_annotated_video_shape=args.sam3_annotated_video_shape,
        sam3_write_masks=args.sam3_write_masks,
        sam3_score_threshold_detection=args.sam3_score_threshold_detection,
        sam3_det_nms_thresh=args.sam3_det_nms_thresh,
        sam3_new_det_thresh=args.sam3_new_det_thresh,
        sam3_fill_hole_area=args.sam3_fill_hole_area,
        sam3_recondition_every_nth_frame=args.sam3_recondition_every_nth_frame,
        sam3_recondition_on_trk_masks=args.sam3_recondition_on_trk_masks,
        sam3_high_conf_thresh=args.sam3_high_conf_thresh,
        sam3_high_iou_thresh=args.sam3_high_iou_thresh,
        sam3_multiplex_count=args.sam3_multiplex_count,
        sam3_max_num_objects=args.sam3_max_num_objects,
        sam3_compile=args.sam3_compile,
        extract_crops=args.extract_crops,
        crop_classes=tuple(args.crop_classes),
        crops_per_track=args.crops_per_track,
        crop_padding=args.crop_padding,
        min_crop_size=args.min_crop_size,
        min_detection_score=args.min_detection_score,
        min_track_seconds=args.min_track_seconds,
        crop_format=args.crop_format,
        crop_subdir=args.crop_subdir,
        tracks_sidecar=args.tracks_sidecar,
    )


def _validate_tracker_arg(parser: argparse.ArgumentParser, tracker: object) -> TrackerKindValue:
    tracker_name = str(tracker)
    try:
        trackers = list_trackers()
    except Exception as exc:
        parser.error(f"Could not list detection tracker backends: {exc}")
    if tracker_name not in trackers:
        choices = ", ".join(trackers) or "<none>"
        parser.error(f"Invalid tracker: {tracker_name}. Available trackers: {choices}")
    return cast(TrackerKindValue, tracker_name)


def _float_0_to_1(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a float between 0.0 and 1.0") from exc
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0.0 and 1.0")
    return parsed


def _float_ge_zero(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a float >= 0.0") from exc
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be >= 0.0")
    return parsed


def _int_ge_zero(value: str) -> int:
    return _int_at_least(value, minimum=0)


def _int_ge_one(value: str) -> int:
    return _int_at_least(value, minimum=1)


def _int_at_least(value: str, *, minimum: int) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"must be an integer >= {minimum}") from exc
    if parsed < minimum:
        raise argparse.ArgumentTypeError(f"must be >= {minimum}")
    return parsed


def _optional_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("must be one of: true, false, 1, 0, yes, no")


def main() -> None:
    """Run the detection-and-tracking service CLI."""
    DetectionAndTrackingService().run()


if __name__ == "__main__":
    main()
