# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RF-DETR detector with ByteTrack or BoostTrack association."""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Literal, cast

from core import SceneContext, ScenePaths, is_image_path

from detection_and_tracking.backends.runtime import (
    first_configured_gpu,
    import_optional,
    list_from_runtime,
    media_output_suffix,
    resolve_model_cache_root,
    write_tracking_artifacts,
)
from detection_and_tracking.config import DetectionAndTrackingConfig
from detection_and_tracking.media import Vp9VideoWriter, decode_video_bgr
from detection_and_tracking.tracker import Tracker, TrackingResult

RFDETR_BASE_URL = "https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth"
RFDETR_BASE_FILENAME = "rf-detr-base.pth"
RFDETR_BASE_SHA256 = "d8f70210e425a4a4234d547737f57500bcc4ac24a333b99e33d9d5a371e0b80f"
RFDETR_SHA256_ENV = "RFDETR_MODEL_SHA256"
_HASH_CHUNK_SIZE = 1024 * 1024


class RFDetrTracker(Tracker):
    """RF-DETR detector plus a stateful tracker recreated per media sample."""

    def __init__(
        self,
        logger: logging.Logger,
        config: DetectionAndTrackingConfig,
        *,
        tracker_backend: Literal["bytetrack", "boosttrack"],
    ) -> None:
        super().__init__(logger)
        self.config = config
        self.tracker_backend = tracker_backend
        self._runtime = _RFDetrRuntime(config=config, logger=logger)

    def run(
        self,
        media_path: Path,
        scene_paths: ScenePaths,
        *,
        scene_ctx: SceneContext,
    ) -> TrackingResult:
        result = self._runtime.run(
            media_path=Path(media_path),
            scene_paths=scene_paths,
            scene_ctx=scene_ctx,
            tracker_backend=self.tracker_backend,
        )
        return TrackingResult(
            success=True,
            objects_json=scene_paths.contextual_dir / "objects.json",
            instances_json=scene_paths.contextual_dir / "instances.json",
            detection_overlay=result.detection_overlay_path,
            tracking_overlay=result.tracking_overlay_path,
            tracking_video_red_id=result.red_id_overlay_path,
        )


class _RFDetrRunResult:
    def __init__(
        self,
        *,
        detection_overlay_path: Path | None,
        tracking_overlay_path: Path | None,
        red_id_overlay_path: Path | None,
    ) -> None:
        self.detection_overlay_path = detection_overlay_path
        self.tracking_overlay_path = tracking_overlay_path
        self.red_id_overlay_path = red_id_overlay_path


class _RFDetrRuntime:
    def __init__(self, *, config: DetectionAndTrackingConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.cv2 = import_optional("cv2", backend="rfdetr")
        self.np = import_optional("numpy", backend="rfdetr")
        self.image_module = import_optional("PIL.Image", backend="rfdetr")
        self.torch = import_optional("torch", backend="rfdetr")
        rfdetr_module = import_optional("rfdetr", backend="rfdetr")
        coco_module = import_optional("rfdetr.util.coco_classes", backend="rfdetr")
        self._model_cls = rfdetr_module.RFDETRBase
        self._coco_classes = _normalize_coco_classes(coco_module.COCO_CLASSES)
        self._class_filter_ids = _resolve_class_filter_ids(
            self._coco_classes,
            self.config.classes,
        )
        self._model = self._load_model()

    def run(
        self,
        *,
        media_path: Path,
        scene_paths: ScenePaths,
        scene_ctx: SceneContext,
        tracker_backend: Literal["bytetrack", "boosttrack"],
    ) -> _RFDetrRunResult:
        tracker = self._new_tracker(tracker_backend)
        frame_iter = self._iter_frames(media_path)
        detections_out: list[dict[str, Any]] = []
        frames_out: list[dict[str, Any]] = []
        instances: dict[str, dict[str, Any]] = {}
        rgb_dir = scene_paths.sidecars_dir / "rfdetr" / "rgb" if self.config.save_rgb else None
        if rgb_dir is not None:
            rgb_dir.mkdir(parents=True, exist_ok=True)
        frame_count = 0

        for frame_number, frame_bgr, fps, width, height in frame_iter:
            if rgb_dir is not None:
                self._write_rgb_frame(rgb_dir, frame_number, frame_bgr)
            pil_img = self.image_module.fromarray(
                self.cv2.cvtColor(frame_bgr, self.cv2.COLOR_BGR2RGB)
            )
            predictions = self._model.predict(pil_img, threshold=float(self.config.threshold))
            detections = self._predictions_to_array(predictions)
            detections_out.append(
                {
                    "frame_number": frame_number,
                    "detections": self._detections_for_frame(detections),
                }
            )
            tracks = tracker.update(detections, frame_bgr)
            frame_instances: list[dict[str, Any]] = []

            for track in tracks:
                track_id = int(track[4])
                confidence = float(track[5]) if len(track) > 5 else 1.0
                class_id = int(track[6]) if len(track) > 6 else 0
                class_name = self._class_name(class_id)
                object_id = f"{class_name}_{track_id}"
                bbox = [round(float(v), 2) for v in track[:4]]
                loose_bbox = self._loose_bbox(
                    bbox,
                    width=width,
                    height=height,
                    expansion_ratio=float(self.config.bbox_expansion_ratio),
                )
                instance = instances.setdefault(
                    object_id,
                    {
                        "object_id": object_id,
                        "object_type": class_name,
                        "semantic_id": class_id,
                        "track_id": track_id,
                        "first_frame": frame_number,
                        "last_frame": frame_number,
                        "frame_count": 0,
                        "confidence_avg": confidence,
                    },
                )
                instance["last_frame"] = frame_number
                instance["frame_count"] = int(instance["frame_count"]) + 1
                old_avg = float(instance["confidence_avg"])
                n = int(instance["frame_count"])
                instance["confidence_avg"] = old_avg + (confidence - old_avg) / n
                frame_instances.append(
                    {
                        "object_id": object_id,
                        "instance_id": track_id,
                        "semantic_id": class_id,
                        "bounding_box_2d_tight": bbox,
                        "bounding_box_2d_loose": loose_bbox,
                        "confidence": round(confidence, 4),
                    }
                )

            frames_out.append(
                {
                    "frame_number": frame_number,
                    "width": width,
                    "height": height,
                    "instances": frame_instances,
                }
            )
            frame_count += 1
            _ = fps

        instances_out = [
            inst
            for inst in instances.values()
            if int(inst.get("frame_count", 0)) >= self.config.min_track_frames
        ]
        kept_object_ids = {str(inst["object_id"]) for inst in instances_out}
        frames_out = self._filter_frames_by_object_ids(frames_out, kept_object_ids)
        write_tracking_artifacts(
            scene_paths=scene_paths,
            scene_ctx=scene_ctx,
            frames=frames_out,
            instances=instances_out,
        )
        detection_overlay = (
            self._write_detection_overlay(media_path, scene_paths, scene_ctx, detections_out)
            if self.config.save_video
            else None
        )
        tracking_overlay = (
            self._write_tracking_overlay(media_path, scene_paths, scene_ctx, frames_out)
            if self.config.save_video
            else None
        )
        red_id_path = (
            self._write_tracking_overlay(
                media_path,
                scene_paths,
                scene_ctx,
                frames_out,
                red_id=True,
            )
            if self.config.save_red_id_overlay
            else None
        )
        self.logger.info(
            "RF-DETR %s processed %s: %d frames, %d tracks",
            self.config.tracker,
            media_path,
            frame_count,
            len(instances_out),
        )
        return _RFDetrRunResult(
            detection_overlay_path=detection_overlay,
            tracking_overlay_path=tracking_overlay,
            red_id_overlay_path=red_id_path,
        )

    def _load_model(self) -> Any:
        checkpoint = _resolve_rfdetr_checkpoint(self.config, logger=self.logger)
        gpu_id = first_configured_gpu(self.config.gpu_ids)
        if bool(self.torch.cuda.is_available()):
            if gpu_id is not None:
                self.torch.cuda.set_device(gpu_id)
            device = "cuda"
        else:
            device = "cpu"
        self.logger.info("Loading RF-DETR from %s on %s", checkpoint, device)
        return self._model_cls(device=device, pretrain_weights=str(checkpoint))

    def _new_tracker(self, backend: Literal["bytetrack", "boosttrack"]) -> Any:
        if backend == "bytetrack":
            return _ByteTrackAdapter(config=self.config)
        return _BoostTrackAdapter(config=self.config)

    def _iter_frames(self, media_path: Path) -> Any:
        if is_image_path(media_path):
            frame = self.cv2.imread(str(media_path))
            if frame is None:
                raise RuntimeError(f"Could not read image: {media_path}")
            height, width = frame.shape[:2]
            yield 0, frame, 1.0, width, height
            return

        decoded = decode_video_bgr(media_path, cv2=self.cv2, np=self.np)
        fps = decoded.stream.fps or 30.0
        for frame_number, frame in enumerate(decoded.frames):
            yield frame_number, frame, fps, decoded.stream.width, decoded.stream.height

    def _predictions_to_array(self, predictions: Any) -> Any:
        xyxy = predictions.xyxy
        confidence = predictions.confidence
        class_id = predictions.class_id.astype(int)
        if self._class_filter_ids is not None:
            mask = self.np.isin(class_id, self._class_filter_ids)
            xyxy = xyxy[mask]
            confidence = confidence[mask]
            class_id = class_id[mask]
        if len(xyxy) == 0:
            return self.np.empty((0, 6), dtype=float)
        return self.np.column_stack((xyxy, confidence, class_id))

    def _class_name(self, class_id: int) -> str:
        if 0 <= class_id < len(self._coco_classes):
            return str(self._coco_classes[class_id])
        return f"class_{class_id}"

    def _write_rgb_frame(self, rgb_dir: Path, frame_number: int, frame_bgr: Any) -> None:
        output_path = rgb_dir / f"{frame_number:06d}.jpg"
        ok = self.cv2.imwrite(str(output_path), frame_bgr)
        if not ok:
            raise RuntimeError(f"Failed to write RGB sidecar frame: {output_path}")

    def _detections_for_frame(self, detections: Any) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for det in detections:
            class_id = int(det[5])
            out.append(
                {
                    "semantic_id": class_id,
                    "class_name": self._class_name(class_id),
                    "confidence": round(float(det[4]), 4),
                    "bounding_box_2d_tight": [round(float(v), 2) for v in det[:4]],
                }
            )
        return out

    @staticmethod
    def _loose_bbox(
        bbox: list[float],
        *,
        width: int,
        height: int,
        expansion_ratio: float = 0.1,
    ) -> list[float]:
        """Return the legacy-style expanded box, clipped to image bounds."""
        x1, y1, x2, y2 = bbox
        x_pad = max(0.0, (x2 - x1) * float(expansion_ratio))
        y_pad = max(0.0, (y2 - y1) * float(expansion_ratio))
        loose = [
            max(0.0, x1 - x_pad),
            max(0.0, y1 - y_pad),
            min(float(width), x2 + x_pad),
            min(float(height), y2 + y_pad),
        ]
        return [round(float(v), 2) for v in loose]

    @staticmethod
    def _filter_frames_by_object_ids(
        frames: list[dict[str, Any]], kept_object_ids: set[str]
    ) -> list[dict[str, Any]]:
        filtered: list[dict[str, Any]] = []
        for frame in frames:
            out = dict(frame)
            out["instances"] = [
                inst for inst in frame["instances"] if str(inst["object_id"]) in kept_object_ids
            ]
            filtered.append(out)
        return filtered

    def _draw_detections(self, frame_bgr: Any, detections: list[dict[str, Any]]) -> Any:
        out = frame_bgr.copy()
        for detection in detections:
            bbox = [int(round(float(v))) for v in detection["bounding_box_2d_tight"]]
            label = f"{detection['class_name']} {float(detection['confidence']):.2f}"
            self.cv2.rectangle(out, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 180, 0), 2)
            self.cv2.putText(
                out,
                label,
                (bbox[0] + 2, max(12, bbox[1] + 14)),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 180, 0),
                1,
                self.cv2.LINE_AA,
            )
        return out

    def _draw_tracks(
        self,
        frame_bgr: Any,
        instances: list[dict[str, Any]],
        *,
        red_id: bool = False,
    ) -> Any:
        out = frame_bgr.copy()
        color = (0, 0, 255) if red_id else (255, 120, 0)
        for instance in instances:
            bbox = [int(round(float(v))) for v in instance["bounding_box_2d_tight"]]
            if red_id:
                label = str(instance["instance_id"])
                thickness = 1
                font_scale = 0.6
            else:
                label = f"{instance['object_id']} {float(instance['confidence']):.2f}"
                thickness = 2
                font_scale = 0.5
            self.cv2.rectangle(out, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
            self.cv2.putText(
                out,
                label,
                (bbox[0] + 2, max(12, bbox[1] + 14)),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
                self.cv2.LINE_AA,
            )
        return out

    def _write_detection_overlay(
        self,
        media_path: Path,
        scene_paths: ScenePaths,
        scene_ctx: SceneContext,
        detections_out: list[dict[str, Any]],
    ) -> Path:
        detections_by_frame = {
            int(frame["frame_number"]): frame["detections"] for frame in detections_out
        }
        output_path = (
            scene_paths.sidecars_dir
            / "rfdetr"
            / (f"{scene_ctx.media_id}_detection.{media_output_suffix(media_path)}")
        )
        return self._write_media_overlay(
            media_path,
            output_path,
            lambda frame_number, frame: self._draw_detections(
                frame, detections_by_frame.get(frame_number, [])
            ),
        )

    def _write_tracking_overlay(
        self,
        media_path: Path,
        scene_paths: ScenePaths,
        scene_ctx: SceneContext,
        frames_out: list[dict[str, Any]],
        *,
        red_id: bool = False,
    ) -> Path:
        instances_by_frame = {
            int(frame["frame_number"]): frame["instances"] for frame in frames_out
        }
        kind = "tracking_red_id" if red_id else "tracking"
        output_path = (
            scene_paths.sidecars_dir
            / "rfdetr"
            / (f"{scene_ctx.media_id}_{kind}.{media_output_suffix(media_path)}")
        )
        return self._write_media_overlay(
            media_path,
            output_path,
            lambda frame_number, frame: self._draw_tracks(
                frame, instances_by_frame.get(frame_number, []), red_id=red_id
            ),
        )

    def _write_media_overlay(self, media_path: Path, output_path: Path, draw_frame: Any) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if is_image_path(media_path):
            frame = self.cv2.imread(str(media_path))
            if frame is None:
                raise RuntimeError(f"Could not read image for overlay: {media_path}")
            ok = self.cv2.imwrite(str(output_path), draw_frame(0, frame))
            if not ok:
                raise RuntimeError(f"Failed to write overlay image: {output_path}")
            return output_path

        decoded = decode_video_bgr(media_path, cv2=self.cv2, np=self.np)
        with Vp9VideoWriter(
            output_path,
            width=decoded.stream.width,
            height=decoded.stream.height,
            fps=decoded.stream.fps or 30.0,
        ) as writer:
            for frame_number, frame in enumerate(decoded.frames):
                writer.write(draw_frame(frame_number, frame))
        return output_path

    @staticmethod
    def _normalize_class(name: str) -> str:
        return str(name).strip().lower().replace(" ", "_")


def _normalize_coco_classes(coco_classes: Any) -> list[str]:
    if isinstance(coco_classes, dict):
        indexed_classes: list[tuple[int, str]] = []
        for key, value in coco_classes.items():
            try:
                class_id = int(key)
            except (TypeError, ValueError):
                continue
            if class_id >= 0:
                indexed_classes.append((class_id, str(value)))
        if indexed_classes:
            classes = [f"class_{idx}" for idx in range(max(idx for idx, _ in indexed_classes) + 1)]
            for class_id, name in indexed_classes:
                classes[class_id] = name
            return classes
        return [str(value) for value in coco_classes.values()]
    return [str(name) for name in list_from_runtime(coco_classes)]


def _resolve_class_filter_ids(
    coco_classes: list[str],
    class_names: tuple[str, ...],
) -> list[int] | None:
    if not class_names:
        return None
    name_to_id = {_normalize_class(name): idx for idx, name in enumerate(coco_classes)}
    resolved: list[int] = []
    unknown: list[str] = []
    for class_name in class_names:
        key = _normalize_class(class_name)
        if key not in name_to_id:
            unknown.append(class_name)
            continue
        resolved.append(name_to_id[key])
    if unknown:
        examples = ", ".join(coco_classes[:10])
        raise ValueError(
            "Unknown RF-DETR class name(s): "
            + ", ".join(unknown)
            + f". Valid examples: {examples} ..."
        )
    return sorted(set(resolved))


def _normalize_class(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


class _ByteTrackAdapter:
    def __init__(self, *, config: DetectionAndTrackingConfig) -> None:
        sv = import_optional("supervision", backend="rfdetr-bytetrack")
        self.np = import_optional("numpy", backend="rfdetr-bytetrack")
        self.per_class = bool(config.per_class)
        self._tracker_cls = sv.ByteTrack
        self._tracker = self._new_tracker()
        self._trackers_by_class: dict[int, Any] = {}
        self._detections_cls = sv.Detections
        self._ = config

    def _new_tracker(self) -> Any:
        return self._tracker_cls()

    def _empty_detections(self) -> Any:
        return self._detections_cls(
            xyxy=self.np.empty((0, 4), dtype=float),
            confidence=self.np.empty((0,), dtype=float),
            class_id=self.np.empty((0,), dtype=int),
        )

    def _to_sv_detections(self, detections: Any) -> Any:
        if detections.size == 0:
            return self._empty_detections()
        return self._detections_cls(
            xyxy=detections[:, 0:4].astype(float, copy=False),
            confidence=detections[:, 4].astype(float, copy=False),
            class_id=detections[:, 5].astype(int, copy=False),
        )

    def _sv_to_tracks(self, tracked: Any) -> Any:
        tracker_id = getattr(tracked, "tracker_id", None)
        if tracker_id is None or len(tracked.xyxy) == 0:
            return self.np.empty((0, 7), dtype=float)
        confidence = (
            tracked.confidence
            if tracked.confidence is not None
            else self.np.ones((len(tracked.xyxy),), dtype=float)
        )
        class_id = (
            tracked.class_id
            if tracked.class_id is not None
            else self.np.zeros((len(tracked.xyxy),), dtype=int)
        )
        return self.np.column_stack([tracked.xyxy, tracker_id, confidence, class_id]).astype(float)

    def _update_tracker(self, tracker: Any, detections: Any) -> Any:
        sv_dets = self._to_sv_detections(detections)
        tracked = tracker.update_with_detections(sv_dets)
        return self._sv_to_tracks(tracked)

    def update(self, detections: Any, frame_bgr: Any) -> Any:
        _ = frame_bgr
        if not self.per_class:
            return self._update_tracker(self._tracker, detections)

        out_tracks: list[Any] = []
        present_classes = (
            {int(c) for c in self.np.unique(detections[:, 5].astype(int))}
            if detections.size != 0
            else set()
        )
        for class_id, tracker in list(self._trackers_by_class.items()):
            if class_id not in present_classes:
                tracker.update_with_detections(self._empty_detections())
        for class_id in sorted(present_classes):
            mask = detections[:, 5].astype(int) == int(class_id)
            tracker = self._trackers_by_class.get(class_id)
            if tracker is None:
                tracker = self._new_tracker()
                self._trackers_by_class[class_id] = tracker
            tracks = self._update_tracker(tracker, detections[mask])
            if tracks.size:
                out_tracks.append(tracks)
        if not out_tracks:
            return self.np.empty((0, 7), dtype=float)
        return self.np.concatenate(out_tracks, axis=0)


class _BoostTrackAdapter:
    def __init__(self, *, config: DetectionAndTrackingConfig) -> None:
        self.np = import_optional("numpy", backend="rfdetr-boosttrack")
        module = import_optional(
            "detection_and_tracking.backends.boosttrack.boost_track",
            backend="rfdetr-boosttrack",
        )
        self.per_class = bool(config.per_class)
        self._tracker_cls = module.BoostTrack
        self._tracker_kwargs = {
            "max_age": int(config.max_age),
            "min_hits": int(config.min_hits),
            "det_thresh": float(config.threshold),
            "iou_threshold": float(config.iou_threshold),
        }
        self._tracker = self._new_tracker()
        self._trackers_by_class: dict[int, Any] = {}

    def _new_tracker(self) -> Any:
        return self._tracker_cls(**self._tracker_kwargs)

    def _update_tracker(self, tracker: Any, detections: Any, frame_bgr: Any) -> Any:
        if detections.size == 0:
            tracker.update(self.np.empty((0, 5), dtype=float), frame_bgr=frame_bgr)
            return self.np.empty((0, 7), dtype=float)
        dets_xyxy_conf = self.np.column_stack([detections[:, 0:4], detections[:, 4]])
        tracks = tracker.update(dets_xyxy_conf, frame_bgr=frame_bgr)
        return self._postprocess_tracks(tracks, detections)

    def _postprocess_tracks(self, tracks: Any, detections: Any) -> Any:
        if tracks.size == 0:
            return self.np.empty((0, 7), dtype=float)
        boxes = tracks[:, 0:4].astype(float, copy=False)
        tids = tracks[:, 4].astype(int, copy=False)
        classes = []
        confidences = []
        for box in boxes:
            idx = _best_iou_index(self.np, box, detections[:, 0:4])
            classes.append(int(detections[idx, 5]))
            confidences.append(float(detections[idx, 4]))
        return self.np.column_stack([boxes, tids, confidences, classes]).astype(float)

    def update(self, detections: Any, frame_bgr: Any) -> Any:
        if not getattr(self, "per_class", False):
            return self._update_tracker(self._tracker, detections, frame_bgr)

        out_tracks: list[Any] = []
        present_classes = (
            {int(c) for c in self.np.unique(detections[:, 5].astype(int))}
            if detections.size != 0
            else set()
        )
        for class_id, tracker in list(self._trackers_by_class.items()):
            if class_id not in present_classes:
                tracker.update(self.np.empty((0, 5), dtype=float), frame_bgr=frame_bgr)
        for class_id in sorted(present_classes):
            mask = detections[:, 5].astype(int) == int(class_id)
            tracker = self._trackers_by_class.get(class_id)
            if tracker is None:
                tracker = self._new_tracker()
                self._trackers_by_class[class_id] = tracker
            tracks = self._update_tracker(tracker, detections[mask], frame_bgr)
            if tracks.size:
                out_tracks.append(tracks)
        if not out_tracks:
            return self.np.empty((0, 7), dtype=float)
        return self.np.concatenate(out_tracks, axis=0)


def _resolve_rfdetr_checkpoint(
    config: DetectionAndTrackingConfig,
    *,
    logger: logging.Logger,
) -> Path:
    explicit = os.getenv("RFDETR_MODEL_PATH")
    if explicit:
        path = Path(explicit).expanduser().resolve()
    else:
        path = resolve_model_cache_root(config.model_cache_path) / "rfdetr" / RFDETR_BASE_FILENAME
    expected_sha256 = _expected_rfdetr_sha256(path, explicit=bool(explicit))
    if path.exists() and path.stat().st_size > 0:
        if expected_sha256 is None:
            logger.info(
                "Using RF-DETR checkpoint at %s without SHA-256 verification; set %s to "
                "validate custom checkpoint files.",
                path,
                RFDETR_SHA256_ENV,
            )
            return path
        if _has_expected_sha256(path, expected_sha256=expected_sha256, logger=logger):
            return path
        if explicit or not config.allow_model_download:
            raise ValueError(
                f"RF-DETR checkpoint at {path} failed SHA-256 verification; expected "
                f"{expected_sha256}."
            )
        logger.info(
            "Existing RF-DETR checkpoint at %s failed SHA-256 verification; re-downloading.",
            path,
        )
    if not config.allow_model_download:
        raise FileNotFoundError(
            "RF-DETR checkpoint missing. Mount it at /models/rfdetr/rf-detr-base.pth, "
            "set RFDETR_MODEL_PATH, or enable allow_model_download."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    _download_rfdetr_checkpoint(
        path,
        expected_sha256=expected_sha256 or RFDETR_BASE_SHA256,
        logger=logger,
    )
    return path


def _expected_rfdetr_sha256(path: Path, *, explicit: bool) -> str | None:
    configured = os.getenv(RFDETR_SHA256_ENV)
    if configured:
        return configured.strip().lower()
    if explicit and path.name != RFDETR_BASE_FILENAME:
        return None
    return RFDETR_BASE_SHA256


def _has_expected_sha256(
    path: Path,
    *,
    expected_sha256: str,
    logger: logging.Logger,
) -> bool:
    actual_sha256 = _file_sha256(path)
    if actual_sha256 == expected_sha256:
        logger.info("Using RF-DETR checkpoint at %s verified by SHA-256.", path)
        return True
    logger.warning(
        "RF-DETR checkpoint at %s has SHA-256 %s, expected %s.",
        path,
        actual_sha256,
        expected_sha256,
    )
    return False


def _download_rfdetr_checkpoint(
    path: Path,
    *,
    expected_sha256: str,
    logger: logging.Logger,
) -> None:
    temp_path: Path | None = None
    try:
        logger.info("Downloading RF-DETR checkpoint from %s to %s", RFDETR_BASE_URL, path)
        with urllib.request.urlopen(RFDETR_BASE_URL, timeout=600) as response:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
                digest = hashlib.sha256()
                while chunk := response.read(_HASH_CHUNK_SIZE):
                    digest.update(chunk)
                    temp_file.write(chunk)
                temp_file.flush()
                os.fsync(temp_file.fileno())

        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Downloaded RF-DETR checkpoint SHA-256 mismatch: got {actual_sha256}, "
                f"expected {expected_sha256}."
            )
        temp_path.replace(path)
        logger.info("Installed verified RF-DETR checkpoint at %s", path)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _best_iou_index(np: Any, box: Any, candidates: Any) -> int:
    if len(candidates) == 0:
        return 0
    box_list = list_from_runtime(box)
    x1 = np.maximum(float(box_list[0]), candidates[:, 0])
    y1 = np.maximum(float(box_list[1]), candidates[:, 1])
    x2 = np.minimum(float(box_list[2]), candidates[:, 2])
    y2 = np.minimum(float(box_list[3]), candidates[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    box_area = max(0.0, float(box_list[2]) - float(box_list[0])) * max(
        0.0,
        float(box_list[3]) - float(box_list[1]),
    )
    cand_area = np.maximum(0.0, candidates[:, 2] - candidates[:, 0]) * np.maximum(
        0.0,
        candidates[:, 3] - candidates[:, 1],
    )
    iou = inter / (box_area + cand_area - inter + 1e-9)
    return cast(int, iou.argmax())


__all__ = ["RFDetrTracker"]
