# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SAM3 text-prompted video tracking backend."""

from __future__ import annotations

import collections
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core import SceneContext, ScenePaths, is_image_path

from detection_and_tracking.backends.runtime import (
    as_float,
    first_configured_gpu,
    import_optional,
    list_from_runtime,
    media_output_suffix,
    write_tracking_artifacts,
)
from detection_and_tracking.backends.sam3_identity import (
    TrackLabelAllocator,
    format_sam3_overlay_label,
    resolve_sam3_chunk_size,
    resolve_sam3_frame_step,
)
from detection_and_tracking.backends.sam3_weights import (
    resolve_sam3_runtime,
    resolve_sam3_weights,
)
from detection_and_tracking.config import DetectionAndTrackingConfig
from detection_and_tracking.media import Vp9VideoWriter, decode_video_bgr
from detection_and_tracking.tracker import Tracker, TrackingResult


class SAM3Tracker(Tracker):
    """SAM3 backend migrated from cosmos-curator's chunked bbox stage."""

    def __init__(self, logger: logging.Logger, config: DetectionAndTrackingConfig) -> None:
        super().__init__(logger)
        self.config = config
        self.prompts = tuple(config.sam3_prompts or config.classes)
        if not self.prompts:
            raise ValueError("SAM3 tracker requires sam3_prompts or classes.")
        runtime_kind = resolve_sam3_runtime(config)
        runtime: Any
        if runtime_kind == "native":
            from detection_and_tracking.backends.sam3_native import (  # noqa: PLC0415
                SAM3NativeRuntime,
            )

            runtime = SAM3NativeRuntime(config=config, logger=logger)
        else:
            runtime = _SAM3Runtime(config=config, logger=logger)
        self._runtime = runtime
        logger.info(
            "SAM3 tracker using runtime=%s version=%s",
            runtime_kind,
            config.sam3_version,
        )

    def run(
        self,
        media_path: Path,
        scene_paths: ScenePaths,
        *,
        scene_ctx: SceneContext,
    ) -> TrackingResult:
        annotated_path = self._runtime.run(
            media_path=Path(media_path),
            scene_paths=scene_paths,
            scene_ctx=scene_ctx,
            prompts=list(self.prompts),
        )
        return TrackingResult(
            success=True,
            objects_json=scene_paths.contextual_dir / "objects.json",
            instances_json=scene_paths.contextual_dir / "instances.json",
            tracking_overlay=annotated_path,
            annotated_video=annotated_path,
        )


@dataclass
class _FrameChunk:
    rgb_frames: list[Any]
    bgr_frames: list[Any]
    source_indices: list[int]
    source_fps: float
    output_fps: float


class _SAM3Runtime:
    def __init__(self, *, config: DetectionAndTrackingConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.cv2 = import_optional("cv2", backend="sam3")
        self.np = import_optional("numpy", backend="sam3")
        self.torch = import_optional("torch", backend="sam3")
        transformers = import_optional("transformers", backend="sam3")
        self._processor_cls = transformers.Sam3VideoProcessor
        self._model_cls = transformers.Sam3VideoModel
        self._config_cls = transformers.Sam3VideoConfig
        self._processor: Any | None = None
        self._model: Any | None = None
        self._device = "cuda"
        self._dtype: Any | None = None

    def run(
        self,
        *,
        media_path: Path,
        scene_paths: ScenePaths,
        scene_ctx: SceneContext,
        prompts: list[str],
    ) -> Path | None:
        self._ensure_model_loaded()
        objects_by_frame: dict[int, dict[str, Any]] = {}
        instances: dict[tuple[int, str], dict[str, Any]] = {}
        # SAM3 resets instance ids each internal session, so a raw id is only
        # unique within a chunk. This monotonic counter assigns a video-global
        # track_id per distinct (chunk, object) so the track_XXXX crop dirs and
        # downstream PAS identity never merge different people across sessions.
        next_track_id = 0
        track_labels = TrackLabelAllocator()
        trails: collections.defaultdict[str, list[tuple[int, int]]] = collections.defaultdict(list)
        chunk_size = resolve_sam3_chunk_size(self.config)
        write_annotated = (
            self.config.save_video
            or self.config.save_red_id_overlay
            or self.config.sam3_write_annotated_video
        )
        write_masks = bool(self.config.sam3_write_masks)
        mask_root = scene_paths.sidecars_dir / "sam3" / "masks" if write_masks else None
        if mask_root is not None:
            mask_root.mkdir(parents=True, exist_ok=True)
        chunks_seen = False
        annotated_path: Path | None = None
        annotated_writer: Any | None = None

        assert self._processor is not None
        assert self._model is not None
        assert self._dtype is not None
        try:
            with self.torch.no_grad():
                for chunk_idx, chunk in enumerate(self._iter_frame_chunks(media_path, chunk_size)):
                    chunks_seen = True
                    session = self._processor.init_video_session(
                        video=chunk.rgb_frames,
                        inference_device=self._device,
                        video_storage_device="cpu",
                        dtype=self._dtype,
                    )
                    for prompt in prompts:
                        self._processor.add_text_prompt(session, prompt)
                    for outputs in self._model.propagate_in_video_iterator(
                        inference_session=session,
                        show_progress_bar=False,
                    ):
                        local_idx = int(outputs.frame_idx)
                        if local_idx >= len(chunk.rgb_frames):
                            continue
                        source_frame = chunk.source_indices[local_idx]
                        processed = self._processor.postprocess_outputs(session, outputs)
                        detections = self._detections_from_processed(processed, prompts)
                        for det in detections:
                            raw_object_id = int(det["instance_id"])
                            object_id = self._object_id(
                                prompt=str(det["prompt"]),
                                chunk_idx=chunk_idx,
                                raw_object_id=raw_object_id,
                            )
                            det["object_id"] = object_id
                            det["display_id"] = self._display_id(
                                chunk_idx=chunk_idx,
                                raw_object_id=raw_object_id,
                            )
                        frame_bgr = chunk.bgr_frames[local_idx]
                        height, width = frame_bgr.shape[:2]
                        time_s = (
                            round(source_frame / chunk.source_fps, 3)
                            if chunk.source_fps > 0
                            else 0.0
                        )
                        for det in detections:
                            key = (chunk_idx, str(det["object_id"]))
                            entry = instances.get(key)
                            if entry is None:
                                track_label = track_labels.next_label(str(det["prompt"]))
                                entry = {
                                    "object_id": str(det["object_id"]),
                                    "object_type": det["prompt"],
                                    "track_id": next_track_id,
                                    "track_label": track_label,
                                    "display_id": str(det["display_id"]),
                                    "first_frame": source_frame,
                                    "last_frame": source_frame,
                                    "start_time_s": time_s,
                                    "end_time_s": time_s,
                                    "frame_count": 0,
                                    "detection_score": None,
                                }
                                instances[key] = entry
                                next_track_id += 1
                            det["track_label"] = str(entry["track_label"])
                            entry["last_frame"] = source_frame
                            entry["end_time_s"] = time_s
                            entry["frame_count"] = int(entry["frame_count"]) + 1
                            score = det.get("score")
                            if score is not None:
                                current = entry["detection_score"]
                                entry["detection_score"] = (
                                    float(score)
                                    if current is None
                                    else max(float(current), float(score))
                                )
                            if mask_root is not None:
                                self._write_mask_png(
                                    mask_root,
                                    frame_number=source_frame,
                                    track_label=str(entry["track_label"]),
                                    mask=det["mask"],
                                )
                        objects_by_frame[source_frame] = {
                            "frame_number": source_frame,
                            "width": width,
                            "height": height,
                            "instances": [self._frame_instance(det) for det in detections],
                        }
                        if write_annotated:
                            for det in detections:
                                trails[str(det["object_id"])].append(self._center(det))
                            annotated_frame = self._draw_annotated_frame(
                                frame_bgr,
                                detections,
                                prompts,
                                trails,
                                current_time_s=time_s,
                            )
                            if annotated_path is None:
                                annotated_path = self._annotated_output_path(
                                    media_path,
                                    scene_paths,
                                    scene_ctx,
                                )
                                annotated_path.parent.mkdir(parents=True, exist_ok=True)
                                annotated_path.unlink(missing_ok=True)
                            if is_image_path(media_path):
                                ok = self.cv2.imwrite(str(annotated_path), annotated_frame)
                                if not ok:
                                    raise RuntimeError(
                                        f"Failed to write SAM3 annotated image: {annotated_path}"
                                    )
                            else:
                                if annotated_writer is None:
                                    annotated_writer = self._open_annotated_video_writer(
                                        annotated_path,
                                        annotated_frame,
                                        output_fps=chunk.output_fps,
                                    )
                                annotated_writer.write(annotated_frame)
                    trails.clear()
                    del session
                    self.torch.cuda.empty_cache()
        finally:
            if annotated_writer is not None:
                annotated_writer.close()

        if not chunks_seen:
            write_tracking_artifacts(
                scene_paths=scene_paths,
                scene_ctx=scene_ctx,
                frames=[],
                instances=[],
            )
            return None

        frames = [frame for _, frame in sorted(objects_by_frame.items())]
        write_tracking_artifacts(
            scene_paths=scene_paths,
            scene_ctx=scene_ctx,
            frames=frames,
            instances=sorted(
                instances.values(),
                key=lambda item: (int(item["first_frame"]), str(item["object_id"])),
            ),
        )
        return annotated_path

    def _ensure_model_loaded(self) -> None:
        if self._processor is not None and self._model is not None:
            if self._dtype is None:
                self._dtype = _sam3_runtime_dtype(self.torch)
            return
        if not bool(self.torch.cuda.is_available()):
            raise RuntimeError("SAM3 requires a CUDA-capable GPU.")
        gpu_id = first_configured_gpu(self.config.gpu_ids)
        if gpu_id is not None:
            self.torch.cuda.set_device(gpu_id)
            self._device = f"cuda:{gpu_id}"
        else:
            self._device = "cuda"
        weights = resolve_sam3_weights(self.config, runtime="transformers")
        model_dir = weights.path_for_runtime
        self._dtype = _sam3_runtime_dtype(self.torch)
        self.logger.info(
            "Loading Transformers SAM3 from %s on %s with dtype %s",
            model_dir,
            self._device,
            self._dtype,
        )
        self._processor = self._processor_cls.from_pretrained(model_dir, local_files_only=True)
        model_kwargs: dict[str, Any] = {
            "torch_dtype": self._dtype,
            "local_files_only": True,
        }
        overrides = _sam3_config_overrides(self.config)
        if overrides:
            model_config = self._config_cls.from_pretrained(model_dir, local_files_only=True)
            for key, value in overrides.items():
                if not hasattr(model_config, key):
                    raise ValueError(f"SAM3 config does not support override {key!r}.")
                setattr(model_config, key, value)
            model_kwargs["config"] = model_config
            self.logger.info("SAM3 config overrides: %s", overrides)
        self._model = self._model_cls.from_pretrained(model_dir, **model_kwargs).to(self._device)
        self._model.eval()

    def _iter_frame_chunks(self, media_path: Path, chunk_size: int) -> Iterator[_FrameChunk]:
        if is_image_path(media_path):
            bgr = self.cv2.imread(str(media_path))
            if bgr is None:
                raise RuntimeError(f"Could not read image: {media_path}")
            rgb = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
            yield _FrameChunk(
                rgb_frames=[rgb],
                bgr_frames=[bgr],
                source_indices=[0],
                source_fps=1.0,
                output_fps=1.0,
            )
            return

        decoded = decode_video_bgr(media_path, cv2=self.cv2, np=self.np)
        source_fps = decoded.stream.fps or 30.0
        duration_seconds = decoded.stream.duration_seconds
        if duration_seconds is None:
            frame_count = getattr(decoded.stream, "frame_count", None)
            if frame_count is not None:
                duration_seconds = frame_count / source_fps
        if duration_seconds is not None and duration_seconds > self.config.sam3_max_duration_s:
            raise RuntimeError(
                f"SAM3 input duration {duration_seconds:.2f}s exceeds "
                f"sam3_max_duration_s={self.config.sam3_max_duration_s:.2f}s"
            )
        step = resolve_sam3_frame_step(source_fps, self.config.sam3_target_fps)
        output_fps = source_fps / step
        rgb_frames: list[Any] = []
        bgr_frames: list[Any] = []
        source_indices: list[int] = []
        idx = 0
        for idx, bgr in enumerate(decoded.frames):
            if idx / source_fps > self.config.sam3_max_duration_s:
                raise RuntimeError(
                    f"SAM3 input duration exceeds "
                    f"sam3_max_duration_s={self.config.sam3_max_duration_s:.2f}s"
                )
            if idx % step == 0:
                rgb_frames.append(self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB))
                bgr_frames.append(bgr)
                source_indices.append(idx)
                if len(rgb_frames) >= chunk_size:
                    yield _FrameChunk(
                        rgb_frames=rgb_frames,
                        bgr_frames=bgr_frames,
                        source_indices=source_indices,
                        source_fps=source_fps,
                        output_fps=output_fps,
                    )
                    rgb_frames = []
                    bgr_frames = []
                    source_indices = []
        if rgb_frames:
            yield _FrameChunk(
                rgb_frames=rgb_frames,
                bgr_frames=bgr_frames,
                source_indices=source_indices,
                source_fps=source_fps,
                output_fps=output_fps,
            )

    def _detections_from_processed(
        self,
        processed: dict[str, Any],
        prompts: list[str],
    ) -> list[dict[str, Any]]:
        obj_ids = [int(v) for v in list_from_runtime(processed["object_ids"])]
        boxes = processed["boxes"]
        masks = processed["masks"]
        prompt_to_obj_ids = processed["prompt_to_obj_ids"]
        raw_scores = processed.get("scores")
        scores = list_from_runtime(raw_scores) if raw_scores is not None else None
        detections: list[dict[str, Any]] = []
        for prompt in prompts:
            for object_id_value in prompt_to_obj_ids.get(prompt, []):
                object_id = int(object_id_value)
                if object_id not in obj_ids:
                    continue
                idx = obj_ids.index(object_id)
                mask = self._mask_to_bool(masks[idx])
                if not bool(mask.any()):
                    continue
                detection: dict[str, Any] = {
                    "prompt": prompt,
                    "instance_id": int(object_id),
                    "box_xyxy": [round(as_float(v, 0.0), 2) for v in list_from_runtime(boxes[idx])],
                    "mask": mask,
                    "contours_xy": self._contours_xy(mask),
                }
                if scores is not None and idx < len(scores):
                    detection["score"] = round(as_float(scores[idx], 0.0), 4)
                detections.append(detection)
        return detections

    def _mask_to_bool(self, mask_value: Any) -> Any:
        if hasattr(mask_value, "detach"):
            mask_value = mask_value.detach().cpu().numpy()
        mask = self.np.asarray(mask_value).astype(bool)
        if len(mask.shape) > 2:
            mask = self.np.squeeze(mask)
        return mask

    def _contours_xy(self, mask: Any) -> list[list[int]]:
        mask_u8 = mask.astype(self.np.uint8) * 255
        contours, _ = self.cv2.findContours(
            mask_u8,
            self.cv2.RETR_EXTERNAL,
            self.cv2.CHAIN_APPROX_SIMPLE,
        )
        return [[int(v) for v in contour.flatten().tolist()] for contour in contours]

    @staticmethod
    def _object_id(*, prompt: str, chunk_idx: int, raw_object_id: int) -> str:
        clean_prompt = prompt.strip().lower().replace(" ", "_") or "object"
        return f"{clean_prompt}_{chunk_idx}_{raw_object_id}"

    @staticmethod
    def _display_id(*, chunk_idx: int, raw_object_id: int) -> str:
        return f"{chunk_idx}:{raw_object_id}"

    @staticmethod
    def _frame_instance(det: dict[str, Any]) -> dict[str, Any]:
        instance: dict[str, Any] = {
            "object_id": str(det["object_id"]),
            "instance_id": int(det["instance_id"]),
            "display_id": str(det["display_id"]),
            "bounding_box_2d_tight": det["box_xyxy"],
            "prompt": det["prompt"],
            "contours_xy": det["contours_xy"],
        }
        if det.get("track_label") is not None:
            instance["track_label"] = str(det["track_label"])
        if det.get("score") is not None:
            instance["score"] = float(det["score"])
        return instance

    def _write_mask_png(
        self,
        mask_root: Path,
        *,
        frame_number: int,
        track_label: str,
        mask: Any,
    ) -> None:
        safe_label = str(track_label).replace("/", "_")
        out_path = mask_root / f"frame_{frame_number:06d}_{safe_label}.png"
        mask_u8 = mask.astype(self.np.uint8) * 255
        ok = self.cv2.imwrite(str(out_path), mask_u8)
        if not ok:
            raise RuntimeError(f"Failed to write SAM3 mask sidecar: {out_path}")

    @staticmethod
    def _center(det: dict[str, Any]) -> tuple[int, int]:
        x1, y1, x2, y2 = [float(v) for v in det["box_xyxy"]]
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def _draw_annotated_frame(
        self,
        frame_bgr: Any,
        detections: list[dict[str, Any]],
        prompts: list[str],
        trails: dict[str, list[tuple[int, int]]],
        *,
        current_time_s: float | None,
    ) -> Any:
        out = frame_bgr.copy()
        colours = [
            (0, 200, 255),
            (255, 80, 80),
            (80, 255, 80),
            (255, 80, 255),
            (80, 255, 255),
        ]
        prompt_colours = {prompt: colours[idx % len(colours)] for idx, prompt in enumerate(prompts)}
        mask_opacity = self.config.sam3_annotated_video_mask_opacity / 100.0
        label_style = self.config.sam3_annotated_video_label_style
        shape = self.config.sam3_annotated_video_shape
        _, frame_width = out.shape[:2]
        for det in detections:
            colour = prompt_colours.get(str(det["prompt"]), (200, 200, 200))
            contours = self._contours_from_mask(det["mask"]) if shape == "contour" else []
            if shape == "box":
                x1, y1, x2, y2 = [int(round(float(v))) for v in det["box_xyxy"]]
                if mask_opacity > 0.0:
                    overlay = out.copy()
                    self.cv2.rectangle(
                        overlay, (x1, y1), (x2, y2), colour, thickness=self.cv2.FILLED
                    )
                    blended = self.cv2.addWeighted(
                        overlay, mask_opacity, out, 1.0 - mask_opacity, 0
                    )
                    out[y1:y2, x1:x2] = blended[y1:y2, x1:x2]
                self.cv2.rectangle(out, (x1, y1), (x2, y2), colour, thickness=2)
            else:
                if mask_opacity > 0.0 and contours:
                    overlay = out.copy()
                    self.cv2.drawContours(
                        overlay,
                        contours,
                        -1,
                        colour,
                        thickness=self.cv2.FILLED,
                    )
                    mask_bool = det["mask"].astype(bool)
                    blended = self.cv2.addWeighted(
                        overlay,
                        mask_opacity,
                        out,
                        1.0 - mask_opacity,
                        0,
                    )
                    out[mask_bool] = blended[mask_bool]
                if contours:
                    self.cv2.drawContours(out, contours, -1, colour, thickness=2)

            label = format_sam3_overlay_label(
                label_style=label_style,
                display_id=str(det["display_id"]),
                prompt=str(det["prompt"]),
                track_label=det.get("track_label"),
            )
            if label is not None:
                anchor_x, anchor_y = self._label_anchor(det, contours)
                (text_width, text_height), _ = self.cv2.getTextSize(
                    label,
                    self.cv2.FONT_HERSHEY_DUPLEX,
                    0.6,
                    1,
                )
                origin_x = min(
                    max(0, anchor_x - text_width // 2),
                    max(0, frame_width - text_width),
                )
                origin_y = max(text_height + 2, anchor_y - 6)
                self.cv2.putText(
                    out,
                    label,
                    (origin_x, origin_y),
                    self.cv2.FONT_HERSHEY_DUPLEX,
                    0.6,
                    (0, 0, 0),
                    3,
                    self.cv2.LINE_AA,
                )
                self.cv2.putText(
                    out,
                    label,
                    (origin_x, origin_y),
                    self.cv2.FONT_HERSHEY_DUPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    self.cv2.LINE_AA,
                )

            if self.config.sam3_annotated_video_trails:
                trail = trails.get(str(det["object_id"]), [])
                if len(trail) >= 2:
                    points = self.np.array(trail, dtype=self.np.int32).reshape(-1, 1, 2)
                    self.cv2.polylines(out, [points], isClosed=False, color=colour, thickness=2)
        if current_time_s is not None:
            self._draw_timestamp(out, current_time_s)
        return out

    def _contours_from_mask(self, mask: Any) -> Any:
        mask_u8 = mask.astype(self.np.uint8) * 255
        contours, _ = self.cv2.findContours(
            mask_u8,
            self.cv2.RETR_EXTERNAL,
            self.cv2.CHAIN_APPROX_SIMPLE,
        )
        return contours

    def _label_anchor(self, det: dict[str, Any], contours: Any) -> tuple[int, int]:
        if contours:
            all_points = self.np.vstack([contour.reshape(-1, 2) for contour in contours])
            anchor_y = int(all_points[:, 1].min())
            top_points = all_points[all_points[:, 1] == anchor_y, 0]
            return int(self.np.median(top_points)), anchor_y
        x1, y1, _, _ = [int(round(float(v))) for v in det["box_xyxy"]]
        return x1, y1

    def _draw_timestamp(self, frame_bgr: Any, current_time_s: float) -> None:
        text = f"t={current_time_s:.2f}s"
        (text_width, text_height), baseline = self.cv2.getTextSize(
            text,
            self.cv2.FONT_HERSHEY_DUPLEX,
            1.1,
            2,
        )
        margin = 8
        pad = 6
        self.cv2.rectangle(
            frame_bgr,
            (margin, margin),
            (margin + text_width + (2 * pad), margin + text_height + baseline + (2 * pad)),
            (0, 0, 0),
            thickness=self.cv2.FILLED,
        )
        self.cv2.putText(
            frame_bgr,
            text,
            (margin + pad, margin + pad + text_height),
            self.cv2.FONT_HERSHEY_DUPLEX,
            1.1,
            (255, 255, 255),
            2,
            self.cv2.LINE_AA,
        )

    def _annotated_output_path(
        self,
        media_path: Path,
        scene_paths: ScenePaths,
        scene_ctx: SceneContext,
    ) -> Path:
        return (
            scene_paths.sidecars_dir
            / "sam3"
            / (
                f"{scene_ctx.media_id}_annotated_"
                f"{self.config.sam3_annotated_video_label_style}.{media_output_suffix(media_path)}"
            )
        )

    def _open_annotated_video_writer(
        self,
        output_path: Path,
        frame: Any,
        *,
        output_fps: float,
    ) -> Any:
        height, width = frame.shape[:2]
        return Vp9VideoWriter(
            output_path,
            width=width,
            height=height,
            fps=output_fps,
        )


def _sam3_runtime_dtype(torch: Any) -> Any:
    is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
    if is_bf16_supported is None:
        return torch.float16
    try:
        if bool(is_bf16_supported(including_emulation=False)):
            return torch.bfloat16
    except TypeError:
        if bool(is_bf16_supported()):
            return torch.bfloat16
    return torch.float16


def _sam3_config_overrides(config: DetectionAndTrackingConfig) -> dict[str, Any]:
    candidates: dict[str, Any | None] = {
        "score_threshold_detection": config.sam3_score_threshold_detection,
        "det_nms_thresh": config.sam3_det_nms_thresh,
        "new_det_thresh": config.sam3_new_det_thresh,
        "fill_hole_area": config.sam3_fill_hole_area,
        "recondition_every_nth_frame": config.sam3_recondition_every_nth_frame,
        "recondition_on_trk_masks": config.sam3_recondition_on_trk_masks,
        "high_conf_thresh": config.sam3_high_conf_thresh,
        "high_iou_thresh": config.sam3_high_iou_thresh,
    }
    return {key: value for key, value in candidates.items() if value is not None}


__all__ = ["SAM3Tracker"]
