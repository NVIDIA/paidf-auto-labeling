# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Meta SAM3 / SAM 3.1 Object Multiplex video tracking runtime."""

from __future__ import annotations

import collections
import inspect
import logging
import tempfile
import time
import uuid
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
from detection_and_tracking.backends.sam3_weights import resolve_sam3_weights
from detection_and_tracking.config import DetectionAndTrackingConfig
from detection_and_tracking.media import Vp9VideoWriter, decode_video_bgr


@dataclass
class _FrameChunk:
    rgb_frames: list[Any]
    bgr_frames: list[Any]
    source_indices: list[int]
    source_fps: float
    output_fps: float


class SAM3NativeRuntime:
    """Run Meta ``sam3`` predictors and emit the Auto-Labeling DAFT tracking contract."""

    def __init__(self, *, config: DetectionAndTrackingConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.cv2 = import_optional("cv2", backend="sam3")
        self.np = import_optional("numpy", backend="sam3")
        self.torch = import_optional("torch", backend="sam3")
        self._predictor: Any | None = None
        self._device = "cuda"

    def run(
        self,
        *,
        media_path: Path,
        scene_paths: ScenePaths,
        scene_ctx: SceneContext,
        prompts: list[str],
    ) -> Path | None:
        self._ensure_predictor_loaded()
        assert self._predictor is not None

        objects_by_frame: dict[int, dict[str, Any]] = {}
        instances: dict[tuple[int, str], dict[str, Any]] = {}
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
        output_prob_thresh = (
            self.config.sam3_score_threshold_detection
            if self.config.sam3_score_threshold_detection is not None
            else 0.5
        )

        try:
            with self.torch.no_grad():
                for chunk_idx, chunk in enumerate(self._iter_frame_chunks(media_path, chunk_size)):
                    chunks_seen = True
                    frame_detections = self._propagate_chunk(
                        chunk=chunk,
                        prompts=prompts,
                        chunk_idx=chunk_idx,
                        output_prob_thresh=float(output_prob_thresh),
                    )
                    for local_idx, detections in frame_detections.items():
                        source_frame = chunk.source_indices[local_idx]
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

    def _ensure_predictor_loaded(self) -> None:
        if self._predictor is not None:
            return
        if not bool(self.torch.cuda.is_available()):
            raise RuntimeError("Native SAM3 requires a CUDA-capable GPU.")
        gpu_id = first_configured_gpu(self.config.gpu_ids)
        if gpu_id is not None:
            self.torch.cuda.set_device(gpu_id)
            self._device = f"cuda:{gpu_id}"
        else:
            self._device = "cuda"

        weights = resolve_sam3_weights(self.config, runtime="native")
        checkpoint = weights.path_for_runtime
        self.logger.info(
            "Loading native SAM3 %s from %s on %s",
            self.config.sam3_version,
            checkpoint,
            self._device,
        )
        try:
            from sam3 import build_sam3_predictor  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "Native SAM3 runtime requires the Meta `sam3` package. Install it into "
                "the detection-and-tracking[sam3] image (see service README) or set "
                "sam3_runtime=transformers with sam3_version=sam3."
            ) from exc

        _ensure_sam3_start_session_compat()

        build_kwargs: dict[str, Any] = {
            "checkpoint_path": str(checkpoint),
            "version": self.config.sam3_version,
            "compile": bool(self.config.sam3_compile),
            "async_loading_frames": False,
            # NGC service images do not ship FlashAttention-3; keep SDPA path.
            "use_fa3": False,
        }
        if self.config.sam3_version == "sam3.1":
            build_kwargs["multiplex_count"] = int(self.config.sam3_multiplex_count)
            build_kwargs["max_num_objects"] = int(self.config.sam3_max_num_objects)
            build_kwargs["warm_up"] = bool(self.config.sam3_compile)
        elif gpu_id is not None:
            # Multi-GPU placement is supported by the base SAM3 predictor only.
            build_kwargs["gpus_to_use"] = [gpu_id]
        self._predictor = build_sam3_predictor(**build_kwargs)

    def _propagate_chunk(
        self,
        *,
        chunk: _FrameChunk,
        prompts: list[str],
        chunk_idx: int,
        output_prob_thresh: float,
    ) -> dict[int, list[dict[str, Any]]]:
        """Run one native session per prompt and merge detections by local frame index."""
        assert self._predictor is not None
        merged: dict[int, list[dict[str, Any]]] = {idx: [] for idx in range(len(chunk.bgr_frames))}
        next_obj_id_offset = 0
        with tempfile.TemporaryDirectory(prefix="upa-sam3-native-") as tmp:
            frame_dir = Path(tmp) / "frames"
            frame_dir.mkdir(parents=True, exist_ok=True)
            for local_idx, frame_bgr in enumerate(chunk.bgr_frames):
                frame_path = frame_dir / f"{local_idx:05d}.jpg"
                ok = self.cv2.imwrite(str(frame_path), frame_bgr)
                if not ok:
                    raise RuntimeError(f"Failed to write temporary SAM3 frame: {frame_path}")

            for prompt in prompts:
                response = self._predictor.handle_request(
                    {"type": "start_session", "resource_path": str(frame_dir)}
                )
                session_id = response["session_id"]
                try:
                    self._predictor.handle_request(
                        {
                            "type": "add_prompt",
                            "session_id": session_id,
                            "frame_index": 0,
                            "text": prompt,
                        }
                    )
                    request = {
                        "type": "propagate_in_video",
                        "session_id": session_id,
                        "propagation_direction": "forward",
                        "start_frame_index": 0,
                        "max_frame_num_to_track": len(chunk.bgr_frames),
                        "output_prob_thresh": output_prob_thresh,
                    }
                    for stream_response in self._predictor.handle_stream_request(request):
                        local_idx = int(stream_response["frame_index"])
                        if local_idx < 0 or local_idx >= len(chunk.bgr_frames):
                            continue
                        detections = self._detections_from_native_outputs(
                            stream_response.get("outputs") or {},
                            prompt=prompt,
                            chunk_idx=chunk_idx,
                            obj_id_offset=next_obj_id_offset,
                        )
                        merged[local_idx].extend(detections)
                    prompt_max = self._max_raw_obj_id(merged, prompt=prompt, chunk_idx=chunk_idx)
                    if prompt_max >= next_obj_id_offset:
                        next_obj_id_offset = prompt_max + 1
                finally:
                    self._predictor.handle_request(
                        {"type": "close_session", "session_id": session_id}
                    )
        return merged

    def _detections_from_native_outputs(
        self,
        outputs: dict[str, Any],
        *,
        prompt: str,
        chunk_idx: int,
        obj_id_offset: int,
    ) -> list[dict[str, Any]]:
        obj_ids = [int(v) for v in list_from_runtime(outputs.get("out_obj_ids", []))]
        masks = outputs.get("out_binary_masks")
        boxes_xywh = outputs.get("out_boxes_xywh")
        scores_raw = outputs.get("out_probs")
        if scores_raw is None:
            scores_raw = outputs.get("out_scores")
        scores = list_from_runtime(scores_raw) if scores_raw is not None else None
        if masks is None or not obj_ids:
            return []

        detections: list[dict[str, Any]] = []
        for idx, raw_object_id in enumerate(obj_ids):
            mask = self._mask_to_bool(masks[idx])
            if not bool(mask.any()):
                continue
            shifted_id = int(raw_object_id) + int(obj_id_offset)
            if boxes_xywh is not None and idx < len(boxes_xywh):
                box_xyxy = self._xywh_to_xyxy(list_from_runtime(boxes_xywh[idx]), mask.shape)
            else:
                box_xyxy = self._box_from_mask(mask)
            detection: dict[str, Any] = {
                "prompt": prompt,
                "instance_id": shifted_id,
                "object_id": self._object_id(
                    prompt=prompt,
                    chunk_idx=chunk_idx,
                    raw_object_id=shifted_id,
                ),
                "display_id": self._display_id(chunk_idx=chunk_idx, raw_object_id=shifted_id),
                "box_xyxy": [round(as_float(v, 0.0), 2) for v in box_xyxy],
                "mask": mask,
                "contours_xy": self._contours_xy(mask),
            }
            if scores is not None and idx < len(scores):
                detection["score"] = round(as_float(scores[idx], 0.0), 4)
            detections.append(detection)
        return detections

    def _max_raw_obj_id(
        self,
        merged: dict[int, list[dict[str, Any]]],
        *,
        prompt: str,
        chunk_idx: int,
    ) -> int:
        max_id = -1
        prefix = self._object_id(prompt=prompt, chunk_idx=chunk_idx, raw_object_id=0).rsplit(
            "_", 1
        )[0]
        for detections in merged.values():
            for det in detections:
                if str(det["prompt"]) != prompt:
                    continue
                object_id = str(det["object_id"])
                if not object_id.startswith(prefix + "_"):
                    continue
                try:
                    max_id = max(max_id, int(object_id.rsplit("_", 1)[-1]))
                except ValueError:
                    continue
        return max_id

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

    def _xywh_to_xyxy(self, box_xywh: list[Any], shape: tuple[int, ...]) -> list[float]:
        x, y, w, h = [as_float(v, 0.0) for v in box_xywh[:4]]
        # Native predictor may emit normalized or absolute coordinates.
        if max(x, y, w, h) <= 1.5 and len(shape) >= 2:
            height, width = int(shape[0]), int(shape[1])
            x *= width
            y *= height
            w *= width
            h *= height
        return [x, y, x + w, y + h]

    def _box_from_mask(self, mask: Any) -> list[float]:
        ys, xs = self.np.where(mask)
        if len(xs) == 0:
            return [0.0, 0.0, 0.0, 0.0]
        return [
            float(xs.min()),
            float(ys.min()),
            float(xs.max() + 1),
            float(ys.max() + 1),
        ]

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
        _, frame_width = out.shape[:2]
        for det in detections:
            colour = prompt_colours.get(str(det["prompt"]), (200, 200, 200))
            contours = self._contours_from_mask(det["mask"])
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


_SAM3_START_SESSION_PATCHED = False


def _ensure_sam3_start_session_compat() -> None:
    """Filter init_state kwargs for older Meta sam3 wheels.

    GitHub ``main`` builds of ``Sam3BasePredictor.start_session`` pass
    ``offload_state_to_cpu`` unconditionally, but SAM 3.1 multiplex
    ``init_state`` rejects that argument. Newer sam3 sources already filter
    kwargs by signature; patch only when that filter is missing.
    """
    global _SAM3_START_SESSION_PATCHED
    if _SAM3_START_SESSION_PATCHED:
        return

    from sam3.model.sam3_base_predictor import Sam3BasePredictor  # noqa: PLC0415

    if "init_state_params" in inspect.getsource(Sam3BasePredictor.start_session):
        _SAM3_START_SESSION_PATCHED = True
        return

    def start_session(
        self: Any,
        resource_path: Any,
        session_id: Any = None,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
    ) -> dict[str, Any]:
        init_kwargs: dict[str, Any] = {
            "resource_path": resource_path,
            "offload_video_to_cpu": offload_video_to_cpu,
            "offload_state_to_cpu": offload_state_to_cpu,
        }
        if hasattr(self, "async_loading_frames"):
            init_kwargs["async_loading_frames"] = self.async_loading_frames
        if hasattr(self, "video_loader_type"):
            init_kwargs["video_loader_type"] = self.video_loader_type
        init_state_params = inspect.signature(self.model.init_state).parameters
        accepts_var_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in init_state_params.values()
        )
        if not accepts_var_kwargs:
            init_kwargs = {
                key: value for key, value in init_kwargs.items() if key in init_state_params
            }
        inference_state = self.model.init_state(**init_kwargs)
        if not session_id:
            session_id = str(uuid.uuid4())
        self._all_inference_states[session_id] = {
            "state": inference_state,
            "session_id": session_id,
            "start_time": time.time(),
            "last_use_time": time.time(),
        }
        return {"session_id": session_id}

    Sam3BasePredictor.start_session = start_session
    _SAM3_START_SESSION_PATCHED = True


__all__ = ["SAM3NativeRuntime"]
