#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
RFDETR + Tracking Script

Processes a single video using RFDETR for object detection and a tracker
(ByteTrack, Deep-OC-SORT, or BoostTrack) for multi-object tracking.

Outputs are written into a DAFT scene directory::

    <output_dir>/
    ├── contextual/
    │   ├── instances.json                   # tracked-object catalogue (DAFT)
    │   └── objects.json                     # per-frame detections (DAFT)
    └── sidecars/                            # diagnostic / debug only
        ├── <video_id>.<ext>                 # (copy_video) source media copy
        ├── rgb/                             # (save_rgb) extracted RGB frames
        ├── vis_detection/                   # (save_vis) detection frame overlays
        ├── vis_tracking/                    # (save_vis) tracking frame overlays
        ├── <video_id>_detection.<ext>       # (save_video) detection overlay
        ├── <video_id>_tracking.<ext>        # (save_video) tracking overlay
        └── <video_id>_tracking_red_id.<ext> # (save_video_red_id) red-id overlay

``<ext>`` is ``mp4`` for video inputs and ``png`` for image inputs.

``instances.json`` and ``objects.json`` are written in DAFT v3 format via
``daft_export``; overlay/vis artifacts under ``sidecars/`` are diagnostic and
not DAFT-validated.
"""

import colorsys
import inspect
import logging
import os
import shutil
import tempfile
from fractions import Fraction
from pathlib import Path
from threading import Lock
from typing import Any, Literal, Optional, Sequence

import av
import cv2
import numpy as np
import supervision as sv
import torch
from al_utils.ckpts import ensure_url_downloaded, resolve_ckpts_root
from al_utils.media_paths import is_image_path
from al_utils.schema.config import PipelineConfig
from al_utils.schema.tracking import DetectionAndTrackingConfig
from daft_export.common import get_scene_media_id, write_daft_json
from daft_export.paths import scene_paths
from daft_export.tracking import to_daft_instances, to_daft_objects
from detection_and_tracking.base import BaseTracker, TrackingResult
from detection_and_tracking.boosttrack.boost_track import BoostTrack as _BoostTrack  # vendored

# Vendored from the standalone Deep-OC-SORT repo (MIT).
from detection_and_tracking.deepocsort.ocsort import OCSort as _DeepOCSort
from detection_and_tracking.reid.vehicle_clip_vit import load_vehicle_clip_vit_b16_256, preprocess_vehicle_clip
from PIL import Image
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES

TrackerType = Literal["bytetrack", "deepocsort", "boosttrack"]

TRACKER_CHOICES: list[str] = ["bytetrack", "deepocsort", "boosttrack"]

logger = logging.getLogger(__name__)

# MP4 writer encoder selection. This image's FFmpeg is built LGPL-pure and
# ships exactly two encoders that can legally produce MP4 output:
#   * h264_nvenc (HW) - preferred. Requires a GPU with NVENC silicon
#     (consumer RTX 20+/40/50, RTX PRO, A10/A40/L40-class GPUs). Datacenter SKUs without
#     NVENC (e.g. H100 NVL) and CI runners without a GPU will fail the probe
#     below and fall back to mpeg4.
#   * mpeg4 (SW) - MPEG-4 Part 2 fallback. Lower compression efficiency than
#     H.264 but LGPL-clean and runs anywhere. Selected only when NVENC is
#     unavailable.
_NVENC_PROBE_LOCK = Lock()
_NVENC_PROBE_RESULT: bool | None = None
_MPEG4_FALLBACK_WARNED = False


RFDETR_PRETRAIN_URLS: dict[str, str] = {
    # Mirrors rfdetr.main.HOSTED_MODELS, but we download into a host-persisted folder under ./downloads/
    "rf-detr-base.pth": "https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth",
}


def _is_nvenc_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "h264_nvenc",
            "libnvidia-encode.so",
            "nvenc",
            "minimum required nvidia driver",
        )
    )


def _probe_nvenc_available(*, force: bool = False) -> bool:
    global _NVENC_PROBE_RESULT
    with _NVENC_PROBE_LOCK:
        if _NVENC_PROBE_RESULT is not None and not force:
            return _NVENC_PROBE_RESULT

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            probe_path = Path(tmp.name)
        try:
            container = av.open(str(probe_path), mode="w")
            try:
                stream = container.add_stream("h264_nvenc", rate=Fraction(30, 1))
                stream.width = 256
                stream.height = 256
                stream.pix_fmt = "yuv420p"
                stream.options = {"preset": "p4", "tune": "hq"}

                blank = np.zeros((256, 256, 3), dtype=np.uint8)
                for packet in stream.encode(av.VideoFrame.from_ndarray(blank, format="rgb24")):
                    container.mux(packet)
                for packet in stream.encode(None):
                    container.mux(packet)
            finally:
                container.close()

            _NVENC_PROBE_RESULT = True
        except Exception as exc:
            # Probe only: _select_video_encoder() converts a False result into
            # the intended mpeg4 fallback; unrelated write errors should surface.
            if _is_nvenc_error(exc):
                _NVENC_PROBE_RESULT = False
            else:
                raise
        finally:
            try:
                os.remove(probe_path)
            except OSError:
                pass

    return _NVENC_PROBE_RESULT


def _select_video_encoder() -> tuple[str, dict[str, str]]:
    """Return (codec_name, encoder_options) for PyAV stream creation.

    Prefers h264_nvenc; falls back to mpeg4 if NVENC is unavailable. The
    mpeg4 fallback is logged once per process so the operator notices the
    lower-quality output codec.
    """
    global _MPEG4_FALLBACK_WARNED
    if _probe_nvenc_available():
        return "h264_nvenc", {"preset": "p4", "tune": "hq"}
    if not _MPEG4_FALLBACK_WARNED:
        logger.warning(
            "h264_nvenc unavailable; detection/tracking video writer falling "
            "back to mpeg4 (LGPL-clean, lower compression than H.264)."
        )
        _MPEG4_FALLBACK_WARNED = True
    return "mpeg4", {}


def _resolve_av_video_stream(container: av.container.InputContainer) -> av.video.stream.VideoStream:
    stream = next((s for s in container.streams.video), None)
    if stream is None:
        raise ValueError("No video stream found")
    return stream


def _resolve_video_fps(stream: av.video.stream.VideoStream) -> float:
    for rate in (stream.average_rate, stream.guessed_rate, stream.base_rate):
        if rate:
            fps = float(rate)
            if fps > 0:
                return fps

    if stream.duration is not None and stream.time_base is not None and stream.frames:
        duration_sec = float(stream.duration * stream.time_base)
        if duration_sec > 0:
            return float(stream.frames) / duration_sec

    return 1.0


def _resolve_total_frames(stream: av.video.stream.VideoStream, fps: float) -> int:
    if stream.frames:
        return int(stream.frames)

    if stream.duration is not None and stream.time_base is not None:
        duration_sec = float(stream.duration * stream.time_base)
        if duration_sec > 0:
            return max(1, int(round(duration_sec * fps)))

    return 0


class _PyAvVideoReader:
    def __init__(self, video_path: Path) -> None:
        try:
            self._container = av.open(str(video_path))
        except Exception as e:
            raise ValueError(f"Could not open video: {video_path}") from e

        self._stream = _resolve_av_video_stream(self._container)
        self.fps = _resolve_video_fps(self._stream)
        self.width = int(self._stream.width or 0)
        self.height = int(self._stream.height or 0)
        self.total_frames = _resolve_total_frames(self._stream, self.fps)
        if self.width <= 0 or self.height <= 0:
            self.close()
            raise ValueError(f"Could not determine video dimensions: {video_path}")

    def iter_bgr_frames(self) -> Any:
        for frame in self._container.decode(self._stream):
            rgb = frame.to_ndarray(format="rgb24")
            yield cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def close(self) -> None:
        self._container.close()


class _PyAvVideoWriter:
    def __init__(self, output_path: Path, *, fps: float, width: int, height: int) -> None:
        codec_name, codec_options = _select_video_encoder()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._container = av.open(str(output_path), mode="w")
        rate = Fraction(max(float(fps), 1e-6)).limit_denominator(1000)
        try:
            self._stream = self._container.add_stream(codec_name, rate=rate)
            self._stream.width = int(width)
            self._stream.height = int(height)
            self._stream.pix_fmt = "yuv420p"
            if codec_options:
                self._stream.options = codec_options
        except Exception:
            self._container.close()
            raise

    def write(self, frame_bgr: np.ndarray) -> None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        video_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        for packet in self._stream.encode(video_frame):
            self._container.mux(packet)

    def close(self) -> None:
        for packet in self._stream.encode(None):
            self._container.mux(packet)
        self._container.close()


class _SVByteTrackAdapter:
    """
    Adapter to provide the same `tracker.update(dets, frame)` contract expected by the rest of this script,
    implemented using `supervision`'s ByteTrack.

    Input dets: np.ndarray (N, 6) with columns [x1, y1, x2, y2, conf, cls]
    Output tracks: np.ndarray (M, 7) with columns [x1, y1, x2, y2, track_id, conf, cls]
    """

    def __init__(
        self,
        *,
        per_class: bool,
        track_buffer: int,
        match_thresh: float,
        track_thresh: float,
        min_conf: float,
    ) -> None:
        self.per_class = bool(per_class)
        self.track_buffer = int(track_buffer)
        self.match_thresh = float(match_thresh)
        self.track_thresh = float(track_thresh)
        self.min_conf = float(min_conf)
        self._tracker_single: sv.ByteTrack | None = None
        self._trackers_by_class: dict[int, sv.ByteTrack] = {}

    def _new_tracker(self) -> sv.ByteTrack:
        # Supervision's ByteTrack API has evolved; keep this resilient across versions.
        kwargs: dict[str, Any] = {}
        try:
            sig = inspect.signature(sv.ByteTrack)
            params = sig.parameters
            # Common names across versions:
            if "track_activation_threshold" in params:
                kwargs["track_activation_threshold"] = self.track_thresh
            if "minimum_matching_threshold" in params:
                kwargs["minimum_matching_threshold"] = self.match_thresh
            if "lost_track_buffer" in params:
                kwargs["lost_track_buffer"] = self.track_buffer
            # Older/alternate names:
            if "track_thresh" in params and "track_activation_threshold" not in kwargs:
                kwargs["track_thresh"] = self.track_thresh
            if "match_thresh" in params and "minimum_matching_threshold" not in kwargs:
                kwargs["match_thresh"] = self.match_thresh
            if "track_buffer" in params and "lost_track_buffer" not in kwargs:
                kwargs["track_buffer"] = self.track_buffer
        except Exception:
            kwargs = {}

        return sv.ByteTrack(**kwargs)  # type: ignore[arg-type]

    def _to_sv_detections(self, dets: np.ndarray) -> sv.Detections:
        if dets.size == 0:
            return sv.Detections(
                xyxy=np.empty((0, 4), dtype=float),
                confidence=np.empty((0,), dtype=float),
                class_id=np.empty((0,), dtype=int),
            )
        return sv.Detections(
            xyxy=dets[:, 0:4].astype(float, copy=False),
            confidence=dets[:, 4].astype(float, copy=False),
            class_id=dets[:, 5].astype(int, copy=False),
        )

    def _sv_to_tracks(self, tracked: sv.Detections) -> np.ndarray:
        # supervision attaches tracker_id; if absent, return empty
        tracker_id = getattr(tracked, "tracker_id", None)
        if tracker_id is None:
            return np.empty((0, 7), dtype=float)

        xyxy = tracked.xyxy.astype(float, copy=False)
        conf = (
            tracked.confidence.astype(float, copy=False)
            if tracked.confidence is not None
            else np.ones((len(xyxy),), dtype=float)
        )
        cls = (
            tracked.class_id.astype(int, copy=False)
            if tracked.class_id is not None
            else np.zeros((len(xyxy),), dtype=int)
        )
        tid = np.asarray(tracker_id).astype(int, copy=False)

        if len(xyxy) == 0:
            return np.empty((0, 7), dtype=float)

        # [x1,y1,x2,y2,track_id,conf,cls]
        return np.column_stack([xyxy, tid, conf, cls]).astype(float, copy=False)

    def update(self, dets: np.ndarray, frame: np.ndarray) -> np.ndarray:  # noqa: ARG002
        # Note: `frame` is unused by supervision's tracker. Kept for compatibility.
        sv_dets = self._to_sv_detections(dets)

        if not self.per_class:
            if self._tracker_single is None:
                self._tracker_single = self._new_tracker()
            tracked = self._tracker_single.update_with_detections(sv_dets)
            return self._sv_to_tracks(tracked)

        # per-class: keep an independent tracker per class id.
        out_tracks: list[np.ndarray] = []
        present_classes = set(int(c) for c in np.unique(sv_dets.class_id)) if sv_dets.class_id is not None else set()

        # Step all existing trackers even if a class has no detections this frame (so tracks can age out).
        for cid, trk in list(self._trackers_by_class.items()):
            if cid in present_classes:
                continue
            empty = sv.Detections(
                xyxy=np.empty((0, 4), dtype=float),
                confidence=np.empty((0,), dtype=float),
                class_id=np.empty((0,), dtype=int),
            )
            trk.update_with_detections(empty)

        if sv_dets.class_id is None:
            return np.empty((0, 7), dtype=float)

        for cid in sorted(present_classes):
            mask = sv_dets.class_id.astype(int) == int(cid)
            cls_dets = sv.Detections(
                xyxy=sv_dets.xyxy[mask],
                confidence=sv_dets.confidence[mask] if sv_dets.confidence is not None else None,
                class_id=sv_dets.class_id[mask],
            )
            trk = self._trackers_by_class.get(cid)
            if trk is None:
                trk = self._new_tracker()
                self._trackers_by_class[cid] = trk
            tracked = trk.update_with_detections(cls_dets)
            out_tracks.append(self._sv_to_tracks(tracked))

        if not out_tracks:
            return np.empty((0, 7), dtype=float)
        return np.concatenate(out_tracks, axis=0)


class _DeepOCSortAdapter:
    """
    Adapter around the standalone Deep-OC-SORT OCSort implementation.

    Input dets: np.ndarray (N, 6) [x1,y1,x2,y2,conf,cls]
    Output tracks: np.ndarray (M, 7) [x1,y1,x2,y2,track_id,conf,cls]
    """

    def __init__(
        self,
        *,
        per_class: bool,
        iou_threshold: float,
        det_thresh: float,
        asso_func: str,
        max_age: int,
        min_hits: int,
        use_byte: bool,
        min_conf: float,
        stage2_off: bool,
        min_hits_nonconsecutive: bool,
    ) -> None:
        self.per_class = bool(per_class)
        self.iou_threshold = float(iou_threshold)
        self.det_thresh = float(det_thresh)
        self.asso_func = str(asso_func)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.use_byte = bool(use_byte)
        self.min_conf = float(min_conf)
        self.stage2_off = bool(stage2_off)
        self.min_hits_nonconsecutive = bool(min_hits_nonconsecutive)
        self._tracker_single: _DeepOCSort | None = None
        self._trackers_by_class: dict[int, _DeepOCSort] = {}

    def _new_tracker(self) -> _DeepOCSort:
        return _DeepOCSort(
            det_thresh=self.det_thresh,
            max_age=self.max_age,
            min_hits=self.min_hits,
            iou_threshold=self.iou_threshold,
            asso_func=self.asso_func,
            use_byte=self.use_byte,
            min_conf=self.min_conf,
            stage2_off=self.stage2_off,
            min_hits_nonconsecutive=self.min_hits_nonconsecutive,
        )

    def _run_tracker(self, tracker: _DeepOCSort, dets_xyxy_conf: np.ndarray, frame: np.ndarray) -> np.ndarray:
        # Deep-OC-SORT expects:
        # - img_tensor: shape (N,C,H,W) (only shape used for rescale)
        # - img_numpy: shape (C,H,W) (only shape used for rescale)
        h, w = frame.shape[0], frame.shape[1]
        img_tensor = torch.empty((1, 3, h, w), dtype=torch.float32)
        img_numpy = frame.transpose(2, 0, 1)  # (3,H,W)
        return tracker.update(dets_xyxy_conf, img_tensor, img_numpy, tag=None)

    def update(self, dets: np.ndarray, frame: np.ndarray) -> np.ndarray:
        if dets.size == 0:
            dets_xyxy_conf = np.empty((0, 5), dtype=float)
            dets_xyxy = np.empty((0, 4), dtype=float)
            dets_conf = np.empty((0,), dtype=float)
            dets_cls = np.empty((0,), dtype=int)
        else:
            dets_xyxy = dets[:, 0:4].astype(float, copy=False)
            dets_conf = dets[:, 4].astype(float, copy=False)
            dets_cls = dets[:, 5].astype(int, copy=False)
            dets_xyxy_conf = np.column_stack([dets_xyxy, dets_conf])

        if not self.per_class:
            if self._tracker_single is None:
                self._tracker_single = self._new_tracker()
            out = self._run_tracker(self._tracker_single, dets_xyxy_conf, frame)
            return self._postprocess_tracks(out, dets_xyxy, dets_conf, dets_cls)

        out_tracks: list[np.ndarray] = []
        present_classes = set(int(c) for c in np.unique(dets_cls)) if dets.size != 0 else set()

        # Step trackers with empty dets for missing classes (so tracks age out)
        for cid, trk in list(self._trackers_by_class.items()):
            if cid in present_classes:
                continue
            _ = self._run_tracker(trk, np.empty((0, 5), dtype=float), frame)

        for cid in sorted(present_classes):
            mask = dets_cls == int(cid)
            trk = self._trackers_by_class.get(int(cid))
            if trk is None:
                trk = self._new_tracker()
                self._trackers_by_class[int(cid)] = trk
            out = self._run_tracker(trk, dets_xyxy_conf[mask], frame)
            out_tracks.append(self._postprocess_tracks(out, dets_xyxy[mask], dets_conf[mask], dets_cls[mask]))

        if not out_tracks:
            return np.empty((0, 7), dtype=float)
        return np.concatenate(out_tracks, axis=0)

    def _postprocess_tracks(
        self,
        out: np.ndarray,
        dets_xyxy: np.ndarray,
        dets_conf: np.ndarray,
        dets_cls: np.ndarray,
    ) -> np.ndarray:
        if out is None or out.size == 0:
            return np.empty((0, 7), dtype=float)
        # Deep-OC-SORT output: (M,5) [x1,y1,x2,y2,track_id]
        boxes = out[:, 0:4].astype(float, copy=False)
        tids = out[:, 4].astype(int, copy=False)

        # Propagate class id + confidence from the best-overlapping detection in this frame.
        if dets_xyxy.size == 0:
            conf = np.ones((len(boxes),), dtype=float)
            cls = np.zeros((len(boxes),), dtype=int)
        else:
            ious = _pairwise_iou_xyxy(boxes, dets_xyxy)
            best = ious.argmax(axis=1)
            best_iou = ious[np.arange(len(boxes)), best]
            conf = dets_conf[best].astype(float, copy=False)
            cls = dets_cls[best].astype(int, copy=False)
            # If match is very weak, fall back to defaults
            conf = np.where(best_iou >= 0.05, conf, 1.0)
            cls = np.where(best_iou >= 0.05, cls, 0)

        return np.column_stack([boxes, tids, conf, cls]).astype(float, copy=False)


class _BoostTrackAdapter:
    """
    Adapter around the standalone BoostTrack implementation (MIT; vendored).

    Input dets: np.ndarray (N, 6) [x1,y1,x2,y2,conf,cls]
    Output tracks: np.ndarray (M, 7) [x1,y1,x2,y2,track_id,conf,cls]

    Notes:
    - BoostTrack is historically evaluated on single-class MOT. Here we support both:
      - per_class=False: single tracker across all classes (class id propagated from the best-matching detection)
      - per_class=True: one independent tracker per class id
    """

    def __init__(
        self,
        *,
        per_class: bool,
        iou_threshold: float,
        det_thresh: float,
        max_age: int,
        min_hits: int,
        min_conf: float,
        lambda_iou: float = 0.5,
        lambda_mhd: float = 0.25,
        lambda_shape: float = 0.25,
        reid_weights: Path | None = None,
        device: str = "cuda",
    ) -> None:
        self.per_class = bool(per_class)
        self.iou_threshold = float(iou_threshold)
        self.det_thresh = float(det_thresh)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.min_conf = float(min_conf)
        self.lambda_iou = float(lambda_iou)
        self.lambda_mhd = float(lambda_mhd)
        self.lambda_shape = float(lambda_shape)
        self._BoostTrack = _BoostTrack
        self._tracker_single: _BoostTrack | None = None
        self._trackers_by_class: dict[int, _BoostTrack] = {}

        # Optional ReID embedder (Vehicle CLIP ViT-B/16 @ 256) using ckpts/reid/clip_vehicleid.pt.
        self._reid_model = None
        self._reid_cfg = None
        self._reid_device = None
        self._reid_dtype = None
        if reid_weights is not None:
            try:
                dev = torch.device(device if torch.cuda.is_available() else "cpu")
                model, cfg = load_vehicle_clip_vit_b16_256(
                    reid_weights, device=dev, dtype=torch.float16 if dev.type == "cuda" else torch.float32
                )
                self._reid_model = model
                self._reid_cfg = cfg
                self._reid_device = dev
                self._reid_dtype = torch.float16 if dev.type == "cuda" else torch.float32
                self._reid_preprocess = preprocess_vehicle_clip
            except Exception as e:
                raise RuntimeError(f"Failed to init BoostTrack ReID from {reid_weights}: {e}") from e

    def _new_tracker(self):
        trk = self._BoostTrack(
            max_age=self.max_age,
            min_hits=self.min_hits,
            det_thresh=max(self.det_thresh, self.min_conf),
            iou_threshold=self.iou_threshold,
            lambda_iou=self.lambda_iou,
            lambda_mhd=self.lambda_mhd,
            lambda_shape=self.lambda_shape,
        )
        if self._reid_model is not None:
            trk.embedder = self._embed_dets  # type: ignore[attr-defined]
        return trk

    def _embed_dets(self, frame_bgr: np.ndarray, dets_xyxy: np.ndarray) -> np.ndarray:
        # Returns L2-normalized embeddings (N,512)
        if dets_xyxy.size == 0:
            return np.empty((0, 512), dtype=np.float32)
        assert self._reid_model is not None and self._reid_device is not None and self._reid_cfg is not None

        H, W = frame_bgr.shape[0], frame_bgr.shape[1]
        crops = []
        for box in dets_xyxy:
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            x1 = max(0, min(W - 1, x1))
            y1 = max(0, min(H - 1, y1))
            x2 = max(0, min(W, x2))
            y2 = max(0, min(H, y2))
            if x2 <= x1 or y2 <= y1:
                crop = np.zeros((self._reid_cfg.image_size, self._reid_cfg.image_size, 3), dtype=np.uint8)
            else:
                crop = frame_bgr[y1:y2, x1:x2]
                crop = cv2.resize(
                    crop, (self._reid_cfg.image_size, self._reid_cfg.image_size), interpolation=cv2.INTER_LINEAR
                )
            crops.append(crop)

        batch = np.stack(crops, axis=0)  # (N,H,W,3) BGR
        batch_t = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()  # (N,3,H,W)
        x = self._reid_preprocess(batch_t, device=self._reid_device, dtype=self._reid_dtype)
        # Tracking pipeline may run under mixed grad/inference contexts; keep ReID strictly inference-only.
        # Also clone to avoid autograd state conflicts when upstream toggles modes.
        with torch.no_grad():
            feats = self._reid_model(x.clone())  # (N,512)
            feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats.detach().float().cpu().numpy()

    def _postprocess_tracks(
        self, out: np.ndarray, dets_xyxy: np.ndarray, dets_conf: np.ndarray, dets_cls: np.ndarray
    ) -> np.ndarray:
        # BoostTrack output: [x1,y1,x2,y2,track_id,track_conf]
        if out.size == 0:
            return np.empty((0, 7), dtype=float)

        xyxy = out[:, 0:4].astype(float, copy=False)
        tid = out[:, 4].astype(int, copy=False)
        tconf = out[:, 5].astype(float, copy=False) if out.shape[1] > 5 else np.ones((len(out),), dtype=float)

        # Propagate class id by matching to the closest detection (IoU) in this frame.
        if dets_xyxy.size == 0:
            cls = np.zeros((len(xyxy),), dtype=int)
            conf = tconf
        else:
            ious = _pairwise_iou_xyxy(xyxy, dets_xyxy)
            best = ious.argmax(axis=1)
            cls = dets_cls[best].astype(int, copy=False)
            # prefer detection confidence if available
            conf = dets_conf[best].astype(float, copy=False)

        return np.column_stack([xyxy, tid, conf, cls]).astype(float, copy=False)

    def update(self, dets: np.ndarray, frame: np.ndarray) -> np.ndarray:  # noqa: ARG002
        if dets.size == 0:
            dets_xyxy = np.empty((0, 4), dtype=float)
            dets_conf = np.empty((0,), dtype=float)
            dets_cls = np.empty((0,), dtype=int)
            dets_xyxy_conf = np.empty((0, 5), dtype=float)
        else:
            dets_xyxy = dets[:, 0:4].astype(float, copy=False)
            dets_conf = dets[:, 4].astype(float, copy=False)
            dets_cls = dets[:, 5].astype(int, copy=False)
            dets_xyxy_conf = np.column_stack([dets_xyxy, dets_conf])

        if not self.per_class:
            if self._tracker_single is None:
                self._tracker_single = self._new_tracker()
            out = self._tracker_single.update(dets_xyxy_conf, frame_bgr=frame)
            return self._postprocess_tracks(out, dets_xyxy, dets_conf, dets_cls)

        out_tracks: list[np.ndarray] = []
        present_classes = set(int(c) for c in np.unique(dets_cls)) if dets.size != 0 else set()

        # Step trackers with empty dets for missing classes (so tracks age out)
        for cid, trk in list(self._trackers_by_class.items()):
            if cid in present_classes:
                continue
            trk.update(np.empty((0, 5), dtype=float), frame_bgr=frame)

        for cid in sorted(present_classes):
            mask = dets_cls == int(cid)
            cls_dets = (
                np.column_stack([dets_xyxy[mask], dets_conf[mask]]) if dets.size != 0 else np.empty((0, 5), dtype=float)
            )
            trk = self._trackers_by_class.get(cid)
            if trk is None:
                trk = self._new_tracker()
                self._trackers_by_class[cid] = trk
            out = trk.update(cls_dets, frame_bgr=frame)
            out_tracks.append(self._postprocess_tracks(out, dets_xyxy[mask], dets_conf[mask], dets_cls[mask]))

        if not out_tracks:
            return np.empty((0, 7), dtype=float)
        return np.concatenate(out_tracks, axis=0)


def _normalize_class_name(name: str) -> str:
    """Normalize class name for matching."""
    return str(name).strip().lower().replace(" ", "_")


def _iter_coco_classes() -> list[tuple[int, str]]:
    """
    Return COCO classes as a list of (class_id, class_name).

    COCO_CLASSES may be either:
    - list/tuple of names: index is class_id
    - dict-like mapping {class_id: class_name}
    """
    if isinstance(COCO_CLASSES, dict):
        return [(int(k), str(v)) for k, v in COCO_CLASSES.items()]
    return [(int(i), str(n)) for i, n in enumerate(COCO_CLASSES)]


def resolve_class_ids(class_ids: Optional[Sequence[int]], class_names: Optional[Sequence[str]]) -> Optional[list[int]]:
    """
    Resolve a combined class filter from ids and/or names.

    - If both are None/empty: return None (no filtering)
    - If both provided: returns union of ids
    """
    resolved: set[int] = set()

    if class_ids:
        resolved.update(int(x) for x in class_ids)

    if class_names:
        coco_pairs = _iter_coco_classes()
        name_to_id = {_normalize_class_name(n): cid for cid, n in coco_pairs}
        unknown: list[str] = []
        for n in class_names:
            key = _normalize_class_name(n)
            if key not in name_to_id:
                unknown.append(n)
            else:
                resolved.add(name_to_id[key])

        if unknown:
            examples = [name for _, name in coco_pairs[:10]]
            raise ValueError(
                "Unknown class name(s): " + ", ".join(unknown) + ". Valid examples: " + ", ".join(examples) + " ..."
            )

    return sorted(resolved) if resolved else None


def drop_cross_class_overlaps_keep_upper(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    *,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    If two boxes of different classes overlap with IoU > iou_threshold, drop the lower box.

    For overlapping cross-class pairs, we drop the box whose vertical center is lower (larger y-center).
    Tie-breakers: keep the higher score; final tie keep the smaller area.
    """
    if boxes_xyxy.size == 0:
        return boxes_xyxy, scores, class_ids
    if boxes_xyxy.shape[0] <= 1:
        return boxes_xyxy, scores, class_ids

    boxes = boxes_xyxy.astype(np.float32, copy=False)
    cls = class_ids.astype(int, copy=False)
    sc = scores.astype(np.float32, copy=False)

    n = boxes.shape[0]
    drop = np.zeros((n,), dtype=bool)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    w = np.maximum(1.0, x2 - x1)
    h = np.maximum(1.0, y2 - y1)
    cy = (y1 + y2) * 0.5
    area = w * h

    for i in range(n):
        if drop[i]:
            continue
        for j in range(i + 1, n):
            if drop[j]:
                continue
            if cls[i] == cls[j]:
                continue

            if _iou_xyxy(boxes_xyxy[i], boxes_xyxy[j]) <= float(iou_threshold):
                continue

            if cy[i] > cy[j]:
                lower, upper = i, j
            elif cy[j] > cy[i]:
                lower, upper = j, i
            else:
                if sc[i] < sc[j]:
                    lower, upper = i, j
                elif sc[j] < sc[i]:
                    lower, upper = j, i
                else:
                    lower, upper = (i, j) if area[i] >= area[j] else (j, i)

            drop[lower] = True

    keep = ~drop
    return boxes_xyxy[keep], scores[keep], class_ids[keep]


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """
    IoU between two single boxes in [x1,y1,x2,y2].

    (This helper is used by de-dup logic; DeepOCSort postprocess uses `_pairwise_iou_xyxy`.)
    """
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size < 4 or b.size < 4:
        return 0.0
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0.0 else 0.0


def _pairwise_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU for xyxy boxes. a: (N,4), b: (M,4) -> (N,M)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=float)
    a = a.reshape((-1, 4))
    b = b.reshape((-1, 4))
    ax1 = a[:, 0:1]
    ay1 = a[:, 1:2]
    ax2 = a[:, 2:3]
    ay2 = a[:, 3:4]
    bx1 = b[:, 0][None, :]
    by1 = b[:, 1][None, :]
    bx2 = b[:, 2][None, :]
    by2 = b[:, 3][None, :]

    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return (inter / union).astype(float)


def nms_dedup_by_iou(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float = 0.95,
    priority: Literal["conf", "area_large", "area_small", "prev_iou"] = "conf",
    prev_boxes_by_class: Optional[dict[int, np.ndarray]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-frame de-duplication: remove lower-score boxes that overlap with IoU > threshold.
    Applied per class_id (to avoid suppressing different classes).
    """
    if len(boxes_xyxy) == 0:
        return boxes_xyxy, scores, class_ids

    keep_global: list[int] = []
    unique_classes = np.unique(class_ids.astype(int))

    # Precompute areas for optional area-priority sorting
    areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])

    for cid in unique_classes:
        idxs = np.where(class_ids.astype(int) == int(cid))[0]
        if idxs.size == 0:
            continue

        # Sort order controls which box "wins" when IoU > threshold:
        # - priority=conf: higher confidence wins (tie-breaker: larger area)
        # - priority=area_large: larger area wins (tie-breaker: higher confidence)
        # - priority=area_small: smaller area wins (tie-breaker: higher confidence)
        # - priority=prev_iou: higher IoU with previous-frame kept boxes (same class) wins (tie-breaker: higher confidence, larger area)
        if priority == "prev_iou":
            prev_iou = np.zeros(idxs.shape[0], dtype=float)
            if prev_boxes_by_class is not None and int(cid) in prev_boxes_by_class:
                prev_boxes = prev_boxes_by_class[int(cid)]
                if isinstance(prev_boxes, np.ndarray) and prev_boxes.size > 0:
                    for k, det_i in enumerate(idxs):
                        best = 0.0
                        for pb in prev_boxes:
                            best = max(best, _iou_xyxy(boxes_xyxy[int(det_i)], pb))
                        prev_iou[k] = best
            # idxs are the base indices; sort keys aligned with idxs order
            order = idxs[np.lexsort((-areas[idxs], -scores[idxs], -prev_iou))]
        elif priority == "area_large":
            order = idxs[np.lexsort((-scores[idxs], -areas[idxs]))]
        elif priority == "area_small":
            order = idxs[np.lexsort((-scores[idxs], areas[idxs]))]
        else:  # conf
            order = idxs[np.lexsort((-areas[idxs], -scores[idxs]))]
        suppressed = np.zeros(order.shape[0], dtype=bool)

        for i in range(order.shape[0]):
            if suppressed[i]:
                continue
            cur = int(order[i])
            keep_global.append(cur)
            # Suppress highly-overlapping lower-score boxes
            for j in range(i + 1, order.shape[0]):
                if suppressed[j]:
                    continue
                other = int(order[j])
                if _iou_xyxy(boxes_xyxy[cur], boxes_xyxy[other]) > iou_threshold:
                    suppressed[j] = True

    keep_global = sorted(set(keep_global))
    return boxes_xyxy[keep_global], scores[keep_global], class_ids[keep_global]


def generate_distinct_color(index: int, total: int = 100) -> list[int]:
    """
    Generate a distinct RGB color for a given index.

    Args:
        index: Index of the color to generate.
        total: Total number of colors to distribute across the hue spectrum.

    Returns:
        List of [R, G, B] values (0-255).
    """
    hue = (index * 0.618033988749895) % 1.0  # Golden ratio for distribution
    saturation = 0.7 + (index % 3) * 0.1
    value = 0.9 - (index % 2) * 0.1
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return [int(r * 255), int(g * 255), int(b * 255)]


def draw_detections(
    image: np.ndarray,
    bboxes: np.ndarray,
    confidences: np.ndarray,
    class_ids: np.ndarray,
    class_names: list[str],
    line_thickness: int = 1,
    font_scale: float = 0.4,
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on an image.

    Args:
        image: Input image (BGR format).
        bboxes: Bounding boxes array of shape (N, 4) in [x1, y1, x2, y2] format.
        confidences: Confidence scores array of shape (N,).
        class_ids: Class ID array of shape (N,).
        class_names: List of class names.
        line_thickness: Thickness of bounding box lines.
        font_scale: Font scale for labels.

    Returns:
        Annotated image.
    """
    img = image.copy()

    for i, (bbox, conf, cid) in enumerate(zip(bboxes, confidences, class_ids)):
        x1, y1, x2, y2 = map(int, bbox)
        cid = int(cid)

        # Generate color based on class_id
        color = generate_distinct_color(cid)
        # Convert RGB to BGR for OpenCV
        color_bgr = (color[2], color[1], color[0])

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, line_thickness)

        # Prepare label
        class_name = class_names[cid] if cid < len(class_names) else f"class_{cid}"
        label = f"{class_name} {conf:.2f}"

        # Get text size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, 1)

        # Draw label background
        label_y1 = max(y1 - text_h - 8, 0)
        label_y2 = y1
        cv2.rectangle(img, (x1, label_y1), (x1 + text_w + 4, label_y2), color_bgr, -1)

        # Draw label text
        cv2.putText(
            img,
            label,
            (x1 + 2, y1 - 4),
            font,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return img


def draw_tracks(
    image: np.ndarray,
    instances: list[dict[str, Any]],
    instances_info: dict[str, dict[str, Any]],
    class_names: list[str],
    line_thickness: int = 1,
    font_scale: float = 0.4,
    style: Literal["default", "red_id"] = "default",
) -> np.ndarray:
    """
    Draw tracking bounding boxes and labels on an image.

    Args:
        image: Input image (BGR format).
        instances: List of instance dicts with object_id, bounding_box_2d_tight, etc.
        instances_info: Dict mapping object_id to instance info (for color).
        class_names: List of class names.
        line_thickness: Thickness of bounding box lines.
        font_scale: Font scale for labels.
        style: ``"red_id"`` draws uniform red boxes with auto-scaled track-ID labels;
            ``"default"`` uses per-class colors with class+ID labels.

    Returns:
        Annotated image.
    """
    img = image.copy()

    def _draw_redbox(frame_bgr: np.ndarray, bbox_xyxy: list[float], track_id: int) -> None:
        x1, y1, x2, y2 = [int(round(x)) for x in bbox_xyxy]

        red = (0, 0, 255)  # BGR
        # Thinner box
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), red, 1)

        text = str(track_id)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Auto scale text to fit inside the bbox.
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        (tw1, th1), _ = cv2.getTextSize(text, font, 1.0, 2)
        tw1 = max(1, int(tw1))
        th1 = max(1, int(th1))
        # Leave some margin; scale down for small boxes, up for large boxes (clamped).
        max_scale_w = (box_w - 6) / tw1
        max_scale_h = (box_h - 6) / th1
        font_scale = 0.85 * max(0.1, min(max_scale_w, max_scale_h))
        font_scale = float(max(0.4, min(font_scale, 2.0)))
        # Thinner text strokes
        text_thickness = int(max(1, round(font_scale * 1.6)))
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, text_thickness)

        cx = int((x1 + x2) / 2 - tw / 2)
        cy = int((y1 + y2) / 2 + th / 2)
        cx = max(x1 + 2, min(cx, x2 - tw - 2))
        cy = max(y1 + th + 2, min(cy, y2 - 2))

        # black shadow + red text
        cv2.putText(
            frame_bgr,
            text,
            (cx + 1, cy + 1),
            font,
            font_scale,
            (0, 0, 0),
            min(3, text_thickness + 1),
            cv2.LINE_AA,
        )
        cv2.putText(frame_bgr, text, (cx, cy), font, font_scale, red, text_thickness, cv2.LINE_AA)

    for inst in instances:
        object_id = inst["object_id"]
        bbox = inst["bounding_box_2d_tight"]
        confidence = inst.get("confidence", 1.0)
        class_id = inst.get("semantic_id", 0)

        x1, y1, x2, y2 = map(int, bbox)

        if style == "red_id":
            track_id = None
            info = instances_info.get(object_id)
            if isinstance(info, dict):
                try:
                    track_id = int(info.get("track_id")) if info.get("track_id") is not None else None
                except Exception:
                    track_id = None
            if track_id is None:
                # object_id format is usually "<class>_<track_id>"
                try:
                    track_id = int(str(object_id).rsplit("_", 1)[-1])
                except Exception:
                    track_id = 0

            _draw_redbox(img, bbox_xyxy=[float(x1), float(y1), float(x2), float(y2)], track_id=int(track_id))
        else:
            # Get color from instances_info (based on track_id for consistency)
            if object_id in instances_info:
                color = instances_info[object_id]["color"]
            else:
                color = generate_distinct_color(class_id)

            # Convert RGB to BGR for OpenCV
            color_bgr = (color[2], color[1], color[0])

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, line_thickness)

            # Prepare label with track ID
            label = f"{object_id} {confidence:.2f}"

            # Get text size for background
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, 1)

            # Draw label background
            label_y1 = max(y1 - text_h - 8, 0)
            label_y2 = y1
            cv2.rectangle(img, (x1, label_y1), (x1 + text_w + 4, label_y2), color_bgr, -1)

            # Draw label text
            cv2.putText(
                img,
                label,
                (x1 + 2, y1 - 4),
                font,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return img


def generate_filtered_video(
    video_path: Path,
    output_path: Path,
    frames_data: dict[str, dict[str, Any]],
    instances_data: dict[str, dict[str, Any]],
    class_names: list[str],
    video_name: str,
    upscale: int = 1,
    track_vis_style: Literal["default", "red_id"] = "default",
) -> None:
    """
    Generate a video with only filtered tracks drawn.

    Args:
        video_path: Path to the original video.
        output_path: Path to save the filtered video.
        frames_data: Filtered frames data with instances.
        instances_data: Filtered instances data.
        class_names: List of class names.
        video_name: Video name for frame key lookup.
        upscale: Upscale factor.
        track_vis_style: Passed to ``draw_tracks``; ``"red_id"`` for red-box overlay,
            ``"default"`` for per-class colors.
    """
    # Image input: render a single annotated frame and write it to either an image
    # file (if output_path is an image) or a single-frame mp4.
    if is_image_path(video_path):
        frame = cv2.imread(str(video_path))
        if frame is None:
            raise ValueError(f"Could not read image: {video_path}")
        height, width = frame.shape[:2]
        fps = 1.0

        out_width = width * upscale
        out_height = height * upscale
        if upscale > 1:
            frame = cv2.resize(frame, (out_width, out_height))

        frame_key = f"{video_name}_frame_{0:06d}"
        if frame_key in frames_data:
            instances = frames_data[frame_key]["instances"]
            frame = draw_tracks(
                image=frame,
                instances=instances,
                instances_info=instances_data,
                class_names=class_names,
                style=track_vis_style,
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if is_image_path(Path(output_path)):
            cv2.imwrite(str(output_path), frame)
            return

        video_out = _PyAvVideoWriter(output_path, fps=float(fps), width=out_width, height=out_height)
        video_out.write(frame)
        video_out.close()
        return

    reader = _PyAvVideoReader(video_path)
    fps = reader.fps
    width = reader.width
    height = reader.height

    out_width = width * upscale
    out_height = height * upscale

    video_out = _PyAvVideoWriter(output_path, fps=fps, width=out_width, height=out_height)
    try:
        for frame_id, frame in enumerate(reader.iter_bgr_frames()):
            # Upscale if needed
            if upscale > 1:
                frame = cv2.resize(frame, (out_width, out_height))

            # Get frame data
            frame_key = f"{video_name}_frame_{frame_id:06d}"
            if frame_key in frames_data:
                instances = frames_data[frame_key]["instances"]
                # Draw filtered tracks
                frame = draw_tracks(
                    image=frame,
                    instances=instances,
                    instances_info=instances_data,
                    class_names=class_names,
                    style=track_vis_style,
                )

            video_out.write(frame)
    finally:
        reader.close()
        video_out.close()


def expand_bbox(
    bbox: list[float],
    expansion_ratio: float,
    img_width: int,
    img_height: int,
) -> list[float]:
    """
    Expand a bounding box by a given ratio (for loose bbox).

    Args:
        bbox: Original bounding box [xmin, ymin, xmax, ymax].
        expansion_ratio: Ratio to expand the bbox (e.g., 0.1 for 10%).
        img_width: Image width for clamping.
        img_height: Image height for clamping.

    Returns:
        Expanded bounding box [xmin, ymin, xmax, ymax].
    """
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin

    expand_w = width * expansion_ratio
    expand_h = height * expansion_ratio

    new_xmin = max(0, xmin - expand_w)
    new_ymin = max(0, ymin - expand_h)
    new_xmax = min(img_width, xmax + expand_w)
    new_ymax = min(img_height, ymax + expand_h)

    return [new_xmin, new_ymin, new_xmax, new_ymax]


def get_tracker(
    tracker_name: TrackerType,
    reid_weights: Path | None = None,
    per_class: bool = False,
    iou_threshold: float = 0.3,
    det_thresh: float = 0.3,
    asso_func: str = "diou",
    # Parameters to reduce ID switching
    max_age: int = 60,
    min_hits: int = 3,
    track_buffer: int = 25,
    match_thresh: float = 0.8,
    # ByteTrack occlusion handling (defaults align to ByteTrack)
    track_thresh: float = 0.45,
    min_conf: float = 0.1,
    # DeepOCSort-only toggles (standalone Deep-OC-SORT)
    deepocsort_stage2_off: bool = False,
    deepocsort_min_hits_nonconsecutive: bool = False,
    device: str = "cuda",
) -> Any:
    """Create and return a tracker instance based on the tracker name."""
    if tracker_name == "bytetrack":
        _ = (reid_weights, iou_threshold, det_thresh, asso_func, max_age, min_hits, device)
        return _SVByteTrackAdapter(
            per_class=per_class,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            track_thresh=track_thresh,
            min_conf=min_conf,
        )

    if tracker_name == "deepocsort":
        _ = (reid_weights, track_buffer, match_thresh, track_thresh, device)
        return _DeepOCSortAdapter(
            per_class=per_class,
            iou_threshold=iou_threshold,
            det_thresh=det_thresh,
            asso_func=asso_func,
            max_age=max_age,
            min_hits=min_hits,
            use_byte=True,
            min_conf=min_conf,
            stage2_off=deepocsort_stage2_off,
            min_hits_nonconsecutive=deepocsort_min_hits_nonconsecutive,
        )

    if tracker_name == "boosttrack":
        _ = (asso_func, track_buffer, match_thresh, track_thresh)
        return _BoostTrackAdapter(
            per_class=per_class,
            iou_threshold=iou_threshold,
            det_thresh=det_thresh,
            max_age=max_age,
            min_hits=min_hits,
            min_conf=min_conf,
            reid_weights=reid_weights,
            device=device,
        )

    raise ValueError(f"Unsupported tracker: {tracker_name}. Available: {TRACKER_CHOICES}")


def process_video(
    video_path: Path,
    output_dir: Path,
    model: RFDETRBase,
    tracker: Any,
    detection_threshold: float = 0.3,
    class_ids_filter: Optional[Sequence[int]] = None,
    save_vis: bool = False,
    save_video: bool = False,
    save_video_red_id: bool = False,
    bbox_expansion_ratio: float = 0.1,
    upscale: int = 1,
    min_track_frames: int = 1,
    write_json: bool = True,
    save_rgb: bool = True,
    copy_video: bool = True,
    dedup_iou_threshold: float = 0.95,
    dedup_priority: Literal["conf", "area_large", "area_small", "prev_iou"] = "conf",
    cross_class_iou_threshold: float = -1,
    video_id: Optional[str] = None,
    *,
    logger: Optional[logging.Logger] = None,
) -> dict[str, Any]:
    """Process a single media file with detection and tracking.

    Args:
        video_path: Path to the input media.
        output_dir: Per-sample scene directory. DAFT ``instances.json`` /
            ``objects.json`` land under ``contextual/``; overlays, RGB frames,
            and the optional source-media copy land under ``sidecars/``.
        model: RFDETR detection model.
        tracker: Tracker adapter instance (ByteTrack via supervision).
        detection_threshold: Confidence threshold for detections.
        class_ids_filter: Optional list of class IDs to keep (e.g. [0, 2, 3]). If None, keep all classes.
        save_vis: Whether to save visualization frames.
        save_video: Whether to save annotated overlay media.
        save_video_red_id: Whether to save extra red-id overlay media (uniform red boxes + track IDs).
        bbox_expansion_ratio: Ratio to expand bbox for loose bbox.
        upscale: Upscale factor for processing.
        min_track_frames: Minimum number of frames a track must appear in to be kept.
        write_json: Whether to write instances.json / objects.json to disk.
        save_rgb: Whether to save extracted RGB frames to disk.
        copy_video: Whether to copy the source media into ``<scene>/sidecars/``.
        dedup_iou_threshold: If > 0, per-frame de-duplication removes boxes with IoU > threshold (default: 0.95).
        dedup_priority: Used with dedup_iou_threshold. Which box to keep when IoU exceeds threshold:
            'conf' (keep higher confidence), 'area_large' (keep larger boxes), 'area_small' (keep smaller boxes),
            'prev_iou' (keep box that overlaps most with previous-frame kept boxes, same class; tie-breaker: conf, area).
            Default: conf.

    Returns:
        Dictionary containing processing summary.
    """
    input_is_image = is_image_path(video_path)

    if input_is_image:
        frame0 = cv2.imread(str(video_path))
        if frame0 is None:
            raise ValueError(f"Could not read image: {video_path}")
        height, width = frame0.shape[:2]
        fps = 1.0
        total_frames = 1
        reader = None
    else:
        reader = _PyAvVideoReader(video_path)
        fps = reader.fps
        width = reader.width
        height = reader.height
        total_frames = reader.total_frames

    # Apply upscale to dimensions
    out_width = width * upscale
    out_height = height * upscale

    vid_name = (video_id or video_path.stem).strip() if isinstance(video_id, str) else video_path.stem
    video_name = vid_name

    paths = scene_paths(output_dir)
    sidecars_dir = paths.sidecars_dir

    rgb_dir = sidecars_dir / "rgb"
    if save_rgb:
        rgb_dir.mkdir(parents=True, exist_ok=True)

    vis_detection_dir = None
    vis_tracking_dir = None
    if save_vis:
        vis_detection_dir = sidecars_dir / "vis_detection"
        vis_detection_dir.mkdir(parents=True, exist_ok=True)
        vis_tracking_dir = sidecars_dir / "vis_tracking"
        vis_tracking_dir.mkdir(parents=True, exist_ok=True)

    if copy_video:
        video_dest = sidecars_dir / video_path.name
        if not video_dest.exists():
            sidecars_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(video_path, video_dest)

    # Image inputs produce a single-frame PNG per overlay; video inputs produce an mp4.
    # Same suffix is reused for tracking / tracking_red_id below so image runs don't
    # end up with confusingly-named 1-frame .mp4 files.
    video_out_detection: _PyAvVideoWriter | None = None
    overlay_suffix = "png" if input_is_image else "mp4"
    video_out_detection_path = sidecars_dir / f"{vid_name}_detection.{overlay_suffix}"
    if save_video:
        video_out_detection_path.parent.mkdir(parents=True, exist_ok=True)
        if not input_is_image:
            video_out_detection = _PyAvVideoWriter(
                video_out_detection_path, fps=float(fps), width=out_width, height=out_height
            )

    # Data structures for JSON outputs
    instances_data: dict[str, dict[str, Any]] = {}  # object_id -> instance info
    frames_data: dict[str, dict[str, Any]] = {}  # frame_id -> frame info

    # Track instance_id counter per class
    class_instance_counters: dict[int, int] = {}

    class_ids_set = set(int(x) for x in class_ids_filter) if class_ids_filter is not None else None
    prev_kept_boxes_by_class: dict[int, np.ndarray] = {}

    # Output-level segmentation is disabled; we keep object_id as <class>_<track_id>
    track_last_bbox_by_seg: dict[tuple[int, int, int], list[float]] = {}  # (class_id, track_id, seg) -> bbox_xyxy

    _log = logger or logging.getLogger(__name__)
    frame_id = 0
    _log.info("Processing: %s (%d frames)", video_path.name, total_frames)

    if input_is_image:
        frames_iter = [frame0]
    else:
        if reader is None:
            raise RuntimeError(
                f"Failed to open video for reading: {video_path}. Ensure the file exists and is a valid video."
            )
        frames_iter = reader.iter_bgr_frames()

    try:
        for frame in frames_iter:
            if frame is None:
                break

            # Convert BGR → RGB for RFDETR
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            # Upscale if needed
            if upscale > 1:
                new_size = (pil_img.width * upscale, pil_img.height * upscale)
                pil_img = pil_img.resize(new_size)
                frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # Save RGB frame
            frame_name = f"{frame_id:06d}"
            rgb_path = rgb_dir / f"{frame_name}.jpg"
            if save_rgb:
                cv2.imwrite(str(rgb_path), frame)

            # Detection
            detections = model.predict(pil_img, threshold=detection_threshold)

            det_xyxy = detections.xyxy
            det_conf = detections.confidence
            det_cls = detections.class_id.astype(int)

            # Optional class filter
            if class_ids_set is not None and len(det_xyxy) > 0:
                mask = np.isin(det_cls, list(class_ids_set))
                det_xyxy = det_xyxy[mask]
                det_conf = det_conf[mask]
                det_cls = det_cls[mask]

            # Optional per-frame de-duplication (very high IoU)
            if dedup_iou_threshold > 0 and len(det_xyxy) > 0:
                det_xyxy, det_conf, det_cls = nms_dedup_by_iou(
                    boxes_xyxy=det_xyxy,
                    scores=det_conf,
                    class_ids=det_cls,
                    iou_threshold=dedup_iou_threshold,
                    priority=dedup_priority,
                    prev_boxes_by_class=prev_kept_boxes_by_class if dedup_priority == "prev_iou" else None,
                )

            # Optional cross-class cleanup: if different classes overlap with IoU > threshold, drop the lower one.
            if cross_class_iou_threshold > 0 and len(det_xyxy) > 0:
                det_xyxy, det_conf, det_cls = drop_cross_class_overlaps_keep_upper(
                    boxes_xyxy=det_xyxy,
                    scores=det_conf,
                    class_ids=det_cls,
                    iou_threshold=cross_class_iou_threshold,
                )

            # Update prev-frame kept boxes for temporal dedup (after filtering/dedup)
            if len(det_xyxy) > 0:
                prev_kept_boxes_by_class = {}
                for cid in np.unique(det_cls.astype(int)):
                    sel = det_cls.astype(int) == int(cid)
                    prev_kept_boxes_by_class[int(cid)] = det_xyxy[sel]
            else:
                prev_kept_boxes_by_class = {}

            # Convert to tracker expected format: (x1, y1, x2, y2, conf, cls)
            if len(det_xyxy) > 0:
                dets = np.column_stack((det_xyxy, det_conf, det_cls))
            else:
                dets = np.empty((0, 6))

            # Tracking
            tracks = tracker.update(dets, frame)

            # Build frame data
            frame_key = f"{video_name}_frame_{frame_id:06d}"
            frame_instances = []

            for track in tracks:
                # track format: [x1, y1, x2, y2, track_id, confidence, class_id]
                track_id = int(track[4])
                bbox_tight = [float(track[0]), float(track[1]), float(track[2]), float(track[3])]
                confidence = float(track[5]) if len(track) > 5 else 1.0
                class_id = int(track[6]) if len(track) > 6 else 0

                # Get class name
                class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else "unknown"

                # Keep a single segment per (class_id, track_id)
                seg = 0
                track_last_bbox_by_seg[(class_id, track_id, seg)] = bbox_tight

                # Create unique object_id
                object_id = f"{class_name}_{track_id}"

                # Calculate loose bbox (expanded)
                bbox_loose = expand_bbox(bbox_tight, bbox_expansion_ratio, out_width, out_height)

                # Add to instances if not exists
                if object_id not in instances_data:
                    if class_id not in class_instance_counters:
                        class_instance_counters[class_id] = 0
                    class_instance_counters[class_id] += 1
                    instance_id = class_instance_counters[class_id]

                    instances_data[object_id] = {
                        "object_type": class_name,
                        "instance_id": instance_id,
                        "semantic_id": class_id,
                        "color": generate_distinct_color(track_id),
                        "caption": f"{class_name} (track {track_id})",
                        "track_id": track_id,
                        "first_frame": frame_id,
                        "last_frame": frame_id,
                        "confidence_avg": confidence,
                        "frame_count": 1,
                    }
                else:
                    instances_data[object_id]["last_frame"] = frame_id
                    instances_data[object_id]["frame_count"] += 1
                    n = instances_data[object_id]["frame_count"]
                    old_avg = instances_data[object_id]["confidence_avg"]
                    instances_data[object_id]["confidence_avg"] = old_avg + (confidence - old_avg) / n

                frame_instances.append(
                    {
                        "object_id": object_id,
                        "instance_id": instances_data[object_id]["instance_id"],
                        "semantic_id": class_id,
                        "bounding_box_2d_tight": [round(x, 2) for x in bbox_tight],
                        "bounding_box_2d_loose": [round(x, 2) for x in bbox_loose],
                        "confidence": round(confidence, 4),
                    }
                )

            # Store frame data
            frames_data[frame_key] = {
                "format": "png",
                "frame_number": frame_id,
                "width": out_width,
                "height": out_height,
                "instances": frame_instances,
                "detection_count": len(det_xyxy) if len(det_xyxy) > 0 else 0,
            }

            # Save visualization if requested
            if save_vis or save_video:
                det_img = draw_detections(
                    image=frame,
                    bboxes=det_xyxy if len(det_xyxy) > 0 else np.empty((0, 4)),
                    confidences=det_conf if len(det_xyxy) > 0 else np.empty((0,)),
                    class_ids=det_cls if len(det_xyxy) > 0 else np.empty((0,)),
                    class_names=COCO_CLASSES,
                )

                if save_vis and vis_detection_dir is not None:
                    cv2.imwrite(str(vis_detection_dir / f"{frame_name}.jpg"), det_img)

                if save_video and video_out_detection is not None:
                    video_out_detection.write(det_img)
                elif save_video and input_is_image:
                    try:
                        ok = cv2.imwrite(str(video_out_detection_path), det_img)
                        if not ok:
                            raise OSError(f"cv2.imwrite returned False for {video_out_detection_path}")
                    except Exception as _e:
                        _log.warning("Failed to write detection image (%s): %s", video_out_detection_path, _e)

                tracked_img = draw_tracks(
                    image=frame,
                    instances=frame_instances,
                    instances_info=instances_data,
                    class_names=COCO_CLASSES,
                )

                if save_vis and vis_tracking_dir is not None:
                    cv2.imwrite(str(vis_tracking_dir / f"{frame_name}.jpg"), tracked_img)

            frame_id += 1

            if frame_id % 100 == 0:
                _log.info("  Processed %d/%d frames...", frame_id, total_frames)
    finally:
        if reader is not None:
            reader.close()
        if video_out_detection is not None:
            video_out_detection.close()

    # Filter instances by minimum frame count
    if min_track_frames > 1:
        # Get object_ids to remove
        filtered_object_ids = {
            obj_id for obj_id, info in instances_data.items() if info["frame_count"] < min_track_frames
        }

        # Remove filtered instances
        instances_data = {obj_id: info for obj_id, info in instances_data.items() if obj_id not in filtered_object_ids}

        # Remove filtered instances from frames_data
        for frame_key in frames_data:
            frames_data[frame_key]["instances"] = [
                inst for inst in frames_data[frame_key]["instances"] if inst["object_id"] not in filtered_object_ids
            ]

        filtered_count = len(filtered_object_ids)
        if filtered_count > 0:
            _log.info("  ✓ Filtered %d tracks with < %d frames", filtered_count, min_track_frames)

    tracking_video_path = sidecars_dir / f"{video_name}_tracking.{overlay_suffix}"
    if save_video:
        _log.info("  Generating tracking video (after filtering)...")
        tracking_video_path.parent.mkdir(parents=True, exist_ok=True)
        generate_filtered_video(
            video_path=video_path,
            output_path=tracking_video_path,
            frames_data=frames_data,
            instances_data=instances_data,
            class_names=COCO_CLASSES,
            video_name=video_name,
            upscale=upscale,
        )
        _log.info("  ✓ Tracking video saved to: %s", tracking_video_path)

    # Optional: extra tracking video variant (uniform red box + red id, no labels)
    if save_video_red_id:
        tracking_video_path_red_id = sidecars_dir / f"{video_name}_tracking_red_id.{overlay_suffix}"
        _log.info("  Generating tracking red-id video (after filtering)...")
        tracking_video_path_red_id.parent.mkdir(parents=True, exist_ok=True)
        generate_filtered_video(
            video_path=video_path,
            output_path=tracking_video_path_red_id,
            frames_data=frames_data,
            instances_data=instances_data,
            class_names=COCO_CLASSES,
            video_name=video_name,
            upscale=upscale,
            track_vis_style="red_id",
        )
        _log.info("  ✓ Tracking red-id video saved to: %s", tracking_video_path_red_id)

    # Internal (auto-labeling-native) in-memory structures. These are the input to the
    # DAFT converters below; non-DAFT fields (video_info, version) are
    # stripped by ``to_daft_instances`` / ``to_daft_objects``.
    instances_json = {
        "video_info": {
            "source": str(video_path),
            "fps": fps,
            "width": out_width,
            "height": out_height,
            "total_frames": frame_id,
        },
        "instances": instances_data,
    }
    objects_json = {"frames": frames_data}

    if write_json:
        paths.contextual_dir.mkdir(parents=True, exist_ok=True)
        write_daft_json(paths.contextual_instances, to_daft_instances(instances_json))
        write_daft_json(paths.contextual_objects, to_daft_objects(objects_json, video_id=get_scene_media_id()))

        _log.info("  ✓ Processed %d frames", frame_id)
        _log.info("  ✓ Found %d unique tracked objects", len(instances_data))
        if save_rgb:
            _log.info("  ✓ RGB frames saved to: %s", rgb_dir)
        _log.info("  ✓ instances.json saved to: %s", paths.contextual_instances)
        _log.info("  ✓ objects.json saved to: %s", paths.contextual_objects)

        if save_vis:
            _log.info("  ✓ Detection frames saved to: %s", vis_detection_dir)
            _log.info("  ✓ Tracking frames saved to: %s", vis_tracking_dir)
        if save_video:
            _log.info("  ✓ Detection overlay saved to: %s", video_out_detection_path)
            _log.info("  ✓ Tracking video saved to: %s", tracking_video_path)

    return {
        "video": str(video_path),
        "status": "success",
        "frames_processed": frame_id,
        "unique_tracks": len(instances_data),
        "scene_dir": str(output_dir),
        "instances_data": instances_data,
        "frames_data": frames_data,
        "video_info": instances_json["video_info"],
    }


class RFDetrTracker(BaseTracker):
    """RF-DETR model loaded once; tracker re-created per video (stateful).

    This is the pipeline-facing tracker stage object. It reuses the lower-level
    `process_video()` implementation in this module, but avoids subprocess calls.
    """

    def __init__(self, config: PipelineConfig, logger: logging.Logger, *, gpu_list: list[int]) -> None:
        super().__init__(logger)

        dt: DetectionAndTrackingConfig = config.detection_and_tracking

        # Resolve ckpts root consistently with SR.
        repo_root = Path(__file__).resolve().parent.parent.parent
        ckpts_root = resolve_ckpts_root(repo_root=repo_root, model_cache_path=config.pipeline.model_cache_path)

        # ---- Resolve class filter ----
        classes_raw = dt.classes
        if isinstance(classes_raw, (list, tuple)):
            class_names = [str(c) for c in classes_raw]
        elif isinstance(classes_raw, str):
            class_names = [c.strip() for c in classes_raw.split() if c.strip()]
        else:
            class_names = []
        self._class_ids_filter = resolve_class_ids(None, class_names) if class_names else None

        # ---- Resolve ReID weights ----
        reid_weights: Optional[Path] = None
        self._use_reid = bool(dt.use_reid)
        if self._use_reid:
            raw = str(dt.reid_weights or "").strip()
            if raw.lower() in {"", "none", "null"}:
                raw = ""
            if raw:
                reid_weights = Path(raw)
                if not reid_weights.exists():
                    raise ValueError(f"[tracking] ReID weights not found (use_reid=true): {reid_weights}")
            else:
                reid_weights = ckpts_root / "reid" / "clip_vehicleid.pt"
                ensure_reid_weights(reid_weights, logger=logger)

        self._reid_weights = reid_weights

        # ---- Load RF-DETR model ONCE ----
        pretrain_name = "rf-detr-base.pth"
        pretrain_path = ckpts_root / "rfdetr" / pretrain_name

        if pretrain_name in RFDETR_PRETRAIN_URLS:
            older_candidates = [
                Path(pretrain_name),
                Path("downloads") / "rfdetr" / pretrain_name,
            ]
            for older in older_candidates:
                if older.exists() and older.is_file() and (not pretrain_path.exists()):
                    pretrain_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        older.replace(pretrain_path)
                    except Exception as e:
                        logger.warning(
                            "Older checkpoint migration failed: %s -> %s (%s); will attempt fresh download from %s",
                            older,
                            pretrain_path,
                            e,
                            RFDETR_PRETRAIN_URLS.get(pretrain_name, "<unknown>"),
                            exc_info=True,
                        )
                    break
            ensure_url_downloaded(url=RFDETR_PRETRAIN_URLS[pretrain_name], dst=pretrain_path)

        if torch.cuda.is_available():
            if not gpu_list:
                raise ValueError("[tracking] gpu_list is empty; at least one GPU index is required")
            try:
                first_gpu = int(gpu_list[0])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"[tracking] gpu_list[0] must be an int, got {gpu_list[0]!r}") from exc
            self._device = f"cuda:{first_gpu}"
            # RFDETRBase only accepts 'cuda'/'cpu'/'mps' (not 'cuda:N'); pin via set_device.
            torch.cuda.set_device(first_gpu)
            rfdetr_device = "cuda"
            logger.info("[tracking] GPU routing: using physical CUDA device %s directly as %s", first_gpu, self._device)
        else:
            self._device = "cpu"
            rfdetr_device = "cpu"
            if gpu_list:
                logger.warning(
                    "CUDA is not available; ignoring gpu_list=%r and falling back to CPU.",
                    gpu_list,
                )

        logger.info("Loading RF-DETR model from %s (device=%s) ...", pretrain_path, self._device)
        self._model = RFDETRBase(
            device=rfdetr_device,
            pretrain_weights=str(pretrain_path),
        )
        logger.info("RF-DETR model loaded successfully")

        # ---- Store tracker/processing config ----
        self._tracker_name = str(dt.tracker)
        self._threshold = float(dt.threshold)
        self._iou_threshold = float(dt.iou_threshold)
        self._per_class = bool(dt.per_class)
        self._asso_func = str(dt.asso_func)
        self._min_hits = int(dt.min_hits)
        self._max_age = int(dt.max_age)
        self._min_track_frames = int(dt.min_track_frames)
        self._deepocsort_stage2_off = bool(dt.deepocsort_stage2_off)
        self._deepocsort_min_hits_nonconsecutive = bool(dt.deepocsort_min_hits_nonconsecutive)
        self._save_vis = bool(dt.save_vis)
        self._save_video = bool(dt.save_video)
        self._save_video_red_id = bool(dt.save_video_red_id)
        self._save_rgb = bool(dt.save_rgb)
        self._copy_video = bool(dt.copy_video)
        self._cross_class_iou_threshold = float(dt.cross_class_iou_threshold)
        self._dedup_iou_threshold = float(dt.dedup_iou_threshold)
        self._dedup_priority = str(dt.dedup_priority)

    def run(self, video_path: Path, output_dir: Path) -> TrackingResult:
        # Fresh tracker for each video (stateful — must be reset between videos).
        tracker = get_tracker(
            tracker_name=self._tracker_name,
            reid_weights=self._reid_weights if self._use_reid else None,
            per_class=self._per_class,
            iou_threshold=self._iou_threshold,
            det_thresh=self._threshold,
            asso_func=self._asso_func,
            min_hits=self._min_hits,
            max_age=self._max_age,
            deepocsort_stage2_off=self._deepocsort_stage2_off,
            deepocsort_min_hits_nonconsecutive=self._deepocsort_min_hits_nonconsecutive,
            device=self._device,
        )

        scene_dir = Path(output_dir)
        process_video(
            video_path=Path(video_path),
            output_dir=scene_dir,
            model=self._model,
            tracker=tracker,
            detection_threshold=self._threshold,
            class_ids_filter=self._class_ids_filter,
            save_vis=self._save_vis,
            save_video=self._save_video,
            save_video_red_id=self._save_video_red_id,
            min_track_frames=self._min_track_frames,
            save_rgb=self._save_rgb,
            copy_video=self._copy_video,
            cross_class_iou_threshold=self._cross_class_iou_threshold,
            dedup_iou_threshold=self._dedup_iou_threshold,
            dedup_priority=self._dedup_priority,
            logger=self.logger,
        )

        paths = scene_paths(scene_dir)
        red_id: Optional[Path] = None
        if self._save_video_red_id:
            overlay_suffix = "png" if is_image_path(Path(video_path)) else "mp4"
            candidate = paths.sidecars_dir / f"{Path(video_path).stem}_tracking_red_id.{overlay_suffix}"
            if candidate.exists():
                red_id = candidate
        return TrackingResult(
            success=True,
            instances_json=paths.contextual_instances,
            objects_json=paths.contextual_objects,
            tracking_video_red_id=red_id,
        )


def ensure_reid_weights(path: Path, *, logger: logging.Logger) -> None:
    """Ensure Vehicle CLIP ReID weights exist at the given path."""
    p = Path(path).expanduser().resolve()
    if p.exists() and p.stat().st_size > 0:
        return

    gdrive_id = str(os.getenv("REID_CLIP_GDRIVE_ID", "")).strip() or "168BLegHHxNqatW5wx1YyL2REaThWoof5"
    url = f"https://drive.usercontent.google.com/download?id={gdrive_id}&export=download&confirm=t"
    sha256 = str(os.getenv("REID_CLIP_SHA256", "")).strip().lower() or None

    logger.info("[tracking] downloading ReID weights -> %s", p)
    ensure_url_downloaded(url=url, dst=p, timeout_s=1200, sha256=sha256, min_bytes=5 * 1024 * 1024)
