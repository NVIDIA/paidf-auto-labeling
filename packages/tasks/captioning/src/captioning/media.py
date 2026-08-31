# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Media probing and frame sampling helpers for captioning."""

from __future__ import annotations

import base64
import importlib
import math
import mimetypes
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core import is_image_path
from core.media import (
    UnsupportedVideoCodecError,
    VideoDecodeError,
    VideoDecodePlan,
    VideoDecoderUnavailableError,
    Vp9VideoWriter,
    iter_decoded_rgb24_frames,
    prepare_video_decode,
)

from captioning.clients import MediaPayload
from captioning.config import CaptioningConfig


class MediaDecodeError(RuntimeError):
    """Raised when media cannot be decoded for captioning."""


@dataclass(frozen=True)
class VideoInfo:
    """Small video metadata record."""

    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        """Video duration in seconds."""
        if self.fps <= 0:
            return 0.0
        return self.frame_count / self.fps


@dataclass(frozen=True)
class ImageInfo:
    """Small image metadata record."""

    width: int
    height: int
    mime_type: str
    num_bytes: int


@dataclass(frozen=True)
class CaptionWindow:
    """Window bounds used for video dense captioning."""

    index: int
    start_s: float
    end_s: float
    start_frame: int
    end_frame: int


WINDOW_MIN_FRAMES = 4


def read_image_payload(path: Path) -> MediaPayload:
    """Read an image into a base64 payload."""
    data = path.read_bytes()
    image_info = probe_image(path)
    return MediaPayload(
        mime_type=image_info.mime_type,
        data_base64=base64.b64encode(data).decode("ascii"),
        filename=path.name,
        width=image_info.width,
        height=image_info.height,
        num_bytes=image_info.num_bytes,
    )


def probe_image(path: Path) -> ImageInfo:
    """Probe image metadata without failing captioning on undecodable fixtures."""
    data = path.read_bytes()
    mime_type = _mime_type(path, default="image/jpeg")
    width = 0
    height = 0
    try:
        cv2 = _import_cv2()
        import numpy as np  # noqa: PLC0415

        decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is not None:
            height, width = decoded.shape[:2]
    except MediaDecodeError:
        pass
    return ImageInfo(width=int(width), height=int(height), mime_type=mime_type, num_bytes=len(data))


def probe_video(path: Path) -> VideoInfo:
    """Validate the input codec and probe metadata with the shared media implementation."""
    plan = _prepare_video_decode(path)
    stream = plan.stream
    fps = stream.fps or 30.0
    frame_count = stream.frame_count
    if frame_count is None and stream.duration_seconds is not None:
        frame_count = round(stream.duration_seconds * fps)
    if frame_count is None:
        try:
            frame_count = sum(1 for _frame in iter_decoded_rgb24_frames(path, plan))
        except VideoDecodeError as e:
            raise MediaDecodeError(
                f"Failed to count decoded frames for {path} using plan {plan}"
            ) from e
    return VideoInfo(
        fps=fps,
        frame_count=frame_count,
        width=stream.width,
        height=stream.height,
    )


def plan_windows(info: VideoInfo, config: CaptioningConfig) -> list[CaptionWindow]:
    """Plan caption windows for a video."""
    if info.frame_count <= 0:
        return []
    if config.single_window:
        return [
            CaptionWindow(
                index=0,
                start_s=0.0,
                end_s=info.duration_s,
                start_frame=0,
                end_frame=max(0, info.frame_count - 1),
            )
        ]
    window_frames = int(config.window_frames)
    if window_frames > 0:
        return _plan_frame_windows(
            info,
            window_frames=window_frames,
            remainder_threshold=int(config.remainder_threshold),
        )
    if config.window_seconds <= 0:
        return []
    return _plan_second_windows(info, config.window_seconds)


def extract_window_video_payload(
    *,
    media_path: Path,
    window: CaptionWindow,
    total_frames: int,
) -> MediaPayload:
    """Extract one MP4 payload for ``window`` using inclusive frame bounds."""
    if (
        media_path.suffix.lower() == ".mp4"
        and window.start_frame <= 0
        and window.end_frame >= total_frames - 1
    ):
        data = media_path.read_bytes()
        filename = media_path.name
    else:
        data, filename = _extract_window_mp4(media_path=media_path, window=window)

    return MediaPayload(
        mime_type="video/mp4",
        data_base64=base64.b64encode(data).decode("ascii"),
        filename=filename,
        frame_index=window.start_frame,
        time_s=window.start_s,
        num_bytes=len(data),
    )


def extract_window_frames(
    *,
    media_path: Path,
    window: CaptionWindow,
    source_fps: float,
    sampling_fps: float,
    max_frames: int,
    resolution: int,
) -> list[MediaPayload]:
    """Extract a bounded set of JPEG frame payloads for ``window``."""
    if sampling_fps <= 0:
        raise MediaDecodeError(f"sampling_fps must be greater than 0, got {sampling_fps}")
    frame_ids = _sample_frame_ids(
        start_frame=window.start_frame,
        end_frame=window.end_frame,
        source_fps=source_fps,
        sampling_fps=sampling_fps,
        max_frames=max_frames,
    )
    plan = _prepare_video_decode(media_path)
    cv2 = _import_cv2()
    import numpy as np  # noqa: PLC0415

    selected_ids = set(frame_ids)
    last_frame_id = frame_ids[-1]
    shape = (plan.stream.height, plan.stream.width, 3)
    out: list[MediaPayload] = []
    try:
        for frame_id, payload in enumerate(iter_decoded_rgb24_frames(media_path, plan)):
            if frame_id > last_frame_id:
                break
            if frame_id not in selected_ids:
                continue
            frame_rgb = np.frombuffer(payload, dtype=np.uint8).reshape(shape)
            resized_rgb = _resize_frame(cv2, frame_rgb, resolution)
            frame_bgr = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)
            success, encoded = cv2.imencode(".jpg", frame_bgr)
            if not success:
                continue
            height, width = resized_rgb.shape[:2]
            frame_bytes = bytes(encoded)
            data = base64.b64encode(frame_bytes).decode("ascii")
            out.append(
                MediaPayload(
                    mime_type="image/jpeg",
                    data_base64=data,
                    filename=f"{media_path.stem}_{frame_id:06d}.jpg",
                    frame_index=frame_id,
                    time_s=round(frame_id / source_fps, 3) if source_fps > 0 else None,
                    width=int(width),
                    height=int(height),
                    num_bytes=len(frame_bytes),
                )
            )
    except VideoDecodeError as e:
        raise MediaDecodeError(
            f"Failed to decode sampled frames from {media_path} using plan {plan}"
        ) from e
    if not out:
        raise MediaDecodeError(f"No frames decoded from video window in {media_path}")
    return out


def _plan_frame_windows(
    info: VideoInfo,
    *,
    window_frames: int,
    remainder_threshold: int,
) -> list[CaptionWindow]:
    if info.frame_count < WINDOW_MIN_FRAMES:
        return []
    frame_ranges = _compute_frame_windows(
        total_frames=info.frame_count,
        window_size=window_frames,
        remainder_threshold=remainder_threshold,
    )
    return [
        CaptionWindow(
            index=index,
            start_s=start / info.fps if info.fps > 0 else 0.0,
            end_s=(end + 1) / info.fps if info.fps > 0 else 0.0,
            start_frame=start,
            end_frame=end,
        )
        for index, (start, end) in enumerate(frame_ranges)
    ]


def _compute_frame_windows(
    *,
    total_frames: int,
    window_size: int,
    remainder_threshold: int,
) -> list[tuple[int, int]]:
    if total_frames <= 0 or total_frames < WINDOW_MIN_FRAMES:
        return []
    if window_size <= 0 or total_frames <= window_size:
        return [(0, total_frames - 1)]

    num_full_windows = total_frames // window_size
    remainder = total_frames % window_size
    out = [
        (index * window_size, index * window_size + window_size - 1)
        for index in range(num_full_windows)
    ]

    if remainder > 0 and remainder >= remainder_threshold:
        out.append((total_frames - remainder, total_frames - 1))
    elif remainder > 0 and out:
        start, _end = out[-1]
        out[-1] = (start, total_frames - 1)
    return out


def _plan_second_windows(info: VideoInfo, window_seconds: float) -> list[CaptionWindow]:
    if info.fps <= 0 or window_seconds <= 0:
        return []
    duration = max(info.duration_s, 0.0)
    if duration <= 0:
        return []
    windows: list[CaptionWindow] = []
    start_s = 0.0
    index = 0
    while start_s < duration:
        end_s = min(duration, start_s + window_seconds)
        start_frame = min(info.frame_count - 1, max(0, math.floor(start_s * info.fps)))
        end_frame = min(
            info.frame_count - 1,
            max(start_frame, math.ceil(end_s * info.fps) - 1),
        )
        windows.append(
            CaptionWindow(
                index=index,
                start_s=start_s,
                end_s=end_s,
                start_frame=start_frame,
                end_frame=end_frame,
            )
        )
        if end_s >= duration:
            break
        start_s = end_s
        index += 1
    return windows


def _sample_frame_ids(
    *,
    start_frame: int,
    end_frame: int,
    source_fps: float,
    sampling_fps: float,
    max_frames: int,
) -> list[int]:
    if sampling_fps <= 0:
        raise MediaDecodeError(f"sampling_fps must be greater than 0, got {sampling_fps}")
    if max_frames <= 0:
        raise MediaDecodeError(f"max_frames must be greater than 0, got {max_frames}")
    step = max(1, round(source_fps / sampling_fps)) if source_fps > 0 else 1
    frame_ids = list(range(start_frame, end_frame + 1, step))
    if not frame_ids:
        frame_ids = [start_frame]
    if len(frame_ids) <= max_frames:
        return frame_ids
    stride = (len(frame_ids) - 1) / max(1, max_frames - 1)
    selected = [frame_ids[round(i * stride)] for i in range(max_frames)]
    return sorted(set(selected))


def _extract_window_mp4(*, media_path: Path, window: CaptionWindow) -> tuple[bytes, str]:
    if not media_path.exists() or not media_path.is_file():
        raise ValueError(f"media_path must be an existing file: {media_path}")
    filename = f"{media_path.stem}_window_{window.index:03d}.mp4"
    with tempfile.TemporaryDirectory(prefix="captioning-window-") as tmp_dir:
        out_path = Path(tmp_dir) / filename
        plan = _prepare_video_decode(media_path)
        import numpy as np  # noqa: PLC0415

        shape = (plan.stream.height, plan.stream.width, 3)
        encoded_frames = 0
        with Vp9VideoWriter(
            out_path,
            width=plan.stream.width,
            height=plan.stream.height,
            fps=plan.stream.fps or 30.0,
            frame_format="rgb24",
        ) as writer:
            try:
                for frame_id, payload in enumerate(iter_decoded_rgb24_frames(media_path, plan)):
                    if frame_id > window.end_frame:
                        break
                    if frame_id < window.start_frame:
                        continue
                    writer.write(np.frombuffer(payload, dtype=np.uint8).reshape(shape))
                    encoded_frames += 1
            except VideoDecodeError as e:
                raise MediaDecodeError(
                    f"Failed to decode window from {media_path} using plan {plan}"
                ) from e
        if encoded_frames == 0:
            raise MediaDecodeError(
                f"No frames decoded from video window {window.start_frame}-{window.end_frame}"
            )
        return out_path.read_bytes(), filename


def _prepare_video_decode(path: Path) -> VideoDecodePlan:
    try:
        return prepare_video_decode(path)
    except (UnsupportedVideoCodecError, VideoDecoderUnavailableError, VideoDecodeError) as exc:
        raise MediaDecodeError(str(exc)) from exc


def _resize_frame(cv2: Any, frame: Any, resolution: int) -> Any:
    if resolution <= 0:
        raise ValueError("resolution must be a positive integer")

    height, width = frame.shape[:2]
    longest = max(width, height)
    if longest <= resolution:
        return frame
    scale = resolution / longest
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def _mime_type(path: Path, *, default: str) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or default


def _import_cv2() -> Any:
    try:
        return importlib.import_module("cv2")
    except ImportError as exc:
        raise MediaDecodeError(
            "Image processing requires headless OpenCV. Install captioning[video] or use the "
            "captioning service container."
        ) from exc


__all__ = [
    "CaptionWindow",
    "ImageInfo",
    "MediaDecodeError",
    "VideoInfo",
    "WINDOW_MIN_FRAMES",
    "extract_window_frames",
    "extract_window_video_payload",
    "is_image_path",
    "plan_windows",
    "probe_image",
    "probe_video",
    "read_image_payload",
]
