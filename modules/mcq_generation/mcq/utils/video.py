# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import ffmpeg  # type: ignore
from al_utils.media_paths import is_image_path
from mcq_generation.mcq.utils.frame_sampling import sample_frames_ffmpeg
from PIL import Image


@dataclass
class VideoInfo:
    duration_sec: float
    fps: float
    width: int
    height: int
    num_frames: int


def _round_even(x: int) -> int:
    v = int(x)
    return v if (v % 2 == 0) else max(2, v - 1)


def _parse_ffprobe_fraction(s: object) -> float:
    if not s:
        return 0.0
    try:
        txt = str(s).strip()
        if not txt:
            return 0.0
        if "/" not in txt:
            return float(txt)
        num_s, den_s = txt.split("/", 1)
        num = float(num_s)
        den = float(den_s)
        if den == 0:
            return 0.0
        return num / den
    except Exception:
        return 0.0


def probe_video(video_path: Path) -> VideoInfo:
    """
    Probe an input clip and return normalized "video-like" info.

    This function supports:
    - videos: via ffprobe
    - images: treated as a single-frame "video" (duration=1s, fps=1, num_frames=1)
    """
    video_path = Path(video_path)
    if is_image_path(video_path):
        try:
            with Image.open(video_path) as im:
                w, h = im.size
        except Exception as e:
            raise ValueError(f"Failed to open image: {video_path} ({e.__class__.__name__})") from e
        return VideoInfo(duration_sec=1.0, fps=1.0, width=int(w), height=int(h), num_frames=1)

    probe = ffmpeg.probe(str(video_path))
    video_stream = next((s for s in probe["streams"] if s.get("codec_type") == "video"), None)
    if not video_stream:
        raise ValueError(f"No video stream found: {video_path}")

    duration = video_stream.get("duration") or probe.get("format", {}).get("duration") or 0
    duration_sec = float(duration)

    fps = _parse_ffprobe_fraction(video_stream.get("avg_frame_rate"))
    if fps <= 0 and duration_sec > 0:
        try:
            nb = float(video_stream.get("nb_frames") or 0)
            fps = nb / duration_sec if nb > 0 else 0.0
        except Exception:
            fps = 0.0
    if fps <= 0:
        fps = 30.0

    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    num_frames = int(round(duration_sec * fps))
    return VideoInfo(duration_sec=duration_sec, fps=fps, width=width, height=height, num_frames=num_frames)


def iter_windows(duration_sec: float, window_sec: float) -> Iterable[Tuple[float, float]]:
    w = float(window_sec)
    if w <= 0:
        yield 0.0, duration_sec
        return
    t = 0.0
    while t < duration_sec - 1e-6:
        end = min(duration_sec, t + w)
        yield round(t, 3), round(end, 3)
        t = end


def iter_windows_by_frames(num_frames: int, window_frames: int) -> Iterable[Tuple[int, int]]:
    nf = int(num_frames)
    wf = int(window_frames)
    if nf <= 0:
        return
    if wf <= 0:
        yield 0, nf - 1
        return
    s = 0
    while s < nf:
        e = min(nf - 1, s + wf - 1)
        yield s, e
        s = e + 1


def extract_frames(
    *,
    video_path: Path,
    out_dir: Path,
    start_sec: float,
    end_sec: float,
    sampling_fps: float,
    resolution: int,
    max_frames: int,
    logger: logging.Logger,
) -> List[Tuple[float, Path]]:
    """
    Extract frames for a time window.

    - For video inputs: uses FFmpeg extraction (returns list of (timestamp_sec, frame_path)).
    - For image inputs: writes a single JPEG frame (frame_000001.jpg) and returns [(0.0, path)].
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    if is_image_path(video_path):
        out_dir.mkdir(parents=True, exist_ok=True)
        # Best-effort cleanup to match ffmpeg extractor semantics.
        try:
            for p in out_dir.glob("frame_*.jpg"):
                p.unlink(missing_ok=True)
        except Exception:
            pass

        try:
            with Image.open(video_path) as im0:
                im = im0.convert("RGB")
                w0, h0 = im.size
                target_h = int(resolution or 0)
                if target_h > 0 and h0 > 0 and int(h0) != target_h:
                    scale = float(target_h) / float(h0)
                    target_w = _round_even(int(round(float(w0) * scale)))
                    target_h2 = _round_even(int(target_h))
                    im = im.resize((int(target_w), int(target_h2)), resample=Image.Resampling.LANCZOS)
        except Exception as e:
            logger.error("Failed extracting frame from image=%s (%s)", video_path, e, exc_info=True)
            return []

        frame_path = out_dir / "frame_000001.jpg"
        try:
            im.save(frame_path, format="JPEG", quality=95, optimize=True)
        except Exception:
            # Pillow may fail optimize on some builds; retry without it.
            try:
                im.save(frame_path, format="JPEG", quality=95)
            except Exception as e:
                logger.error("Failed writing extracted frame=%s (%s)", frame_path, e, exc_info=True)
                return []
        return [(0.0, frame_path)]

    return sample_frames_ffmpeg(
        video_path=video_path,
        out_dir=out_dir,
        start_sec=float(start_sec),
        end_sec=float(end_sec),
        sampling_fps=float(sampling_fps),
        resolution=int(resolution),
        max_frames=int(max_frames),
        logger=logger,
        qscale=4,
    )
