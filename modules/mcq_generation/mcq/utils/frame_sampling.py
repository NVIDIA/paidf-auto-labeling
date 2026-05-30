# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import ffmpeg  # type: ignore
from al_utils.media_decode import h264_stream_requires_unsupported_decoder


def sample_frames_ffmpeg(
    *,
    video_path: Path,
    out_dir: Path,
    start_sec: float,
    end_sec: float,
    sampling_fps: float,
    resolution: int,
    max_frames: int,
    logger: logging.Logger,
    qscale: int = 4,
) -> List[Tuple[float, Path]]:
    """
    Shared frame sampling core for VLM/MCQ stages.

    Returns a list of (timestamp_sec, frame_path).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    _raise_if_h264_requires_unsupported_decoder(video_path)

    dur = max(0.001, float(end_sec) - float(start_sec))
    sfps = float(sampling_fps)
    if sfps <= 0:
        sfps = 1.0

    if max_frames > 0:
        sfps = min(sfps, max_frames / dur)
        sfps = max(sfps, 0.1)

    if dur * sfps < 1.0:
        sfps = max(sfps, min(30.0, 1.0 / dur))

    def _clear_existing_frames() -> None:
        try:
            for p in out_dir.glob("frame_*.jpg"):
                p.unlink(missing_ok=True)
        except Exception:
            # Best-effort cleanup; extraction may still succeed.
            pass

    def _list_frames() -> List[Path]:
        return sorted([p for p in out_dir.glob("frame_*.jpg") if p.is_file()])

    def _stderr_text(e: Exception) -> str:
        err = getattr(e, "stderr", b"") or b""
        try:
            return err.decode("utf-8", errors="replace")
        except Exception:
            return repr(err)

    def _should_fallback(err_txt: str) -> bool:
        t = (err_txt or "").lower()
        return (
            ("qscale is ambiguous" in t)
            or ("non full-range yuv" in t)
            or ("strict_std_compliance" in t)
            or ("[mjpeg" in t and "error while opening encoder" in t)
            or ("mjpeg" in t and "invalid argument" in t and "nothing was written" in t)
            or ("ff_frame_thread_encoder_init failed" in t)
            or ("could not open encoder" in t and "mjpeg" in t)
        )

    def _run(*, use_fallback: bool) -> None:
        stream = (
            ffmpeg.input(str(video_path), ss=float(start_sec), t=float(dur), fflags="+genpts")
            .filter("fps", fps=sfps)
            .filter("scale", -2, int(resolution))
        )
        if use_fallback:
            stream = stream.filter("format", "yuvj420p")
            out_kwargs = {"q:v": int(qscale), "pix_fmt": "yuvj420p", "strict": "-2"}
        else:
            out_kwargs = {"q:v": int(qscale)}

        (stream.output(str(out_dir / "frame_%06d.jpg"), **out_kwargs).overwrite_output().run(quiet=True))

    try:
        _clear_existing_frames()
        _run(use_fallback=False)
    except ffmpeg.Error as e:  # type: ignore[attr-defined]
        err_txt = _stderr_text(e)
        if _should_fallback(err_txt):
            logger.warning(
                "FFmpeg failed extracting frames; retrying with fallback (strict=-2, pix_fmt=yuvj420p). "
                "video=%s window=%s-%s",
                video_path,
                start_sec,
                end_sec,
            )
            _clear_existing_frames()
            try:
                _run(use_fallback=True)
            except ffmpeg.Error as e2:  # type: ignore[attr-defined]
                err2 = _stderr_text(e2)
                logger.error(
                    "FFmpeg failed extracting frames (video=%s window=%s-%s). Stderr tail:\n%s",
                    video_path,
                    start_sec,
                    end_sec,
                    (err2 or err_txt)[-2000:],
                )
                return []
        else:
            logger.error(
                "FFmpeg failed extracting frames (video=%s window=%s-%s). Stderr tail:\n%s",
                video_path,
                start_sec,
                end_sec,
                err_txt[-2000:],
            )
            return []

    frames = _list_frames()
    if not frames:
        logger.warning(
            "No frames extracted (video=%s window=%s-%s); attempting single-frame fallback",
            video_path,
            start_sec,
            end_sec,
        )
        try:
            _clear_existing_frames()
            (
                ffmpeg.input(str(video_path), ss=float(start_sec), t=float(dur), fflags="+genpts")
                .filter("scale", -2, int(resolution))
                .output(str(out_dir / "frame_%06d.jpg"), **{"q:v": int(qscale), "vframes": 1})
                .overwrite_output()
                .run(quiet=True)
            )
        except Exception:
            pass
        frames = _list_frames()
        if not frames:
            logger.warning(
                "No frames extracted after single-frame fallback (video=%s window=%s-%s)",
                video_path,
                start_sec,
                end_sec,
            )
            return []

    total_extracted = len(frames)

    # Decide which extracted indices to keep.
    if max_frames > 0:
        k = min(int(max_frames), total_extracted)
    else:
        k = total_extracted

    if k <= 1 or total_extracted <= 1:
        return [(round(float(start_sec), 3), frames[0])]

    if k == total_extracted:
        idxs = list(range(total_extracted))
    else:
        # Evenly subsample across the full extracted range (always includes first+last).
        last = total_extracted - 1
        idxs = [(i * last) // (k - 1) for i in range(k)]

    out: List[Tuple[float, Path]] = []
    last_f = float(total_extracted - 1)
    for idx in idxs:
        ts = float(start_sec) + (dur * (float(idx) / last_f))
        out.append((round(ts, 3), frames[int(idx)]))
    return out


def _raise_if_h264_requires_unsupported_decoder(video_path: Path) -> None:
    try:
        info = ffmpeg.probe(str(video_path))
    except Exception:
        # Let the normal extraction path produce the detailed ffmpeg error.
        return

    stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    if not stream or not h264_stream_requires_unsupported_decoder(stream):
        return

    codec = str(stream.get("codec_name") or "h264")
    profile = str(stream.get("profile") or "unknown")
    pix_fmt = str(stream.get("pix_fmt") or "unknown")
    raise RuntimeError(
        "h264_cuvid CUDA_ERROR_NOT_SUPPORTED: unsupported H.264 input for this image "
        f"(codec={codec}, profile={profile}, pix_fmt={pix_fmt}). "
        "Only H.264 4:2:0-compatible inputs are supported by the available H.264 hardware decoder."
    )
