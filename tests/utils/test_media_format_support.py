# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests that al_utils.media_paths correctly classifies all claimed formats,
and that PIL can actually open the image types we advertise as supported.

Most tests run outside Docker and do not require GPU. Tests that call ffmpeg-python
or sample_frames_ffmpeg are skipped when ffmpeg/ffprobe binaries are absent (host);
they run inside Docker where the custom ffmpeg build is available.
"""

from __future__ import annotations

import logging
import shutil
import warnings
from pathlib import Path

import av
import ffmpeg as ffmpeg_python
import numpy as np
import pytest
from al_utils.media_paths import IMAGE_EXTS, VIDEO_EXTS, is_image_path, is_video_path
from mcq_generation.mcq.utils.frame_sampling import sample_frames_ffmpeg
from PIL import Image

_ffmpeg_available = shutil.which("ffmpeg") is not None
_ffprobe_available = shutil.which("ffprobe") is not None

# ---------------------------------------------------------------------------
# IMAGE_EXTS / VIDEO_EXTS membership
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ext", [".jpg", ".jpeg", ".png", ".webp", ".bmp"])
def test_image_exts_contains(ext: str) -> None:
    assert ext in IMAGE_EXTS


@pytest.mark.parametrize("ext", [".mp4", ".mov", ".m4v"])
def test_video_exts_contains(ext: str) -> None:
    assert ext in VIDEO_EXTS


@pytest.mark.parametrize("ext", [".jpg", ".jpeg", ".png", ".webp", ".bmp"])
def test_is_image_path_true(ext: str) -> None:
    assert is_image_path(Path(f"file{ext}"))


@pytest.mark.parametrize("ext", [".mp4", ".mov", ".m4v"])
def test_is_image_path_false_for_video(ext: str) -> None:
    assert not is_image_path(Path(f"file{ext}"))


@pytest.mark.parametrize("ext", [".mp4", ".mov", ".m4v"])
def test_is_video_path_true(ext: str) -> None:
    assert is_video_path(Path(f"file{ext}"))


# Regression guard: matroska / avi / webm are intentionally NOT supported
# (their codec ecosystems include royalty-encumbered video that we don't ship).
@pytest.mark.parametrize("ext", [".mkv", ".avi", ".webm"])
def test_is_video_path_false_for_unsupported(ext: str) -> None:
    assert not is_video_path(Path(f"file{ext}"))


@pytest.mark.parametrize("ext", [".jpg", ".jpeg", ".png", ".webp", ".bmp"])
def test_is_video_path_false_for_image(ext: str) -> None:
    assert not is_video_path(Path(f"file{ext}"))


# ---------------------------------------------------------------------------
# PIL can actually open all advertised image formats
# ---------------------------------------------------------------------------


def _write_rgb_image(path: Path, w: int = 16, h: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h), color=(100, 150, 200)).save(path)


@pytest.mark.parametrize("ext", [".jpg", ".jpeg", ".png", ".bmp"])
def test_pil_can_open_standard_images(tmp_path: Path, ext: str) -> None:
    p = tmp_path / f"test{ext}"
    _write_rgb_image(p)
    with Image.open(p) as im:
        assert im.size == (16, 8)


def test_pil_can_open_webp(tmp_path: Path) -> None:
    p = tmp_path / "test.webp"
    _write_rgb_image(p)
    with Image.open(p) as im:
        assert im.size == (16, 8)


# ---------------------------------------------------------------------------
# PyAV and ffmpeg-python can read all claimed video formats
# (uses PyAV to write synthetic videos so no GPL encoder is required)
# ---------------------------------------------------------------------------


def _write_test_video(path: Path, codec: str = "mpeg4", pix_fmt: str = "yuv420p") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream(codec, rate=5)
        stream.width = 64
        stream.height = 32
        stream.pix_fmt = pix_fmt
        for i in range(5):
            frame = av.VideoFrame.from_ndarray(np.full((32, 64, 3), 50 + i * 10, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


_VIDEO_CODEC_MAP = {
    ".mp4": ("mpeg4", "yuv420p"),
    ".mov": ("mpeg4", "yuv420p"),
    ".m4v": ("mpeg4", "yuv420p"),
}


def _write_video_with_ext(tmp_path: Path, ext: str) -> Path:
    """Helper: write a synthetic test clip with the requested extension.

    The shipped FFmpeg build only ships the `mp4` muxer (the `mov` demuxer
    handles .mp4 / .mov / .m4v / .3gp reads transparently, but there is no
    standalone `mov` or `m4v` muxer). So we always write as .mp4 first and
    rename — exercising the demuxer with the requested extension on read.
    """
    src = tmp_path / "clip.mp4"
    codec, pix_fmt = _VIDEO_CODEC_MAP[ext]
    _write_test_video(src, codec=codec, pix_fmt=pix_fmt)
    if ext == ".mp4":
        return src
    dst = tmp_path / f"clip{ext}"
    src.rename(dst)
    return dst


@pytest.mark.parametrize("ext", [".mp4", ".mov", ".m4v"])
def test_pyav_can_read_video_format(tmp_path: Path, ext: str) -> None:
    p = _write_video_with_ext(tmp_path, ext)
    with av.open(str(p)) as container:
        frame = next(container.decode(video=0))
    arr = frame.to_ndarray(format="rgb24")
    assert arr.shape == (32, 64, 3), f"PyAV produced bad frame shape for {ext}: {arr.shape}"


@pytest.mark.skipif(not _ffprobe_available, reason="ffprobe not in PATH (runs inside Docker)")
@pytest.mark.parametrize("ext", [".mp4", ".mov", ".m4v"])
def test_ffmpeg_probe_can_read_video_format(tmp_path: Path, ext: str) -> None:
    p = _write_video_with_ext(tmp_path, ext)
    info = ffmpeg_python.probe(str(p))
    assert info["streams"], f"ffmpeg.probe returned no streams for {ext}"


# ---------------------------------------------------------------------------
# sample_frames_ffmpeg (VLM JSON / MCQ stage) can extract frames from all video formats
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _ffmpeg_available, reason="ffmpeg not in PATH (runs inside Docker)")
@pytest.mark.parametrize("ext", [".mp4", ".mov", ".m4v"])
def test_sample_frames_ffmpeg_can_extract_frames(tmp_path: Path, ext: str) -> None:
    p = _write_video_with_ext(tmp_path, ext)
    out_dir = tmp_path / f"frames{ext}"
    out_dir.mkdir()
    frames = sample_frames_ffmpeg(
        video_path=p,
        out_dir=out_dir,
        start_sec=0.0,
        end_sec=2.0,
        sampling_fps=1.0,
        resolution=32,
        max_frames=3,
        logger=logging.getLogger("test"),
    )
    assert len(frames) > 0, f"sample_frames_ffmpeg extracted 0 frames for {ext}"
    assert frames[0][1].exists(), f"frame file not written for {ext}"


# ---------------------------------------------------------------------------
# torchvision.io.VideoReader (SR stage) can read all claimed video formats
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ext", [".mp4", ".mov", ".m4v"])
def test_torchvision_video_reader_can_read_format(tmp_path: Path, ext: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from torchvision.io import VideoReader  # type: ignore[import]

    p = _write_video_with_ext(tmp_path, ext)
    vr = VideoReader(str(p), "video")
    frame = next(iter(vr))
    assert tuple(frame["data"].shape) == (3, 32, 64), f"VideoReader bad shape for {ext}"


# ---------------------------------------------------------------------------
# SR inference extension filter accepts all VIDEO_EXTS
# (regression test for the previous .mp4-only restriction)
# ---------------------------------------------------------------------------


def test_sr_inference_filter_accepts_all_video_exts() -> None:
    """SR keeps a standalone allowlist for SeedVR runtime importability.

    The direct SR allowlist equality check lives in
    tests/super_resolution/test_seedvr2_window_video_writer.py; this guard keeps
    the shared predicate aligned with the advertised VIDEO_EXTS policy.
    """
    for ext in VIDEO_EXTS:
        fake_name = f"clip{ext}"
        assert is_video_path(Path(fake_name)), (
            f"{ext} not accepted as a video by the shared filter — SR would silently skip"
        )
