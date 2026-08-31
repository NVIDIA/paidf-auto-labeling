# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression guard: captioning must decode VP9 media.

The super-resolution stage now emits VP9-encoded video (per the ``main`` codec
policy). These tests fail if the captioning decode paths (OpenCV probing and the
ffmpeg window extractor) can no longer read VP9 input, e.g. after an ffmpeg or
OpenCV build regression in the service image.

They are skipped where the media stack is unavailable so minimal environments
stay deterministic; they only assert VP9 support where the tools exist.
"""

from __future__ import annotations

import base64
import shutil
import subprocess
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2", reason="OpenCV is required to decode VP9 media")

from captioning import media  # noqa: E402 - imported after the OpenCV availability gate.
from captioning.media import CaptionWindow, extract_window_frames  # noqa: E402


def _ffmpeg_with_vp9_encoder() -> bool:
    """Return True when a local ffmpeg can encode VP9 to build the fixture."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return "libvpx-vp9" in result.stdout


def _ffprobe_available() -> bool:
    """Return True when ``ffprobe`` is on PATH (used by the codec-check helper)."""
    return shutil.which("ffprobe") is not None


pytestmark = pytest.mark.skipif(
    not (_ffmpeg_with_vp9_encoder() and _ffprobe_available()),
    reason=(
        "ffmpeg with the libvpx-vp9 encoder and ffprobe are required to build and "
        "inspect the VP9 fixture"
    ),
)

_CLIP_WIDTH = 64
_CLIP_HEIGHT = 48
_CLIP_FPS = 12
_CLIP_FRAMES = 12


@pytest.fixture
def vp9_clip(tmp_path: Path) -> Path:
    """Write a tiny VP9 clip standing in for super-resolution output."""
    clip = tmp_path / "sr_output.webm"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={_CLIP_WIDTH}x{_CLIP_HEIGHT}:rate={_CLIP_FPS}"
            f":duration={_CLIP_FRAMES / _CLIP_FPS}",
            "-c:v",
            "libvpx-vp9",
            "-pix_fmt",
            "yuv420p",
            str(clip),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    return clip


def test_probe_video_decodes_vp9(vp9_clip: Path) -> None:
    """OpenCV must open VP9 media and report a positive frame count."""
    info = media.probe_video(vp9_clip)

    assert info.frame_count > 0
    assert info.width == _CLIP_WIDTH
    assert info.height == _CLIP_HEIGHT


def _payload_video_codec(data: bytes, tmp_path: Path) -> str:
    """Return the video codec name inside an encoded payload via ffprobe."""
    clip = tmp_path / "payload_probe.mp4"
    clip.write_bytes(data)
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(clip),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


def test_extract_window_video_payload_decodes_vp9(vp9_clip: Path, tmp_path: Path) -> None:
    """The window extractor must decode VP9 input and re-encode to VP9-in-MP4.

    The re-encode must use the royalty-free ``libvpx-vp9`` encoder (never a
    GPL/patent-encumbered H.264 encoder), so the payload stays license-clean.
    """
    info = media.probe_video(vp9_clip)
    window = CaptionWindow(
        index=0,
        start_s=0.0,
        end_s=0.5,
        start_frame=0,
        end_frame=min(5, info.frame_count - 1),
    )

    payload = media.extract_window_video_payload(
        media_path=vp9_clip,
        window=window,
        total_frames=info.frame_count,
    )

    assert payload.mime_type == "video/mp4"
    assert payload.num_bytes is not None
    assert payload.num_bytes > 0
    assert _payload_video_codec(base64.b64decode(payload.data_base64), tmp_path) == "vp9"


def test_extract_window_frames_decodes_vp9(vp9_clip: Path) -> None:
    """The seek-based frame extractor must decode VP9 into JPEG payloads."""
    info = media.probe_video(vp9_clip)
    window = CaptionWindow(
        index=0,
        start_s=0.0,
        end_s=1.0,
        start_frame=0,
        end_frame=info.frame_count - 1,
    )

    payloads = extract_window_frames(
        media_path=vp9_clip,
        window=window,
        source_fps=info.fps,
        sampling_fps=info.fps,
        max_frames=info.frame_count,
        resolution=32,
    )

    assert payloads
    assert all(p.mime_type == "image/jpeg" for p in payloads)
    assert all(p.num_bytes and p.num_bytes > 0 for p in payloads)
    assert all(
        p.frame_index is not None and 0 <= p.frame_index <= info.frame_count - 1 for p in payloads
    )
