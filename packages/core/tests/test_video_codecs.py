# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from core.media import video_codecs
from core.media.video_codecs import (
    H264_GPU_REQUIRED_MESSAGE,
    UnsupportedVideoCodecError,
    VideoDecodeError,
    VideoDecoderUnavailableError,
    VideoStreamInfo,
    prepare_video_decode,
    probe_video_stream,
)


def _stream(codec_name: str) -> VideoStreamInfo:
    return VideoStreamInfo(codec_name=codec_name, width=16, height=8, fps=24.0)


def _probe_result(
    streams: list[dict[str, object]], *, format_payload: dict[str, object] | None = None
) -> SimpleNamespace:
    payload: dict[str, object] = {"streams": streams}
    if format_payload is not None:
        payload["format"] = format_payload
    return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")


def test_probe_video_stream_parses_and_normalizes_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _probe_result(
        [
            {
                "codec_name": "vp9_cuvid",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30000/1001",
                "nb_frames": "300",
            }
        ],
        format_payload={"duration": "10.01"},
    )
    monkeypatch.setattr(video_codecs.subprocess, "run", lambda *_args, **_kwargs: result)

    stream = probe_video_stream("clip.webm")

    assert stream.codec_name == "vp9"
    assert stream.width == 1920
    assert stream.height == 1080
    assert stream.fps == pytest.approx(29.97003)
    assert stream.frame_count == 300
    assert stream.duration_seconds == pytest.approx(10.01)


def test_probe_video_stream_derives_duration_from_frame_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _probe_result(
        [
            {
                "codec_name": "vp9",
                "width": 16,
                "height": 8,
                "avg_frame_rate": "24/1",
                "nb_frames": "48",
            }
        ]
    )
    monkeypatch.setattr(video_codecs.subprocess, "run", lambda *_args, **_kwargs: result)

    stream = probe_video_stream("clip.webm")

    assert stream.frame_count == 48
    assert stream.duration_seconds == 2.0


def test_probe_video_stream_rejects_missing_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        video_codecs.subprocess,
        "run",
        lambda *_args, **_kwargs: _probe_result([]),
    )

    with pytest.raises(VideoDecodeError, match="no video stream"):
        probe_video_stream("audio.mp4")


def test_probe_video_stream_maps_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired(["ffprobe"], 30)

    monkeypatch.setattr(video_codecs.subprocess, "run", timeout)

    with pytest.raises(VideoDecodeError, match="Could not probe"):
        probe_video_stream("clip.mp4")


def test_h264_requires_working_hardware_decoder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(video_codecs, "probe_video_stream", lambda _path: _stream("h264"))
    monkeypatch.setattr(video_codecs, "_decoder_works", lambda *_args: False)

    with pytest.raises(VideoDecoderUnavailableError) as exc_info:
        prepare_video_decode(Path("clip.mp4"))

    assert str(exc_info.value) == H264_GPU_REQUIRED_MESSAGE


def test_h264_selects_cuvid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(video_codecs, "probe_video_stream", lambda _path: _stream("h264"))
    monkeypatch.setattr(video_codecs, "_decoder_works", lambda *_args: True)

    assert prepare_video_decode("clip.mp4").decoder_name == "h264_cuvid"


def test_vp9_prefers_cuvid_then_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[str] = []

    def decoder_works(_path: object, _stream_info: object, decoder_name: str) -> bool:
        attempts.append(decoder_name)
        return decoder_name == "vp9"

    monkeypatch.setattr(video_codecs, "probe_video_stream", lambda _path: _stream("vp9"))
    monkeypatch.setattr(video_codecs, "_decoder_works", decoder_works)

    assert prepare_video_decode("clip.webm").decoder_name == "vp9"
    assert attempts == ["vp9_cuvid", "vp9"]


def test_mpeg4_part2_selects_software_decoder(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[str] = []

    def decoder_works(_path: object, _stream_info: object, decoder_name: str) -> bool:
        attempts.append(decoder_name)
        return True

    monkeypatch.setattr(video_codecs, "probe_video_stream", lambda _path: _stream("mpeg4"))
    monkeypatch.setattr(video_codecs, "_decoder_works", decoder_works)

    assert prepare_video_decode("clip.mp4").decoder_name == "mpeg4"
    assert attempts == ["mpeg4"]


def test_mpeg4_part2_requires_working_software_decoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(video_codecs, "probe_video_stream", lambda _path: _stream("mpeg4"))
    monkeypatch.setattr(video_codecs, "_decoder_works", lambda *_args: False)

    with pytest.raises(VideoDecoderUnavailableError, match="MPEG-4 Part 2"):
        prepare_video_decode("clip.mp4")


def test_rejects_other_video_codecs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(video_codecs, "probe_video_stream", lambda _path: _stream("hevc"))

    with pytest.raises(UnsupportedVideoCodecError, match="supported codecs: h264, vp9, mpeg4"):
        prepare_video_decode("clip.mp4")


def test_read_exact_combines_short_reads() -> None:
    class ShortReader(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            limit = -1 if size is None else size
            return super().read(min(limit, 2))

    assert video_codecs._read_exact(ShortReader(b"abcdef"), 6) == b"abcdef"
