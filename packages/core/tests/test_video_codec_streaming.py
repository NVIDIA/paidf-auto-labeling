# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
from typing import Any

import pytest
from core.media import video_codecs
from core.media.video_codecs import (
    VideoDecodeError,
    VideoDecodePlan,
    VideoStreamInfo,
    iter_decoded_rgb24_frames,
)


class _FakeProcess:
    def __init__(self, payload: bytes, *, return_code: int = 0) -> None:
        self.stdout = io.BytesIO(payload)
        self.return_code = return_code
        self.finished = False
        self.terminated = False
        self.killed = False

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.finished = True
        return self.return_code

    def poll(self) -> int | None:
        return self.return_code if self.finished else None

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


class _TimeoutProcess(_FakeProcess):
    def wait(self, timeout: float | None = None) -> int:
        if not self.terminated and not self.killed:
            raise video_codecs.subprocess.TimeoutExpired(["ffmpeg"], timeout or 0)
        self.finished = True
        return self.return_code


class _ExitedProcess(_FakeProcess):
    def terminate(self) -> None:
        self.terminated = True
        raise ProcessLookupError


class _KillRaceProcess(_FakeProcess):
    def wait(self, timeout: float | None = None) -> int:
        if timeout is not None:
            raise video_codecs.subprocess.TimeoutExpired(["ffmpeg"], timeout)
        self.finished = True
        raise ProcessLookupError

    def kill(self) -> None:
        self.killed = True
        raise ProcessLookupError


def _plan() -> VideoDecodePlan:
    return VideoDecodePlan(
        stream=VideoStreamInfo(codec_name="vp9", width=2, height=1, fps=24.0),
        decoder_name="vp9",
    )


def test_iter_decoded_frames_yields_exact_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    process = _FakeProcess(b"abcdefghijkl")
    monkeypatch.setattr(video_codecs.subprocess, "Popen", lambda *_args, **_kwargs: process)

    assert list(iter_decoded_rgb24_frames("clip.webm", _plan())) == [
        b"abcdef",
        b"ghijkl",
    ]
    assert process.finished is True
    assert process.terminated is False


def test_iter_decoded_frames_rejects_partial_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    process = _FakeProcess(b"short")
    monkeypatch.setattr(video_codecs.subprocess, "Popen", lambda *_args, **_kwargs: process)

    with pytest.raises(VideoDecodeError, match="partial RGB frame"):
        list(iter_decoded_rgb24_frames("clip.webm", _plan()))

    assert process.terminated is True


def test_iter_decoded_frames_reports_ffmpeg_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    process = _FakeProcess(b"", return_code=1)

    def popen(*_args: object, **kwargs: Any) -> _FakeProcess:
        kwargs["stderr"].write(b"corrupt bitstream")
        return process

    monkeypatch.setattr(video_codecs.subprocess, "Popen", popen)

    with pytest.raises(VideoDecodeError, match="corrupt bitstream"):
        list(iter_decoded_rgb24_frames("clip.webm", _plan()))


def test_iter_decoded_frames_terminates_and_reaps_after_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _TimeoutProcess(b"")

    def popen(*_args: object, **kwargs: Any) -> _TimeoutProcess:
        kwargs["stderr"].write(b"decoder stalled")
        return process

    monkeypatch.setattr(video_codecs.subprocess, "Popen", popen)

    with pytest.raises(
        VideoDecodeError,
        match=r"FFmpeg decoder 'vp9' timed out decoding clip\.webm: decoder stalled",
    ):
        list(iter_decoded_rgb24_frames("clip.webm", _plan()))

    assert process.terminated is True
    assert process.finished is True


def test_iter_decoded_frames_closes_process_when_consumer_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess(b"abcdefghijkl")
    monkeypatch.setattr(video_codecs.subprocess, "Popen", lambda *_args, **_kwargs: process)
    frames = iter(iter_decoded_rgb24_frames("clip.webm", _plan()))

    assert next(frames) == b"abcdef"
    frames.close()  # type: ignore[attr-defined]

    assert process.terminated is True


def test_terminate_and_reap_waits_when_process_already_exited() -> None:
    process = _ExitedProcess(b"")

    video_codecs._terminate_and_reap(process)  # type: ignore[arg-type]

    assert process.terminated is True
    assert process.finished is True


def test_terminate_and_reap_ignores_kill_and_wait_process_lookup_races() -> None:
    process = _KillRaceProcess(b"")

    video_codecs._terminate_and_reap(process)  # type: ignore[arg-type]

    assert process.killed is True
    assert process.finished is True
