# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from core.media import VideoDecodePlan, VideoStreamInfo, vp9_output
from detection_and_tracking import media
from detection_and_tracking.media import (
    Vp9VideoWriter,
    decode_video_bgr,
    iter_selected_video_frames_bgr,
)


class _FakeCv2:
    COLOR_RGB2BGR = 1

    @staticmethod
    def cvtColor(frame: Any, code: int) -> Any:  # noqa: N802
        assert code == _FakeCv2.COLOR_RGB2BGR
        return frame[:, :, ::-1]


def test_decode_video_bgr_uses_core_decode_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = VideoStreamInfo(codec_name="vp9", width=2, height=1, fps=24.0)
    plan = VideoDecodePlan(stream=stream, decoder_name="vp9")
    payload = bytes([1, 2, 3, 4, 5, 6])
    monkeypatch.setattr(media, "prepare_video_decode", lambda _path: plan)
    monkeypatch.setattr(
        media,
        "iter_decoded_rgb24_frames",
        lambda _path, _plan: iter([payload]),
    )

    decoded = decode_video_bgr(Path("clip.webm"), cv2=_FakeCv2(), np=np)

    assert decoded.stream is stream
    assert list(decoded.frames)[0].tolist() == [[[3, 2, 1], [6, 5, 4]]]


def test_iter_selected_video_frames_bgr_skips_and_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = VideoStreamInfo(codec_name="vp9", width=1, height=1, fps=24.0)
    plan = VideoDecodePlan(stream=stream, decoder_name="vp9")
    payloads = [
        bytes([10, 20, 30]),
        bytes([40, 50, 60]),
        bytes([70, 80, 90]),
        bytes([1, 2, 3]),
    ]
    seen_plans: list[VideoDecodePlan] = []

    def _iter(_path: Path, decode_plan: VideoDecodePlan) -> Any:
        seen_plans.append(decode_plan)
        yield from payloads

    monkeypatch.setattr(media, "prepare_video_decode", lambda _path: plan)
    monkeypatch.setattr(media, "iter_decoded_rgb24_frames", _iter)

    selected = list(
        iter_selected_video_frames_bgr(
            Path("clip.webm"),
            [1, 2],
            cv2=_FakeCv2(),
            np=np,
        )
    )

    assert seen_plans == [plan]
    assert [idx for idx, _frame in selected] == [1, 2]
    assert selected[0][1].tolist() == [[[60, 50, 40]]]
    assert selected[1][1].tolist() == [[[90, 80, 70]]]


def test_iter_selected_video_frames_bgr_empty() -> None:
    assert (
        list(
            iter_selected_video_frames_bgr(
                Path("clip.webm"),
                [],
                cv2=_FakeCv2(),
                np=np,
            )
        )
        == []
    )


def test_iter_selected_video_frames_bgr_rejects_negative_indices() -> None:
    with pytest.raises(ValueError, match="non-negative 0-based"):
        list(
            iter_selected_video_frames_bgr(
                Path("clip.webm"),
                [0, -1, 2],
                cv2=_FakeCv2(),
                np=np,
            )
        )


class _FakeCodecContext:
    color_range = 0
    color_primaries = 0
    color_trc = 0
    colorspace = 0


class _FakeStream:
    def __init__(self) -> None:
        self.codec_context = _FakeCodecContext()
        self.options: dict[str, str] = {}
        self.width = 0
        self.height = 0
        self.pix_fmt = ""
        self.frames: list[Any] = []

    def encode(self, frame: Any) -> list[str]:
        self.frames.append(frame)
        return ["flush" if frame is None else "packet"]


class _FakeContainer:
    def __init__(self) -> None:
        self.stream = _FakeStream()
        self.codec_name = ""
        self.rate: Any = None
        self.packets: list[str] = []
        self.closed = False

    def add_stream(self, codec_name: str, *, rate: Any) -> _FakeStream:
        self.codec_name = codec_name
        self.rate = rate
        return self.stream

    def mux(self, packet: str) -> None:
        self.packets.append(packet)

    def close(self) -> None:
        self.closed = True


def test_vp9_writer_uses_shared_encoder_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    container = _FakeContainer()
    av_module = SimpleNamespace(
        open=lambda *_args, **_kwargs: container,
        VideoFrame=SimpleNamespace(from_ndarray=lambda frame, format: (frame, format)),
    )
    monkeypatch.setattr(vp9_output, "_import_av", lambda: av_module)

    with Vp9VideoWriter(
        tmp_path / "overlay.mp4",
        width=1280,
        height=720,
        fps=30.0,
    ) as writer:
        writer.write("frame")

    assert container.codec_name == "libvpx-vp9"
    assert container.stream.width == 1280
    assert container.stream.height == 720
    assert container.stream.pix_fmt == "yuv420p"
    assert container.stream.frames == [("frame", "bgr24"), None]
    assert container.packets == ["packet", "flush"]
    assert container.closed is True
