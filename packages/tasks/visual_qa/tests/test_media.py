# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

import pytest
from core.media import VideoDecodePlan, VideoStreamInfo
from visual_qa import media
from visual_qa.media import (
    MediaDecodeError,
    MediaWindow,
    VideoInfo,
    extract_window_frames,
    extract_window_video_payload,
    probe_video,
)


def test_probe_video_uses_shared_decode_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = VideoStreamInfo(
        codec_name="vp9",
        width=640,
        height=480,
        fps=24.0,
        frame_count=48,
    )
    plan = VideoDecodePlan(stream=stream, decoder_name="vp9")
    monkeypatch.setattr(media, "_prepare_video_decode", lambda _path: plan)

    assert probe_video(Path("clip.webm")) == VideoInfo(
        fps=24.0,
        frame_count=48,
        width=640,
        height=480,
    )


def test_extract_window_video_payload_uses_original_for_full_window(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-mp4")

    payload = extract_window_video_payload(
        media_path=video,
        window=MediaWindow(
            index=0,
            start_s=0.0,
            end_s=1.0,
            start_frame=0,
            end_frame=3,
        ),
        total_frames=4,
    )

    assert payload.mime_type == "video/mp4"
    assert payload.filename == "clip.mp4"
    assert payload.num_bytes == len(b"fake-mp4")


def test_extract_window_frames_rejects_empty_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeCv2:
        COLOR_RGB2BGR = 1

    stream = VideoStreamInfo(codec_name="vp9", width=2, height=1, fps=30.0)
    plan = VideoDecodePlan(stream=stream, decoder_name="vp9")
    monkeypatch.setattr(media, "_import_cv2", lambda: _FakeCv2)
    monkeypatch.setattr(media, "_prepare_video_decode", lambda _path: plan)
    monkeypatch.setattr(media, "iter_decoded_rgb24_frames", lambda *_args: iter(()))

    with pytest.raises(MediaDecodeError, match=r"No frames decoded.*clip\.mp4"):
        extract_window_frames(
            media_path=tmp_path / "clip.mp4",
            window=MediaWindow(
                index=0,
                start_s=0.0,
                end_s=1.0,
                start_frame=0,
                end_frame=3,
            ),
            source_fps=30.0,
            sampling_fps=1.0,
            max_frames=2,
            resolution=64,
        )


def test_extract_window_frames_samples_shared_rgb_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeCv2:
        COLOR_RGB2BGR = 1

        @staticmethod
        def cvtColor(frame: Any, _code: int) -> Any:  # noqa: N802
            return frame[:, :, ::-1]

        @staticmethod
        def imencode(_extension: str, frame: Any) -> tuple[bool, Any]:
            return True, frame.reshape(-1)

    stream = VideoStreamInfo(codec_name="vp9", width=1, height=1, fps=2.0)
    plan = VideoDecodePlan(stream=stream, decoder_name="vp9")
    monkeypatch.setattr(media, "_prepare_video_decode", lambda _path: plan)
    monkeypatch.setattr(media, "_import_cv2", lambda: _FakeCv2)
    monkeypatch.setattr(
        media,
        "iter_decoded_rgb24_frames",
        lambda *_args: iter([b"\x00\x00\x00", b"\x01\x02\x03", b"\x04\x05\x06"]),
    )

    payloads = extract_window_frames(
        media_path=Path("clip.webm"),
        window=MediaWindow(
            index=0,
            start_s=0.0,
            end_s=1.0,
            start_frame=1,
            end_frame=2,
        ),
        source_fps=2.0,
        sampling_fps=1.0,
        max_frames=2,
        resolution=64,
    )

    assert [payload.frame_index for payload in payloads] == [1]
    assert payloads[0].width == 1
    assert payloads[0].height == 1


def test_extract_window_video_payload_uses_core_decode_and_vp9_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video fixture")
    stream = VideoStreamInfo(codec_name="vp9", width=1, height=1, fps=24.0)
    plan = VideoDecodePlan(stream=stream, decoder_name="vp9")
    writes: list[Any] = []
    writer_options: dict[str, Any] = {}

    class _FakeWriter:
        def __init__(self, output_path: Path, **kwargs: Any) -> None:
            writer_options.update(kwargs)
            self.output_path = output_path

        def __enter__(self) -> "_FakeWriter":
            return self

        def __exit__(self, *_args: object) -> None:
            self.output_path.write_bytes(b"vp9-mp4")

        def write(self, frame: Any) -> None:
            writes.append(frame.copy())

    monkeypatch.setattr(media, "_prepare_video_decode", lambda _path: plan)
    monkeypatch.setattr(media, "Vp9VideoWriter", _FakeWriter)
    monkeypatch.setattr(
        media,
        "iter_decoded_rgb24_frames",
        lambda *_args: iter([b"\x00\x00\x00", b"\x01\x02\x03", b"\x04\x05\x06"]),
    )

    payload = extract_window_video_payload(
        media_path=video,
        window=MediaWindow(
            index=0,
            start_s=0.0,
            end_s=0.5,
            start_frame=1,
            end_frame=2,
        ),
        total_frames=10,
    )

    assert payload.num_bytes == len(b"vp9-mp4")
    assert writer_options["frame_format"] == "rgb24"
    assert [frame.tolist() for frame in writes] == [[[[1, 2, 3]]], [[[4, 5, 6]]]]
