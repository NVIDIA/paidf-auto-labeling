# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from captioning import media
from captioning.config import CaptioningConfig
from captioning.media import (
    CaptionWindow,
    MediaDecodeError,
    VideoInfo,
    _resize_frame,
    _sample_frame_ids,
    extract_window_frames,
    extract_window_video_payload,
    plan_windows,
    probe_video,
)
from core.media import VideoDecodeError, VideoDecodePlan, VideoStreamInfo


def _failing_frames(*_args: object) -> Iterator[bytes]:
    return (_raise_decode_error() for _ in range(1))


def _raise_decode_error() -> bytes:
    raise VideoDecodeError("decode failed")


def test_plan_windows_by_seconds() -> None:
    windows = plan_windows(
        VideoInfo(fps=10.0, frame_count=25, width=640, height=480),
        CaptioningConfig(window_seconds=1.0, window_frames=0),
    )

    assert [(w.start_frame, w.end_frame) for w in windows] == [(0, 9), (10, 19), (20, 24)]
    assert [(w.start_s, round(w.end_s, 1)) for w in windows] == [(0.0, 1.0), (1.0, 2.0), (2.0, 2.5)]


def test_plan_windows_by_seconds_uses_ceil_end_boundary() -> None:
    windows = plan_windows(
        VideoInfo(fps=2.0, frame_count=5, width=640, height=480),
        CaptioningConfig(window_seconds=0.75, window_frames=0),
    )

    assert [(w.start_frame, w.end_frame) for w in windows] == [
        (0, 1),
        (1, 2),
        (3, 4),
        (4, 4),
    ]


def test_plan_windows_returns_empty_for_non_positive_seconds() -> None:
    config = CaptioningConfig(window_seconds=1.0)
    config.window_frames = 0
    config.window_seconds = 0.0

    assert plan_windows(VideoInfo(fps=10.0, frame_count=25, width=640, height=480), config) == []

    config.window_seconds = -1.0

    assert plan_windows(VideoInfo(fps=10.0, frame_count=25, width=640, height=480), config) == []


def test_plan_windows_by_frames_uses_remainder_threshold() -> None:
    windows = plan_windows(
        VideoInfo(fps=0.0, frame_count=5, width=640, height=480),
        CaptioningConfig(window_frames=2, remainder_threshold=2),
    )

    assert [(w.start_frame, w.end_frame) for w in windows] == [(0, 1), (2, 4)]
    assert [(w.start_s, w.end_s) for w in windows] == [(0.0, 0.0), (0.0, 0.0)]


def test_plan_windows_by_frames_keeps_large_remainder() -> None:
    windows = plan_windows(
        VideoInfo(fps=10.0, frame_count=5, width=640, height=480),
        CaptioningConfig(window_frames=2, remainder_threshold=1),
    )

    assert [(w.start_frame, w.end_frame) for w in windows] == [(0, 1), (2, 3), (4, 4)]


def test_plan_windows_by_frames_requires_minimum_frames() -> None:
    windows = plan_windows(
        VideoInfo(fps=10.0, frame_count=3, width=640, height=480),
        CaptioningConfig(window_frames=2, remainder_threshold=1),
    )

    assert windows == []


def test_plan_single_window() -> None:
    windows = plan_windows(
        VideoInfo(fps=10.0, frame_count=25, width=640, height=480),
        CaptioningConfig(single_window=True),
    )

    assert len(windows) == 1
    assert windows[0].start_frame == 0
    assert windows[0].end_frame == 24


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


def test_probe_video_normalizes_frame_count_decode_error(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = VideoStreamInfo(codec_name="vp9", width=1, height=1, fps=24.0)
    plan = VideoDecodePlan(stream=stream, decoder_name="vp9")
    monkeypatch.setattr(media, "_prepare_video_decode", lambda _path: plan)

    monkeypatch.setattr(media, "iter_decoded_rgb24_frames", _failing_frames)

    with pytest.raises(MediaDecodeError, match=r"count decoded frames.*clip\.webm") as exc_info:
        probe_video(Path("clip.webm"))

    assert isinstance(exc_info.value.__cause__, VideoDecodeError)


def test_extract_window_video_payload_uses_original_for_full_window(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-mp4")

    payload = extract_window_video_payload(
        media_path=video,
        window=CaptionWindow(
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


def test_extract_window_frames_rejects_non_positive_sampling_fps(tmp_path: Path) -> None:
    with pytest.raises(MediaDecodeError, match="sampling_fps must be greater than 0"):
        extract_window_frames(
            media_path=tmp_path / "missing.mp4",
            window=CaptionWindow(
                index=0,
                start_s=0.0,
                end_s=1.0,
                start_frame=0,
                end_frame=1,
            ),
            source_fps=30.0,
            sampling_fps=0.0,
            max_frames=1,
            resolution=64,
        )


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
    video = tmp_path / "clip.mp4"

    with pytest.raises(MediaDecodeError, match=r"No frames decoded.*clip\.mp4"):
        extract_window_frames(
            media_path=video,
            window=CaptionWindow(
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


def test_extract_window_frames_normalizes_decode_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = VideoStreamInfo(codec_name="vp9", width=1, height=1, fps=30.0)
    plan = VideoDecodePlan(stream=stream, decoder_name="vp9")
    monkeypatch.setattr(media, "_prepare_video_decode", lambda _path: plan)
    monkeypatch.setattr(media, "_import_cv2", lambda: object())

    monkeypatch.setattr(media, "iter_decoded_rgb24_frames", _failing_frames)

    with pytest.raises(MediaDecodeError, match=r"sampled frames.*clip\.webm") as exc_info:
        extract_window_frames(
            media_path=Path("clip.webm"),
            window=CaptionWindow(0, 0.0, 1.0, 0, 1),
            source_fps=30.0,
            sampling_fps=1.0,
            max_frames=1,
            resolution=64,
        )

    assert isinstance(exc_info.value.__cause__, VideoDecodeError)


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
        window=CaptionWindow(
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


def test_sample_frame_ids_rejects_non_positive_max_frames() -> None:
    with pytest.raises(MediaDecodeError, match="max_frames must be greater than 0"):
        _sample_frame_ids(
            start_frame=0,
            end_frame=10,
            source_fps=30.0,
            sampling_fps=2.0,
            max_frames=0,
        )


def test_extract_window_video_payload_rejects_missing_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="media_path must be an existing file"):
        extract_window_video_payload(
            media_path=tmp_path / "missing.mp4",
            window=CaptionWindow(
                index=0,
                start_s=0.0,
                end_s=0.5,
                start_frame=0,
                end_frame=1,
            ),
            total_frames=10,
        )


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
        window=CaptionWindow(
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


def test_extract_window_video_payload_normalizes_decode_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "clip.webm"
    video.write_bytes(b"fake video fixture")
    stream = VideoStreamInfo(codec_name="vp9", width=1, height=1, fps=24.0)
    plan = VideoDecodePlan(stream=stream, decoder_name="vp9")

    class _FakeWriter:
        def __init__(self, _output_path: Path, **_kwargs: Any) -> None:
            pass

        def __enter__(self) -> "_FakeWriter":
            return self

        def __exit__(self, *_args: object) -> None:
            pass

    monkeypatch.setattr(media, "_prepare_video_decode", lambda _path: plan)
    monkeypatch.setattr(media, "Vp9VideoWriter", _FakeWriter)
    monkeypatch.setattr(media, "iter_decoded_rgb24_frames", _failing_frames)

    with pytest.raises(MediaDecodeError, match=r"decode window.*clip\.webm") as exc_info:
        extract_window_video_payload(
            media_path=video,
            window=CaptionWindow(0, 0.0, 0.5, 0, 1),
            total_frames=10,
        )

    assert isinstance(exc_info.value.__cause__, VideoDecodeError)


@pytest.mark.parametrize("resolution", [0, -1])
def test_resize_frame_rejects_non_positive_resolution(resolution: int) -> None:
    with pytest.raises(ValueError, match="resolution must be a positive integer"):
        _resize_frame(object(), object(), resolution)
