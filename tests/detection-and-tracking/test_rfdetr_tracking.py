#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for rfdetr_tracking.py"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import av
import numpy as np
import pytest

# rfdetr_tracking imports rfdetr at module import time; skip these tests when the tracking
# dependency is not installed (common in minimal/dev environments).
pytest.importorskip("rfdetr")

from rfdetr_tracking import (
    _is_nvenc_error,
    _normalize_class_name,
    _PyAvVideoWriter,
    process_video,
)


def _make_model():
    model = MagicMock()
    det = MagicMock()
    det.xyxy = np.empty((0, 4))
    det.confidence = np.empty((0,))
    det.class_id = np.array([], dtype=int)
    model.predict.return_value = det
    return model


def _make_tracker():
    tracker = MagicMock()
    tracker.update.return_value = np.empty((0, 7))
    return tracker


def _write_test_video(path: Path, *, frame_count: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=5)
        stream.width = 64
        stream.height = 32
        stream.pix_fmt = "yuv420p"
        for i in range(frame_count):
            frame = av.VideoFrame.from_ndarray(
                np.full((32, 64, 3), 50 + i * 20, dtype=np.uint8),
                format="rgb24",
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _run_process_video_image(tmp_path: Path, imwrite_side_effect) -> None:
    """Call process_video in image mode with save_video=True and a mocked cv2.imwrite.

    The detection overlay for image inputs is a ``.png`` under
    ``<scene>/sidecars/``; see the module docstring for the full layout.
    """
    fake_frame = np.zeros((64, 64, 3), dtype=np.uint8)
    with (
        patch("rfdetr_tracking.cv2.imread", return_value=fake_frame),
        patch("rfdetr_tracking.cv2.imwrite", side_effect=imwrite_side_effect),
        patch("rfdetr_tracking.generate_filtered_video"),
    ):
        process_video(
            video_path=Path("dummy.png"),
            output_dir=tmp_path / "out",
            model=_make_model(),
            tracker=_make_tracker(),
            save_video=True,
            save_rgb=False,
            save_vis=False,
            save_video_red_id=False,
            copy_video=False,
            write_json=False,
        )


class TestDetectionImageWriteErrorHandling:
    """Tests for the error-handling in the save_video+image detection-image write path.

    The detection image is a visualization-only artifact; any write failure is logged
    as a warning and swallowed.  Stage-level failures (e.g. ENOSPC on critical JSON
    outputs) are surfaced by pipeline.py according to empty_output_policy.
    """

    def test_imwrite_returns_false_is_swallowed(self, tmp_path):
        _run_process_video_image(tmp_path, imwrite_side_effect=OSError("cv2.imwrite returned False"))

    def test_generic_oserror_is_swallowed(self, tmp_path):
        _run_process_video_image(tmp_path, imwrite_side_effect=OSError("some codec error"))

    def test_runtime_error_is_swallowed(self, tmp_path):
        _run_process_video_image(tmp_path, imwrite_side_effect=RuntimeError("unexpected"))

    def test_permission_error_is_swallowed(self, tmp_path):
        _run_process_video_image(tmp_path, imwrite_side_effect=PermissionError(13, "Permission denied"))

    def test_enospc_is_swallowed(self, tmp_path):
        _run_process_video_image(tmp_path, imwrite_side_effect=OSError(28, "No space left on device"))


class TestImageInputOverlaySuffix:
    """Image inputs must produce ``.png`` overlays, not 1-frame ``.mp4``.

    Regression guard: image inputs must keep image overlay suffixes instead of
    silently producing 1-frame videos.
    """

    def _run(self, tmp_path: Path, *, save_video: bool, save_video_red_id: bool):
        fake_frame = np.zeros((64, 64, 3), dtype=np.uint8)
        with (
            patch("rfdetr_tracking.cv2.imread", return_value=fake_frame),
            patch("rfdetr_tracking.cv2.imwrite", return_value=True),
            patch("rfdetr_tracking.generate_filtered_video") as gen_filtered,
        ):
            process_video(
                video_path=Path("photo.png"),
                output_dir=tmp_path / "scene",
                model=_make_model(),
                tracker=_make_tracker(),
                save_video=save_video,
                save_rgb=False,
                save_vis=False,
                save_video_red_id=save_video_red_id,
                copy_video=False,
                write_json=False,
            )
        return gen_filtered

    def test_tracking_overlay_written_as_png_for_image_input(self, tmp_path: Path) -> None:
        gen_filtered = self._run(tmp_path, save_video=True, save_video_red_id=False)
        tracking_calls = [c for c in gen_filtered.call_args_list if c.kwargs.get("track_vis_style") != "red_id"]
        assert tracking_calls, "expected a tracking overlay write"
        out = tracking_calls[0].kwargs["output_path"]
        assert out.suffix == ".png", f"image input should yield .png tracking overlay, got {out}"
        assert out.name == "photo_tracking.png"

    def test_red_id_overlay_written_as_png_for_image_input(self, tmp_path: Path) -> None:
        gen_filtered = self._run(tmp_path, save_video=False, save_video_red_id=True)
        red_id_calls = [c for c in gen_filtered.call_args_list if c.kwargs.get("track_vis_style") == "red_id"]
        assert red_id_calls, "expected a red-id overlay write"
        out = red_id_calls[0].kwargs["output_path"]
        assert out.suffix == ".png", f"image input should yield .png red-id overlay, got {out}"
        assert out.name == "photo_tracking_red_id.png"


class TestVideoIoViaPyAv:
    def test_plain_avcodec_open2_error_is_not_nvenc_error(self) -> None:
        assert not _is_nvenc_error(RuntimeError("avcodec_open2 failed for codec mpeg4"))

    def test_writer_closes_container_when_stream_setup_fails(self, tmp_path: Path) -> None:
        container = MagicMock()
        container.add_stream.side_effect = RuntimeError("stream setup failed")

        with (
            patch("rfdetr_tracking._select_video_encoder", return_value=("h264_nvenc", {"preset": "p4"})),
            patch("rfdetr_tracking.av.open", return_value=container),
            pytest.raises(RuntimeError, match="stream setup failed"),
        ):
            _PyAvVideoWriter(tmp_path / "out.mp4", fps=30, width=64, height=32)

        container.close.assert_called_once()

    def test_process_video_writes_overlay_videos(self, tmp_path: Path) -> None:
        video_path = tmp_path / "clip.mp4"
        _write_test_video(video_path)

        # The real writer prefers h264_nvenc and falls back to mpeg4 when NVENC
        # is unavailable (e.g. CI runners without a GPU), so we can exercise the
        # actual PyAV encode path here without mocking.
        process_video(
            video_path=video_path,
            output_dir=tmp_path / "scene",
            model=_make_model(),
            tracker=_make_tracker(),
            save_video=True,
            save_rgb=False,
            save_vis=False,
            save_video_red_id=True,
            copy_video=False,
            write_json=False,
        )

        sidecars = tmp_path / "scene" / "sidecars"
        assert (sidecars / "clip_detection.mp4").exists()
        assert (sidecars / "clip_tracking.mp4").exists()
        assert (sidecars / "clip_tracking_red_id.mp4").exists()
        assert (sidecars / "clip_detection.mp4").stat().st_size > 0
        assert (sidecars / "clip_tracking.mp4").stat().st_size > 0
        assert (sidecars / "clip_tracking_red_id.mp4").stat().st_size > 0


class TestNormalizeClassName:
    """Tests for the _normalize_class_name function."""

    def test_lowercase_conversion(self):
        """Test that uppercase letters are converted to lowercase."""
        assert _normalize_class_name("Person") == "person"
        assert _normalize_class_name("CAR") == "car"

    def test_strip_whitespace(self):
        """Test that leading and trailing whitespace is removed."""
        assert _normalize_class_name("  bicycle  ") == "bicycle"
        assert _normalize_class_name("\tdog\n") == "dog"

    def test_replace_spaces_with_underscores(self):
        """Test that spaces are replaced with underscores."""
        assert _normalize_class_name("traffic light") == "traffic_light"
        assert _normalize_class_name("fire hydrant") == "fire_hydrant"

    def test_combined_transformations(self):
        """Test that all transformations work together."""
        assert _normalize_class_name("  Traffic Light  ") == "traffic_light"
        assert _normalize_class_name("STOP SIGN") == "stop_sign"
        assert _normalize_class_name("  Parking Meter  ") == "parking_meter"

    def test_empty_string(self):
        """Test that empty string is handled correctly."""
        assert _normalize_class_name("") == ""
        assert _normalize_class_name("   ") == ""

    def test_already_normalized(self):
        """Test that already normalized names remain unchanged."""
        assert _normalize_class_name("person") == "person"
        assert _normalize_class_name("traffic_light") == "traffic_light"
