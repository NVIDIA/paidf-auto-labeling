# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for VideoPreprocessor.extract_frames and QwenVLMClient.analyze_frames.

Regression coverage for the max_frames=0 → max_frames=config.max_frames fix:
previously extract_frames dumped all frames to disk then analyze_frames
subsampled in-memory; now FFmpeg is capped at the source and analyze_frames
always sees num_frames <= max_frames.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest
from vlm_json.runners.video_pipeline import QwenVLMClient, VideoPreprocessor, VLMConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _preprocessor(max_frames: int = 24, frame_fps: float = 1.0) -> VideoPreprocessor:
    cfg = VLMConfig(max_frames=max_frames, frame_fps=frame_fps)
    return VideoPreprocessor(cfg, logging.getLogger("test"))


def _vlm_client(max_frames: int = 24, frame_fps: float = 1.0) -> QwenVLMClient:
    cfg = VLMConfig(max_frames=max_frames, frame_fps=frame_fps)
    return QwenVLMClient(cfg, logging.getLogger("test"))


def _fake_frames(n: int) -> list:
    """Return a list of n dummy (timestamp, path) tuples."""
    return [(float(i), Path(f"frame_{i:06d}.jpg")) for i in range(n)]


# ---------------------------------------------------------------------------
# extract_frames: max_frames propagated to sample_frames_ffmpeg
# ---------------------------------------------------------------------------


class TestExtractFramesMaxFramesPropagation:
    """extract_frames must pass config.max_frames (not 0) to sample_frames_ffmpeg."""

    def _run(self, tmp_path: Path, max_frames: int, duration: float = 120.0) -> dict:
        pre = _preprocessor(max_frames=max_frames)
        fake_video = tmp_path / "v.mp4"
        fake_video.write_bytes(b"")

        with (
            patch.object(pre, "get_video_info", return_value={"duration": duration, "height": 480}),
            patch(
                "vlm_json.runners.video_pipeline.sample_frames_ffmpeg",
                return_value=_fake_frames(min(max_frames, int(duration))),
            ) as mock_sfx,
        ):
            ok, info = pre.extract_frames(
                fake_video,
                tmp_path / "frames",
                fps=1.0,
                resolution=480,
            )

        return {"ok": ok, "info": info, "call_kwargs": mock_sfx.call_args.kwargs}

    def test_config_max_frames_forwarded(self, tmp_path):
        result = self._run(tmp_path, max_frames=50)
        assert result["call_kwargs"]["max_frames"] == 50

    def test_max_frames_not_zero(self, tmp_path):
        result = self._run(tmp_path, max_frames=100)
        assert result["call_kwargs"]["max_frames"] != 0

    def test_extraction_succeeds(self, tmp_path):
        result = self._run(tmp_path, max_frames=24)
        assert result["ok"] is True

    def test_custom_max_frames_forwarded(self, tmp_path):
        for cap in (1, 10, 200):
            result = self._run(tmp_path, max_frames=cap)
            assert result["call_kwargs"]["max_frames"] == cap, f"failed for cap={cap}"

    def test_sampling_fps_forwarded(self, tmp_path):
        pre = VideoPreprocessor(VLMConfig(max_frames=50, frame_fps=2.0), logging.getLogger("test"))
        fake_video = tmp_path / "v.mp4"
        fake_video.write_bytes(b"")

        with (
            patch.object(pre, "get_video_info", return_value={"duration": 30.0, "height": 480}),
            patch(
                "vlm_json.runners.video_pipeline.sample_frames_ffmpeg",
                return_value=_fake_frames(10),
            ) as mock_sfx,
        ):
            pre.extract_frames(fake_video, tmp_path / "frames", fps=2.0, resolution=480)

        assert mock_sfx.call_args.kwargs["sampling_fps"] == 2.0
        assert mock_sfx.call_args.kwargs["max_frames"] == 50


# ---------------------------------------------------------------------------
# analyze_frames: step=1 when num_frames <= max_frames (normal post-fix path)
# ---------------------------------------------------------------------------


class TestAnalyzeFramesStep:
    """After the fix, num_frames <= max_frames → step=1 → all frames sent to VLM."""

    def _run_analyze(
        self,
        tmp_path: Path,
        num_frames_on_disk: int,
        max_frames: int = 100,
        frame_fps: float = 1.0,
    ) -> dict:
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        # Write tiny placeholder jpegs.
        for i in range(num_frames_on_disk):
            p = frames_dir / f"frame_{i:06d}.jpg"
            # Minimal valid JPEG header so base64 encode works.
            p.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9")

        client = _vlm_client(max_frames=max_frames, frame_fps=frame_fps)
        captured: dict = {}

        real_messages = client._messages_from_frames

        def _capture_messages(frames_dir, prompt, *, frame_fps_param, video_fps_param):
            captured["frame_fps_param"] = frame_fps_param
            captured["video_fps_param"] = video_fps_param
            captured["step"] = max(1, int(frame_fps_param * video_fps_param))
            return real_messages(frames_dir, prompt, frame_fps_param=frame_fps_param, video_fps_param=video_fps_param)

        with (
            patch.object(client, "_messages_from_frames", side_effect=_capture_messages),
            patch(
                "vlm_json.runners.video_pipeline.call_chat_object_with_structured_fallback",
                return_value=(None, '```json\n{"events": []}\n```'),
            ),
        ):
            client.analyze_frames(frames_dir, "describe", tmp_path / "out")

        return captured

    def test_step_is_one_when_frames_lte_max(self, tmp_path):
        result = self._run_analyze(tmp_path, num_frames_on_disk=50, max_frames=100)
        assert result["step"] == 1

    def test_step_is_one_at_exact_limit(self, tmp_path):
        result = self._run_analyze(tmp_path, num_frames_on_disk=100, max_frames=100)
        assert result["step"] == 1

    def test_timestamps_use_extraction_fps(self, tmp_path):
        """frame_fps_param = 1/fps, video_fps_param = fps → sec = i/fps."""
        result = self._run_analyze(tmp_path, num_frames_on_disk=10, max_frames=100, frame_fps=2.0)
        assert result["video_fps_param"] == pytest.approx(2.0)
        assert result["frame_fps_param"] == pytest.approx(0.5)  # 1/fps

    def test_safety_net_step_gt_one_when_too_many_frames(self, tmp_path):
        """Safety-net branch: if somehow more frames land on disk, step > 1."""
        result = self._run_analyze(tmp_path, num_frames_on_disk=200, max_frames=100)
        assert result["step"] > 1

    def test_all_frames_included_when_lte_max(self, tmp_path):
        """With step=1, _messages_from_frames includes every frame."""
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        n = 5
        for i in range(n):
            p = frames_dir / f"frame_{i:06d}.jpg"
            p.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9")

        client = _vlm_client(max_frames=100, frame_fps=1.0)
        messages = client._messages_from_frames(frames_dir, "p", frame_fps_param=1.0, video_fps_param=1.0)

        # One text + one image per frame, plus final prompt text.
        image_items = [c for c in messages[0]["content"] if c.get("type") == "image_url"]
        assert len(image_items) == n
