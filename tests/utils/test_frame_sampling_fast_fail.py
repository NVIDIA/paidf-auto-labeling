# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest
from al_utils.media_decode import h264_stream_requires_unsupported_decoder


def test_h264_stream_requires_unsupported_decoder_for_non_420_profile() -> None:
    assert h264_stream_requires_unsupported_decoder(
        {
            "codec_type": "video",
            "codec_name": "h264",
            "profile": "High 4:4:4 Predictive",
            "pix_fmt": "yuv444p",
        }
    )


def test_h264_stream_allows_420_compatible_profile() -> None:
    assert not h264_stream_requires_unsupported_decoder(
        {
            "codec_type": "video",
            "codec_name": "h264",
            "profile": "Main",
            "pix_fmt": "yuv420p",
        }
    )


def test_sample_frames_ffmpeg_fast_fails_unsupported_h264_profile(tmp_path: Path) -> None:
    pytest.importorskip("ffmpeg")
    from mcq_generation.mcq.utils.frame_sampling import sample_frames_ffmpeg

    p = tmp_path / "bad_h264.mp4"
    p.write_bytes(b"not used")
    out_dir = tmp_path / "frames"
    out_dir.mkdir()

    with (
        patch(
            "mcq_generation.mcq.utils.frame_sampling.ffmpeg.probe",
            return_value={
                "streams": [
                    {
                        "codec_type": "video",
                        "codec_name": "h264",
                        "profile": "High 4:4:4 Predictive",
                        "pix_fmt": "yuv444p",
                    }
                ]
            },
        ),
        patch("mcq_generation.mcq.utils.frame_sampling.ffmpeg.input") as mock_input,
        pytest.raises(RuntimeError, match="h264_cuvid CUDA_ERROR_NOT_SUPPORTED"),
    ):
        sample_frames_ffmpeg(
            video_path=p,
            out_dir=out_dir,
            start_sec=0.0,
            end_sec=2.0,
            sampling_fps=1.0,
            resolution=32,
            max_frames=3,
            logger=logging.getLogger("test"),
        )

    mock_input.assert_not_called()
