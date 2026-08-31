# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from super_resolution.media_formats import (
    is_supported_sr_image_path,
    is_supported_sr_media_path,
    is_supported_sr_video_path,
    sr_output_name_for_input,
    sr_supported_extensions_message,
)


@pytest.mark.parametrize("path", ["frame.jpg", "frame.JPEG", "frame.png", "frame.webp"])
def test_supported_sr_image_extensions(path: str) -> None:
    assert is_supported_sr_image_path(path) is True
    assert is_supported_sr_media_path(path) is True
    assert is_supported_sr_video_path(path) is False


@pytest.mark.parametrize("path", ["clip.mp4", "clip.MOV", "clip.webm"])
def test_supported_sr_video_extensions(path: str) -> None:
    assert is_supported_sr_video_path(path) is True
    assert is_supported_sr_media_path(path) is True
    assert is_supported_sr_image_path(path) is False


@pytest.mark.parametrize("path", ["clip.mkv", "clip.avi", "clip.m4v", "frame.tiff"])
def test_unsupported_sr_extensions(path: str) -> None:
    assert is_supported_sr_media_path(path) is False


def test_sr_output_name_uses_mp4_for_all_videos() -> None:
    assert sr_output_name_for_input("clip.MOV") == "sr_output.mp4"
    assert sr_output_name_for_input("clip.webm") == "sr_output.mp4"
    assert sr_output_name_for_input("frame.PNG") == "sr_output.png"
    assert sr_output_name_for_input("clip.mp4", stem="clip") == "clip.mp4"


def test_sr_output_name_rejects_unsupported_suffix() -> None:
    with pytest.raises(ValueError, match="Unsupported super-resolution media extension '.mkv'"):
        sr_output_name_for_input("clip.mkv")
    assert ".mp4" in sr_supported_extensions_message()
