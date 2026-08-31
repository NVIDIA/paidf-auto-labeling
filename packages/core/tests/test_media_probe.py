# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from core.utils.media_probe import probe_frame_count

# --- not a readable file ----------------------------------------------------


def test_probe_frame_count_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert probe_frame_count(tmp_path / "nope.mp4") is None


def test_probe_frame_count_returns_none_for_directory(tmp_path: Path) -> None:
    assert probe_frame_count(tmp_path) is None


# --- image branch -----------------------------------------------------------


def test_probe_frame_count_counts_image_as_single_frame(tmp_path: Path) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"not really a png, extension is enough")
    assert probe_frame_count(image) == 1


def test_probe_frame_count_image_extension_is_case_insensitive(tmp_path: Path) -> None:
    image = tmp_path / "frame.PNG"
    image.write_bytes(b"x")
    assert probe_frame_count(image) == 1


def test_probe_frame_count_accepts_str_path(tmp_path: Path) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"x")
    assert probe_frame_count(str(image)) == 1


# --- video branch -----------------------------------------------------------


def test_probe_frame_count_returns_none_for_video(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")

    assert probe_frame_count(video) is None
