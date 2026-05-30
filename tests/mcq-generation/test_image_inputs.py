# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path

from mcq_generation.mcq.utils.video import extract_frames, probe_video
from PIL import Image


def _write_test_image(path: Path, *, w: int = 64, h: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", (w, h), color=(10, 20, 30))
    im.save(path)


def test_probe_video_supports_image(tmp_path: Path) -> None:
    img = tmp_path / "x.png"
    _write_test_image(img, w=64, h=32)

    info = probe_video(img)
    assert info.width == 64
    assert info.height == 32
    assert info.fps == 1.0
    assert info.num_frames == 1
    assert info.duration_sec == 1.0


def test_extract_frames_supports_image(tmp_path: Path) -> None:
    img = tmp_path / "x.png"
    _write_test_image(img, w=64, h=32)

    out_dir = tmp_path / "frames"
    logger = logging.getLogger("test")
    items = extract_frames(
        video_path=img,
        out_dir=out_dir,
        start_sec=0.0,
        end_sec=1.0,
        sampling_fps=2.0,
        resolution=32,
        max_frames=10,
        logger=logger,
    )
    assert len(items) == 1
    t0, fp = items[0]
    assert t0 == 0.0
    assert fp.exists()
    assert fp.name == "frame_000001.jpg"

    # Best-effort validate we wrote a readable JPEG.
    with Image.open(fp) as im:
        assert im.size[1] == 32
