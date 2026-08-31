# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path
from unittest.mock import patch

from example_service.main import main


def test_container_installs_ffmpeg_without_unused_python_media_packages() -> None:
    dockerfile = Path(__file__).parents[1] / "docker" / "Dockerfile"
    text = dockerfile.read_text(encoding="utf-8")

    # FFmpeg is prebuilt in the shared media base; this input-only service just
    # inherits it and never pulls in the OpenCV/PyAV output stack.
    assert "FROM ${BUILDER_BASE_IMAGE} AS builder" in text
    assert "ffmpeg-install" not in text
    assert text.count("media_toolchain.py verify") == 2
    for unused_media_component in ("opencv-python", "opencv-headless", "pyav", "media-wheels"):
        assert unused_media_component not in text.lower()


def test_example_service_creates_data_directories(tmp_path: Path) -> None:
    """
    Test that the service creates new data directories as expected.
    """
    # Prepare input payload: media_path, but data_path does not exist yet
    media_path = tmp_path / "file.mp4"
    media_path.touch()
    data_dir = tmp_path / "data_dir"
    assert not data_dir.exists()
    payload = json.dumps([{"media_path": str(media_path), "data_path": str(data_dir)}])

    argv = ["example-service", "--input", payload]
    with patch.object(sys, "argv", argv):
        main()

    # After running the service, the data_path/directory should be created and seeded
    # with the active media handoff file.
    assert data_dir.exists() and data_dir.is_dir()
    assert (data_dir / "sidecars" / "active.mp4").exists()
    assert (data_dir / "sidecars" / "raw.mp4").exists()


def test_example_service_dev_data_root_copies_before_annotating(tmp_path: Path) -> None:
    media_path = tmp_path / "file.mp4"
    media_path.write_bytes(b"media")
    source_data_dir = tmp_path / "source_data"
    source_data_dir.mkdir()
    (source_data_dir / "seed.txt").write_text("seed", encoding="utf-8")
    dev_root = tmp_path / "dev_data"
    payload = json.dumps(
        [{"id": "entry-1", "media_path": str(media_path), "data_path": str(source_data_dir)}]
    )

    argv = ["example-service", "--dev-data-root", str(dev_root), "--input", payload]
    with patch.object(sys, "argv", argv):
        main()

    copied_data_dir = dev_root / "entry-1"
    assert (source_data_dir / "seed.txt").read_text(encoding="utf-8") == "seed"
    assert not (source_data_dir / "sidecars" / "active.mp4").exists()
    assert (copied_data_dir / "seed.txt").read_text(encoding="utf-8") == "seed"
    assert (copied_data_dir / "sidecars" / "active.mp4").read_bytes() == b"media"
    assert (copied_data_dir / "sidecars" / "raw.mp4").read_bytes() == b"media"
    assert (copied_data_dir / "sidecars" / "example_task.json").exists()


def test_example_service_dev_data_root_allows_missing_source_data_dir(
    tmp_path: Path,
) -> None:
    media_path = tmp_path / "file.mp4"
    media_path.write_bytes(b"media")
    missing_data_dir = tmp_path / "missing_data"
    dev_root = tmp_path / "dev_data"
    payload = json.dumps(
        [{"id": "entry-1", "media_path": str(media_path), "data_path": str(missing_data_dir)}]
    )

    argv = ["example-service", "--dev-data-root", str(dev_root), "--input", payload]
    with patch.object(sys, "argv", argv):
        main()

    copied_data_dir = dev_root / "entry-1"
    assert not missing_data_dir.exists()
    assert (copied_data_dir / "sidecars" / "active.mp4").read_bytes() == b"media"
    assert (copied_data_dir / "sidecars" / "raw.mp4").read_bytes() == b"media"
    assert (copied_data_dir / "sidecars" / "example_task.json").exists()
