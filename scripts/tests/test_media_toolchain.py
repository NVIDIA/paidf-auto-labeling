# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tomllib
from pathlib import Path

import media_toolchain
import pytest


def test_manifest_pins_all_downloaded_media_sources() -> None:
    manifest_path = Path(__file__).parents[1] / "media-toolchain.toml"

    with manifest_path.open("rb") as stream:
        manifest = tomllib.load(stream)

    for section_name in ("ffmpeg", "nv_codec_headers", "libvpx", "pyav", "opencv_headless"):
        section = manifest[section_name]
        assert section["version"]
        assert len(section["sha256"]) == 64
    assert manifest["ffmpeg"]["url"].startswith("https://")
    assert manifest["nv_codec_headers"]["url"].startswith("https://")
    assert manifest["libvpx"]["url"].startswith("https://")
    assert all("==" in item for item in manifest["pyav"]["build_requirements"])
    assert all("==" in item for item in manifest["opencv_headless"]["build_requirements"])
    assert any(
        item.startswith("numpy==") for item in manifest["opencv_headless"]["build_requirements"]
    )


def test_verify_fails_closed_when_ffmpeg_is_missing(
    tmp_path: Path,
) -> None:
    result = media_toolchain.main(
        [
            "verify",
            "--profile",
            "vp9-output",
            "--site-packages",
            str(tmp_path),
            "--ffmpeg",
            str(tmp_path / "missing-ffmpeg"),
        ]
    )

    assert result == 1


@pytest.mark.parametrize("jobs", ["0", "-1"])
def test_ffmpeg_install_rejects_non_positive_jobs(jobs: str) -> None:
    with pytest.raises(SystemExit):
        media_toolchain._parser().parse_args(
            ["ffmpeg-install", "--profile", "vp9-output", "--jobs", jobs]
        )


def test_python_install_passes_wheel_dir_to_build_script(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: list[dict[str, str]] = []
    monkeypatch.setattr(
        media_toolchain, "_run_script", lambda script, arguments, env: captured.append(env)
    )

    media_toolchain.main(
        [
            "python-install",
            "--python",
            str(tmp_path / "python"),
            "--wheel-dir",
            str(tmp_path / "wheels"),
            "--component",
            "pyav",
        ]
    )

    assert captured and captured[0]["MEDIA_WHEEL_DIR"] == str(tmp_path / "wheels")


def test_python_install_omits_wheel_dir_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: list[dict[str, str]] = []
    monkeypatch.setattr(
        media_toolchain, "_run_script", lambda script, arguments, env: captured.append(env)
    )

    media_toolchain.main(
        [
            "python-install",
            "--python",
            str(tmp_path / "python"),
            "--component",
            "opencv-headless",
        ]
    )

    assert captured and "MEDIA_WHEEL_DIR" not in captured[0]


def test_pilot_services_consume_shared_media_base() -> None:
    root = Path(__file__).parents[2]
    example = (root / "services/example_service/docker/Dockerfile").read_text(encoding="utf-8")
    captioning = (root / "services/captioning_service/docker/Dockerfile").read_text(
        encoding="utf-8"
    )

    # Both builders start from the shared media base instead of recompiling FFmpeg.
    for text in (example, captioning):
        assert "FROM ${BUILDER_BASE_IMAGE} AS builder" in text
        assert "ffmpeg-install" not in text
    # The vp9 service installs the prebuilt wheels rather than recompiling OpenCV/PyAV.
    assert "UPA_MEDIA_WHEEL_DIR" in captioning
    assert "python-install" not in captioning
    # The final runtime image still verifies the codec policy surface.
    assert "media_toolchain.py verify" in captioning


def test_migrated_media_services_use_shared_toolchain_and_verify_final_image() -> None:
    repository_root = Path(__file__).parents[2]
    services = (
        "detection_and_tracking_service",
        "super_resolution_service",
    )

    for service in services:
        dockerfile = repository_root / "services" / service / "docker" / "Dockerfile"
        text = dockerfile.read_text(encoding="utf-8")
        assert "media_toolchain.py ffmpeg-install --profile vp9-output" in text
        assert "source=/opt/media-runtime" in text
        assert text.count("media_toolchain.py verify") == 2
        assert "h264_nvenc" not in text
