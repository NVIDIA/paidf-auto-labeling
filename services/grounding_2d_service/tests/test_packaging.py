# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packaging contracts for independent, orchestration-friendly GPU images."""

from __future__ import annotations

import tomllib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SERVICE_PYPROJECT = _REPO_ROOT / "services" / "grounding_2d_service" / "pyproject.toml"
_DETECTION_DOCKERFILE = (
    _REPO_ROOT / "services" / "detection_and_tracking_service" / "docker" / "Dockerfile"
)
_GROUNDING_GPU_DOCKERFILE = (
    _REPO_ROOT / "services" / "grounding_2d_service" / "docker" / "Dockerfile.gpu"
)
_LEGACY_SIBLING_DOCKERFILE = (
    _REPO_ROOT / "services" / "grounding_2d_service" / "docker" / "Dockerfile.sam3"
)
_LEGACY_ROOT_SHARED_DOCKERFILE = _REPO_ROOT / "docker" / "Dockerfile.pytorch-gpu"
_LEGACY_ROOT_SHARED_RENAME = _REPO_ROOT / "docker" / "Dockerfile.shared-gpu-service"


def _build_images() -> dict[str, str]:
    with _SERVICE_PYPROJECT.open("rb") as handle:
        data = tomllib.load(handle)
    images = data.get("tool", {}).get("build", {}).get("images", {})
    assert isinstance(images, dict)
    return {str(key): str(value) for key, value in images.items()}


def test_detection_service_owns_real_gpu_dockerfile() -> None:
    assert _DETECTION_DOCKERFILE.is_file()
    assert not _DETECTION_DOCKERFILE.is_symlink()
    text = _DETECTION_DOCKERFILE.read_text(encoding="utf-8")
    assert "ARG PACKAGE_NAME" in text
    assert "ARG RUNTIME_FLAVOR" in text


def test_grounding_gpu_dockerfile_lives_under_service_docker() -> None:
    assert _GROUNDING_GPU_DOCKERFILE.is_file()
    assert not _GROUNDING_GPU_DOCKERFILE.is_symlink()
    text = _GROUNDING_GPU_DOCKERFILE.read_text(encoding="utf-8")
    assert "ARG PACKAGE_NAME" in text
    assert "ARG RUNTIME_FLAVOR" in text
    assert "FROM detection-and-tracking" not in text


def test_legacy_root_shared_dockerfiles_removed() -> None:
    assert not _LEGACY_ROOT_SHARED_DOCKERFILE.exists()
    assert not _LEGACY_ROOT_SHARED_RENAME.exists()


def test_legacy_sibling_from_dockerfile_removed() -> None:
    assert not _LEGACY_SIBLING_DOCKERFILE.exists()


def test_sam3_image_build_is_independent_of_detection_service_image() -> None:
    images = _build_images()
    sam3_cmd = images["sam3"]

    assert "paidf-grounding-2d-sam3-service" in sam3_cmd
    assert "PACKAGE_NAME=grounding-2d-service" in sam3_cmd
    assert "RUNTIME_FLAVOR=sam3" in sam3_cmd
    assert "-f services/grounding_2d_service/docker/Dockerfile.gpu" in sam3_cmd
    assert "Dockerfile.sam3" not in sam3_cmd
    assert "detection-and-tracking-sam3-service" not in sam3_cmd
    assert "docker/Dockerfile.pytorch-gpu" not in sam3_cmd
    assert "docker/Dockerfile.shared-gpu-service" not in sam3_cmd


def test_main_image_remains_slim_cpu_path() -> None:
    images = _build_images()
    main_cmd = images["main"]

    assert "paidf-grounding-2d-service" in main_cmd
    assert "services/grounding_2d_service/docker/Dockerfile" in main_cmd
    assert "RUNTIME_FLAVOR" not in main_cmd
