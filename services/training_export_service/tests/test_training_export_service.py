# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest
from core import DataEntry
from training_export import TrainingExportConfig
from training_export_service.main import TrainingExportService, training_export_config_from_args


def test_container_installs_ffmpeg_without_unused_python_media_packages() -> None:
    dockerfile = Path(__file__).parents[1] / "docker" / "Dockerfile"
    text = dockerfile.read_text(encoding="utf-8")

    # FFmpeg comes from the shared input-only media base; the service no longer
    # compiles it and installs no OpenCV/PyAV wheels.
    assert "FROM ${BUILDER_BASE_IMAGE} AS builder" in text
    assert "ffmpeg-install" not in text
    assert text.count("media_toolchain.py verify") == 2
    for unused_media_component in ("opencv-python", "opencv-headless", "pyav", "find-links"):
        assert unused_media_component not in text.lower()


def test_training_export_service_config_from_args() -> None:
    parser = argparse.ArgumentParser()
    TrainingExportService().add_service_args(parser)
    args = parser.parse_args(
        [
            "--training-export-format",
            "cosmos-reason-v1.0",
            "--training-export-dir",
            "/exports",
            "--training-export-task",
            "mcq",
            "--training-export-description",
            "training split",
            "--training-export-license",
            "internal",
            "--training-export-tag",
            "traffic",
            "--training-export-no-copy-media",
            "--training-export-emit-media-root-as-null",
        ]
    )

    assert training_export_config_from_args(args) == TrainingExportConfig(
        formats=("cosmos-reason-v1.0",),
        output_dir="/exports",
        task_types=("mcq",),
        metadata={"description": "training split", "license": "internal", "tags": ["traffic"]},
        copy_media=False,
        emit_media_root_as_null=True,
    )


def test_training_export_service_runs_batch_task() -> None:
    service = TrainingExportService()
    parser = argparse.ArgumentParser()
    service.add_service_args(parser)
    args = parser.parse_args(
        [
            "--training-export-format",
            "cosmos-reason-v1.0",
            "--training-export-dir",
            "/exports",
        ]
    )
    entries = [
        DataEntry(media_path="/data/v1.mp4", data_path="/data/v1"),
        DataEntry(media_path="/data/v2.mp4", data_path="/data/v2"),
    ]

    with patch("training_export_service.main.TrainingExportTask") as task_cls:
        task = task_cls.return_value
        task.run_batch.return_value = entries

        service.execute(args, entries)

    task_cls.assert_called_once_with(
        config=TrainingExportConfig(
            formats=("cosmos-reason-v1.0",),
            output_dir="/exports",
        )
    )
    task.run_batch.assert_called_once_with(entries)


def test_training_export_service_requires_entries() -> None:
    parser = argparse.ArgumentParser()
    TrainingExportService().add_service_args(parser)
    args = parser.parse_args(
        [
            "--training-export-format",
            "cosmos-reason-v1.0",
            "--training-export-dir",
            "/exports",
        ]
    )

    with pytest.raises(SystemExit, match="at least one DataEntry"):
        TrainingExportService().execute(args, [])
