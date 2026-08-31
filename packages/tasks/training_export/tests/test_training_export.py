# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import shutil
from pathlib import Path

import pytest
from core import DataEntry
from training_export import TrainingExportConfig, TrainingExportTask, run_training_exports


def test_training_export_writes_configured_formats(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"source")
    scene = _scene_with_mcq(tmp_path / "scene")
    entry = DataEntry(media_path=str(media), data_path=str(scene))
    config = TrainingExportConfig(
        formats=("cosmos-reason-v1.0", "tao-vl-reason-v1.0"),
        output_dir=tmp_path / "exports",
        task_types=("mcq",),
        metadata={"description": "reasoning export", "license": "internal", "tags": ["traffic"]},
    )

    results = run_training_exports([entry], config, logger=logging.getLogger("test"))

    assert [result.samples_written for result in results] == [1, 1]
    cosmos_meta = json.loads(
        (tmp_path / "exports" / "cosmos-reason-v1.0" / "meta.json").read_text(encoding="utf-8")
    )
    tao_annotation = json.loads(
        (tmp_path / "exports" / "tao-vl-reason-v1.0" / "mcq.json").read_text(encoding="utf-8")
    )
    assert cosmos_meta["metadata"]["description"] == "reasoning export"
    assert cosmos_meta["samples"][0]["media"] == "media/raw--clip.mp4"
    assert tao_annotation["metadata"]["license"] == "internal"
    assert tao_annotation["items"][0]["video_id"] == "videos/raw--clip.mp4"


def test_training_export_task_returns_entries_when_disabled(tmp_path: Path) -> None:
    entry = DataEntry(media_path=str(tmp_path / "clip.mp4"), data_path=str(tmp_path / "scene"))
    task = TrainingExportTask()

    assert task.run_batch([entry]) == [entry]
    assert task.last_results == ()


def test_training_export_rejects_unknown_task_type(tmp_path: Path) -> None:
    scene = _scene_with_mcq(tmp_path / "scene")
    entry = DataEntry(media_path=str(tmp_path / "clip.mp4"), data_path=str(scene))
    task = TrainingExportTask(
        config=TrainingExportConfig(
            formats=("cosmos-reason-v1.0",),
            output_dir=tmp_path / "exports",
            task_types=("unknown_task",),
        )
    )

    with pytest.raises(ValueError, match="Unsupported training export task"):
        task.run_batch([entry])


def test_training_export_stages_remote_scene_and_uploads_remote_output(tmp_path: Path) -> None:
    source_scene = _scene_with_mcq(tmp_path / "remote_scene")
    remote_output = tmp_path / "remote_output"
    storage = _FakeStorage(
        remote_dirs={"s3://bucket/scenes/scene-a": source_scene},
        remote_uploads={"s3://bucket/exports": remote_output},
    )
    entry = DataEntry(
        media_path="s3://bucket/media/clip.mp4", data_path="s3://bucket/scenes/scene-a"
    )
    config = TrainingExportConfig(
        formats=("cosmos-reason-v1.0", "tao-vl-reason-v1.0"),
        output_dir="s3://bucket/exports",
        task_types=("mcq",),
    )

    results = run_training_exports(
        [entry],
        config,
        logger=logging.getLogger("test"),
        storage=storage,
    )

    assert [result.samples_written for result in results] == [1, 1]
    assert storage.downloads == [("s3://bucket/scenes/scene-a", False)]
    assert storage.uploads == [("s3://bucket/exports", False)]
    cosmos_meta = json.loads(
        (remote_output / "cosmos-reason-v1.0" / "meta.json").read_text(encoding="utf-8")
    )
    tao_annotation = json.loads(
        (remote_output / "tao-vl-reason-v1.0" / "mcq.json").read_text(encoding="utf-8")
    )
    assert cosmos_meta["samples"][0]["media"] == "media/raw--clip.mp4"
    assert tao_annotation["items"][0]["video_id"] == "videos/raw--clip.mp4"


def test_training_export_rejects_remote_scene_without_media_copy(tmp_path: Path) -> None:
    source_scene = _scene_with_mcq(tmp_path / "remote_scene")
    storage = _FakeStorage(remote_dirs={"s3://bucket/scenes/scene-a": source_scene})
    entry = DataEntry(
        media_path="s3://bucket/media/clip.mp4", data_path="s3://bucket/scenes/scene-a"
    )
    config = TrainingExportConfig(
        formats=("cosmos-reason-v1.0",),
        output_dir=tmp_path / "exports",
        copy_media=False,
    )

    with pytest.raises(ValueError, match="requires copy_media=True"):
        run_training_exports([entry], config, logger=logging.getLogger("test"), storage=storage)


def _scene_with_mcq(scene: Path) -> Path:
    raw_dir = scene / "raw"
    contextual_dir = scene / "contextual"
    task_dir = scene / "task"
    raw_dir.mkdir(parents=True)
    contextual_dir.mkdir()
    task_dir.mkdir()
    (raw_dir / "clip.mp4").write_bytes(b"video")
    _write_json(
        contextual_dir / "video.json",
        {
            "version": "metropolis-v3.0",
            "video_id": "clip",
            "metadata": {"type": "video"},
            "format": "mp4",
            "fps": 30,
            "duration": 1.0,
            "height": 720,
            "width": 1280,
        },
    )
    _write_json(
        task_dir / "mcq.json",
        {
            "version": "metropolis-v3.0",
            "metadata": {"type": "mcq"},
            "items": [
                {
                    "video_id": "clip",
                    "question": "Which object moves?",
                    "options": {"A": "Car", "B": "Bike"},
                    "answer": "A",
                }
            ],
        },
    )
    return scene


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class _FakeStorage:
    def __init__(
        self,
        *,
        remote_dirs: dict[str, Path],
        remote_uploads: dict[str, Path] | None = None,
    ) -> None:
        self.remote_dirs = remote_dirs
        self.remote_uploads = remote_uploads or {}
        self.downloads: list[tuple[str, bool]] = []
        self.uploads: list[tuple[str, bool]] = []

    def is_remote_storage_url(self, url: str) -> bool:
        return url.startswith("s3://")

    def download_if_remote(
        self,
        path: str,
        local_path: str | None = None,
        *,
        is_file: bool,
    ) -> str:
        self.downloads.append((path, is_file))
        if is_file:
            raise AssertionError("training export should download scene directories")
        if local_path is None:
            raise AssertionError("training export should provide a staging directory")
        shutil.copytree(self.remote_dirs[path], local_path, dirs_exist_ok=True)
        return local_path

    def upload_if_remote(
        self,
        local_path: str,
        remote_path: str,
        *,
        delete_unmatched_files: bool = False,
    ) -> str:
        self.uploads.append((remote_path, delete_unmatched_files))
        shutil.copytree(local_path, self.remote_uploads[remote_path], dirs_exist_ok=True)
        return remote_path
