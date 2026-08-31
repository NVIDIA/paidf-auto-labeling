# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import override
from unittest.mock import call, patch

import pytest
from core.interfaces.task import TaskInterface
from core.models import DataEntry
from core.pipelines import LinearPipeline
from core.policy import EmptyOutputPolicy
from core.scene import scene_context_for_entry
from core.tasks import SequentialTask


class SimpleTask(SequentialTask):
    @override
    def run(self, data_entry: DataEntry) -> DataEntry:
        return data_entry


class WriteMarkerTask(SimpleTask):
    @override
    def run(self, data_entry: DataEntry) -> DataEntry:
        assert data_entry.data_path
        marker = Path(data_entry.data_path) / "marker.txt"
        marker.write_text("annotated", encoding="utf-8")
        return data_entry


class RecordMediaTask(WriteMarkerTask):
    observed_media_path: str | None

    def __init__(self) -> None:
        super().__init__()
        self.observed_media_path = None

    @override
    def run(self, data_entry: DataEntry) -> DataEntry:
        self.observed_media_path = data_entry.media_path
        return super().run(data_entry)


class RecordSceneContextTask(WriteMarkerTask):
    observed_media_id: str | None

    def __init__(self) -> None:
        super().__init__()
        self.observed_media_id = None

    @override
    def run(self, data_entry: DataEntry) -> DataEntry:
        self.observed_media_id = scene_context_for_entry(data_entry).media_id
        return super().run(data_entry)


class ReplaceMediaTask(WriteMarkerTask):
    @override
    def run(self, data_entry: DataEntry) -> DataEntry:
        super().run(data_entry)
        replacement = Path(data_entry.data_path) / "sidecars" / "enhanced.mp4"
        replacement.write_bytes(b"enhanced")
        return data_entry.model_copy(update={"media_path": str(replacement)})


class ReplaceWebmWithMp4Task(WriteMarkerTask):
    @override
    def run(self, data_entry: DataEntry) -> DataEntry:
        super().run(data_entry)
        replacement = Path(data_entry.data_path) / "sidecars" / "sr_output.mp4"
        replacement.write_bytes(b"vp9-mp4")
        return data_entry.model_copy(update={"media_path": str(replacement)})


def _local_media(tmp_path: Path) -> Path:
    media_path = tmp_path / "file.mp4"
    media_path.write_bytes(b"media")
    return media_path


def _local_image(tmp_path: Path) -> Path:
    media_path = tmp_path / "frame.png"
    media_path.write_bytes(b"image")
    return media_path


def _local_webm(tmp_path: Path) -> Path:
    media_path = tmp_path / "file.webm"
    media_path.write_bytes(b"vp9-webm")
    return media_path


def _seed_downloaded_data(_remote_path: str, local_path: str, *, is_file: bool) -> str:
    assert not is_file
    destination = Path(local_path)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "seed.txt").write_text("seed", encoding="utf-8")
    return local_path


def _seed_downloaded_data_with_active(_remote_path: str, local_path: str, *, is_file: bool) -> str:
    assert not is_file
    destination = Path(local_path)
    destination.mkdir(parents=True, exist_ok=True)
    sidecars = destination / "sidecars"
    sidecars.mkdir()
    (sidecars / "active.mp4").write_bytes(b"remote-active")
    return local_path


def _download_remote_media(
    _remote_path: str,
    local_path: str | None = None,
    *,
    is_file: bool,
) -> str:
    assert is_file
    assert local_path is not None
    destination = Path(local_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"remote-media")
    return str(destination)


def _seed_downloaded_data_with_active_or_media(
    remote_path: str,
    local_path: str | None = None,
    *,
    is_file: bool,
) -> str:
    assert local_path is not None
    if is_file:
        return _download_remote_media(remote_path, local_path, is_file=True)
    return _seed_downloaded_data_with_active(remote_path, local_path, is_file=False)


def _assert_uploaded_data(local_path: str, remote_path: str) -> str:
    data_path = Path(local_path)
    assert (data_path / "seed.txt").read_text(encoding="utf-8") == "seed"
    assert (data_path / "marker.txt").read_text(encoding="utf-8") == "annotated"
    assert (data_path / "sidecars" / "active.mp4").read_bytes() == b"media"
    assert (data_path / "sidecars" / "raw.mp4").read_bytes() == b"media"
    return remote_path


def test_linear_pipeline_stages_media_as_active_file(
    simple_task: SimpleTask,
    tmp_path: Path,
) -> None:
    media_path = _local_media(tmp_path)
    data_path = tmp_path / "data"

    pipeline = LinearPipeline(tasks=[simple_task], policy=EmptyOutputPolicy.FAIL)
    data_entries = [DataEntry(media_path=str(media_path), data_path=str(data_path))]
    annotated_data_entries = pipeline.run(data_entries)

    assert annotated_data_entries[0].media_path == str(media_path)
    assert annotated_data_entries[0].data_path == str(data_path)
    assert (data_path / "sidecars" / "active.mp4").read_bytes() == b"media"
    assert (data_path / "sidecars" / "raw.mp4").read_bytes() == b"media"
    for dirname in ("raw", "contextual", "task", "sidecars"):
        assert (data_path / dirname).is_dir()


def test_linear_pipeline_stages_image_media_with_image_sidecars(
    simple_task: SimpleTask,
    tmp_path: Path,
) -> None:
    media_path = _local_image(tmp_path)
    data_path = tmp_path / "data"

    pipeline = LinearPipeline(tasks=[simple_task], policy=EmptyOutputPolicy.FAIL)
    data_entries = [DataEntry(media_path=str(media_path), data_path=str(data_path))]
    annotated_data_entries = pipeline.run(data_entries)

    assert annotated_data_entries[0].media_path == str(media_path)
    assert annotated_data_entries[0].data_path == str(data_path)
    assert (data_path / "sidecars" / "active.png").read_bytes() == b"image"
    assert (data_path / "sidecars" / "raw.png").read_bytes() == b"image"
    assert not (data_path / "sidecars" / "raw.mp4").exists()


def test_linear_pipeline_promotes_changed_container_suffix(tmp_path: Path) -> None:
    media_path = _local_webm(tmp_path)
    data_path = tmp_path / "data"

    pipeline = LinearPipeline(tasks=[ReplaceWebmWithMp4Task()], policy=EmptyOutputPolicy.FAIL)
    pipeline.run([DataEntry(media_path=str(media_path), data_path=str(data_path))])

    assert (data_path / "sidecars" / "active.mp4").read_bytes() == b"vp9-mp4"
    assert (data_path / "sidecars" / "raw.webm").read_bytes() == b"vp9-webm"
    assert not (data_path / "sidecars" / "active.webm").exists()


def test_prepare_input_raises_on_missing_media(simple_task: TaskInterface, tmp_path: Path) -> None:
    """Missing media files surface as FileNotFoundError when no active media exists."""
    pipeline = LinearPipeline(tasks=[simple_task], policy=EmptyOutputPolicy.FAIL)
    bad = DataEntry(
        media_path=str(tmp_path / "nope.mp4"),
        data_path=str(tmp_path / "nope_data"),
    )
    with pytest.raises(FileNotFoundError):
        pipeline.run([bad])


def test_existing_active_media_takes_precedence(tmp_path: Path) -> None:
    media_path = _local_media(tmp_path)
    data_path = tmp_path / "data"
    sidecars = data_path / "sidecars"
    sidecars.mkdir(parents=True)
    (sidecars / "active.mp4").write_bytes(b"active")
    task = RecordMediaTask()

    annotated_data_entries = LinearPipeline(tasks=[task]).run(
        [DataEntry(media_path=str(media_path), data_path=str(data_path))]
    )

    assert task.observed_media_path == str(data_path / "sidecars" / "active.mp4")
    assert (data_path / "sidecars" / "active.mp4").read_bytes() == b"active"
    assert (data_path / "sidecars" / "raw.mp4").read_bytes() == b"media"
    assert annotated_data_entries[0].media_path == str(media_path)
    assert annotated_data_entries[0].data_path == str(data_path)


def test_existing_raw_media_is_not_overwritten(tmp_path: Path) -> None:
    media_path = _local_media(tmp_path)
    data_path = tmp_path / "data"
    sidecars = data_path / "sidecars"
    sidecars.mkdir(parents=True)
    (sidecars / "raw.mp4").write_bytes(b"existing-raw")

    LinearPipeline(tasks=[SimpleTask()]).run(
        [DataEntry(media_path=str(media_path), data_path=str(data_path))]
    )

    assert (sidecars / "raw.mp4").read_bytes() == b"existing-raw"
    assert (sidecars / "active.mp4").read_bytes() == b"media"


def test_existing_raw_media_seeds_active_when_original_media_is_missing(tmp_path: Path) -> None:
    media_path = tmp_path / "missing.mp4"
    data_path = tmp_path / "data"
    sidecars = data_path / "sidecars"
    sidecars.mkdir(parents=True)
    (sidecars / "raw.mp4").write_bytes(b"existing-raw")

    annotated_data_entries = LinearPipeline(tasks=[SimpleTask()]).run(
        [DataEntry(media_path=str(media_path), data_path=str(data_path))]
    )

    assert annotated_data_entries[0].media_path == str(media_path)
    assert annotated_data_entries[0].data_path == str(data_path)
    assert (sidecars / "raw.mp4").read_bytes() == b"existing-raw"
    assert (sidecars / "active.mp4").read_bytes() == b"existing-raw"


def test_task_returned_media_is_promoted_to_active_for_downstream_tasks(tmp_path: Path) -> None:
    media_path = _local_media(tmp_path)
    data_path = tmp_path / "data"
    downstream = RecordMediaTask()

    annotated_data_entries = LinearPipeline(tasks=[ReplaceMediaTask(), downstream]).run(
        [DataEntry(media_path=str(media_path), data_path=str(data_path))]
    )

    assert downstream.observed_media_path == str(data_path / "sidecars" / "active.mp4")
    assert (data_path / "sidecars" / "active.mp4").read_bytes() == b"enhanced"
    assert (data_path / "sidecars" / "raw.mp4").read_bytes() == b"media"
    assert annotated_data_entries[0].media_path == str(media_path)
    assert annotated_data_entries[0].data_path == str(data_path)


def test_linear_pipeline_remote_data_uploads_daft_directory(tmp_path: Path) -> None:
    media_path = _local_media(tmp_path)
    data_tmp = tmp_path / "data_tmp"
    remote_data = "s3://bucket/data"

    pipeline = LinearPipeline(tasks=[WriteMarkerTask()])

    with (
        patch.object(
            pipeline.storage,
            "download_if_remote",
            side_effect=_seed_downloaded_data,
        ) as download_if_remote,
        patch.object(
            pipeline.storage,
            "upload_if_remote",
            side_effect=_assert_uploaded_data,
        ) as upload_if_remote,
        patch("core.interfaces.pipeline.tempfile.mkdtemp", return_value=str(data_tmp)),
    ):
        annotated_data_entries = pipeline.run(
            [DataEntry(media_path=str(media_path), data_path=remote_data)]
        )

    assert annotated_data_entries[0].media_path == str(media_path)
    assert annotated_data_entries[0].data_path == remote_data
    download_if_remote.assert_called_once_with(remote_data, str(data_tmp), is_file=False)
    upload_if_remote.assert_called_once_with(str(data_tmp), remote_data)
    assert not data_tmp.exists()


def test_remote_data_staging_preserves_original_scene_media_id(tmp_path: Path) -> None:
    media_path = _local_media(tmp_path)
    data_tmp = tmp_path / "pipeline_data_tmp"
    remote_data = "s3://bucket/scenes/clip_001/"
    task = RecordSceneContextTask()
    pipeline = LinearPipeline(tasks=[task])

    with (
        patch.object(
            pipeline.storage,
            "download_if_remote",
            side_effect=_seed_downloaded_data,
        ),
        patch.object(pipeline.storage, "upload_if_remote"),
        patch("core.interfaces.pipeline.tempfile.mkdtemp", return_value=str(data_tmp)),
    ):
        pipeline.run([DataEntry(media_path=str(media_path), data_path=remote_data)])

    assert task.observed_media_id == "clip_001"
    assert not data_tmp.exists()


def test_existing_remote_active_media_avoids_media_download(tmp_path: Path) -> None:
    data_tmp = tmp_path / "data_tmp"
    remote_media = "s3://bucket/file.mp4"
    remote_data = "s3://bucket/data"
    task = RecordMediaTask()
    pipeline = LinearPipeline(tasks=[task])

    with (
        patch.object(
            pipeline.storage,
            "download_if_remote",
            side_effect=_seed_downloaded_data_with_active_or_media,
        ) as download_if_remote,
        patch.object(pipeline.storage, "upload_if_remote"),
        patch("core.interfaces.pipeline.tempfile.mkdtemp", return_value=str(data_tmp)),
    ):
        annotated_data_entries = pipeline.run(
            [DataEntry(media_path=remote_media, data_path=remote_data)]
        )

    assert task.observed_media_path == str(data_tmp / "sidecars" / "active.mp4")
    assert annotated_data_entries[0].media_path == remote_media
    download_if_remote.assert_has_calls(
        [
            call(remote_data, str(data_tmp), is_file=False),
            call(remote_media, str(data_tmp / "sidecars" / "raw.mp4"), is_file=True),
        ]
    )
    assert not data_tmp.exists()


def test_remote_media_is_downloaded_to_active_when_data_has_no_active(tmp_path: Path) -> None:
    data_path = tmp_path / "data"
    remote_media = "s3://bucket/file.mp4"
    task = RecordMediaTask()
    pipeline = LinearPipeline(tasks=[task])

    with patch.object(
        pipeline.storage,
        "download_if_remote",
        side_effect=_download_remote_media,
    ) as download_if_remote:
        annotated_data_entries = pipeline.run(
            [DataEntry(media_path=remote_media, data_path=str(data_path))]
        )

    assert task.observed_media_path == str(data_path / "sidecars" / "active.mp4")
    assert (data_path / "sidecars" / "active.mp4").read_bytes() == b"remote-media"
    assert (data_path / "sidecars" / "raw.mp4").read_bytes() == b"remote-media"
    assert annotated_data_entries[0].media_path == remote_media
    download_if_remote.assert_called_once_with(
        remote_media,
        str(data_path / "sidecars" / "raw.mp4"),
        is_file=True,
    )


def test_linear_pipeline_skipped_entries_cleanup_temp_without_uploading(tmp_path: Path) -> None:
    media_path = _local_media(tmp_path)
    data_tmp = tmp_path / "data_tmp"
    remote_data = "s3://bucket/data"

    pipeline = LinearPipeline(tasks=[SimpleTask()])

    with (
        patch.object(
            pipeline.storage,
            "download_if_remote",
            side_effect=_seed_downloaded_data,
        ) as download_if_remote,
        patch.object(pipeline.storage, "upload_if_remote") as upload_if_remote,
        patch("core.interfaces.pipeline.tempfile.mkdtemp", return_value=str(data_tmp)),
        patch.object(SimpleTask, "run_batch", return_value=[]),
    ):
        annotated_data_entries = pipeline.run(
            [DataEntry(media_path=str(media_path), data_path=remote_data)]
        )

    assert annotated_data_entries == []
    download_if_remote.assert_called_once_with(remote_data, str(data_tmp), is_file=False)
    upload_if_remote.assert_not_called()
    assert not data_tmp.exists()


def test_prepare_input_cleans_temp_dir_when_entry_storage_fails(tmp_path: Path) -> None:
    data_tmp = tmp_path / "data_tmp"
    remote_data = "s3://bucket/data"

    pipeline = LinearPipeline(tasks=[SimpleTask()])

    with (
        patch.object(
            pipeline.storage,
            "download_if_remote",
            side_effect=_seed_downloaded_data,
        ),
        patch("core.interfaces.pipeline.tempfile.mkdtemp", return_value=str(data_tmp)),
        pytest.raises(FileNotFoundError),
    ):
        pipeline.run(
            [
                DataEntry(
                    media_path=str(tmp_path / "missing.mp4"),
                    data_path=remote_data,
                )
            ]
        )

    assert not data_tmp.exists()


def test_prepare_input_cleans_all_temp_dirs_when_partial_batch_fails(tmp_path: Path) -> None:
    media_path = _local_media(tmp_path)
    remote_data = "s3://bucket/data"
    created_temp_dirs: list[Path] = []

    def _make_temp_dir(prefix: str) -> str:
        temp_dir = tmp_path / f"{prefix}{len(created_temp_dirs)}"
        temp_dir.mkdir()
        created_temp_dirs.append(temp_dir)
        return str(temp_dir)

    pipeline = LinearPipeline(tasks=[SimpleTask()])

    with (
        patch.object(
            pipeline.storage,
            "download_if_remote",
            side_effect=_seed_downloaded_data,
        ),
        patch("core.interfaces.pipeline.tempfile.mkdtemp", side_effect=_make_temp_dir),
        pytest.raises(FileNotFoundError),
    ):
        pipeline.prepare_input(
            [
                DataEntry(
                    id="entry-1",
                    media_path=str(media_path),
                    data_path=remote_data,
                ),
                DataEntry(
                    id="entry-2",
                    media_path=str(tmp_path / "missing.mp4"),
                    data_path=remote_data,
                ),
                DataEntry(
                    id="entry-3",
                    media_path=str(media_path),
                    data_path=remote_data,
                ),
            ]
        )

    assert created_temp_dirs
    assert not list(tmp_path.glob("pipeline_data*"))
    assert pipeline._storage_state_by_entry_id == {}
