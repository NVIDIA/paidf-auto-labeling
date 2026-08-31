# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import override
from unittest.mock import patch

import pytest
from core.interfaces.pipeline import PipelineInterface, _PipelineStorageState
from core.models import DataEntry


class _PipelineHarness(PipelineInterface):
    @override
    def run(self, data_entries: list[DataEntry]) -> list[DataEntry]:
        return self.prepare_output(self.prepare_input(data_entries))


def _media_file(tmp_path: Path) -> Path:
    media_path = tmp_path / "input.mp4"
    media_path.write_bytes(b"media")
    return media_path


def _storage_state(
    working_media_path: Path,
    working_data_path: Path,
    temp_dirs: list[Path] | None = None,
) -> _PipelineStorageState:
    return _PipelineStorageState(
        original_media_path="original.mp4",
        working_media_path=str(working_media_path),
        original_data_path="original-data",
        working_data_path=str(working_data_path),
        final_data_path="final-data",
        temp_dirs=[] if temp_dirs is None else temp_dirs,
    )


def test_prepare_input_rejects_duplicate_entry_ids(tmp_path: Path) -> None:
    media_path = _media_file(tmp_path)
    pipeline = _PipelineHarness()

    with pytest.raises(ValueError, match="Duplicate data entry id"):
        pipeline.prepare_input(
            [
                DataEntry(id="same", media_path=str(media_path), data_path=str(tmp_path / "a")),
                DataEntry(id="same", media_path=str(media_path), data_path=str(tmp_path / "b")),
            ]
        )


def test_prepare_input_rejects_missing_prepared_media(tmp_path: Path) -> None:
    pipeline = _PipelineHarness()
    data_path = tmp_path / "data"
    temp_dir = tmp_path / "pipeline_data_validation"
    temp_dir.mkdir()

    def _prepare_entry_storage(
        _data_entry: DataEntry,
        temp_dirs: list[Path],
    ) -> _PipelineStorageState:
        temp_dirs.append(temp_dir)
        return _storage_state(tmp_path / "missing.mp4", data_path, temp_dirs)

    with (
        patch.object(pipeline, "_prepare_entry_storage", side_effect=_prepare_entry_storage),
        pytest.raises(FileNotFoundError, match="Media file"),
    ):
        pipeline.prepare_input([DataEntry(media_path="input.mp4", data_path="data")])

    assert not temp_dir.exists()
    assert pipeline._storage_state_by_entry_id == {}


def test_prepare_input_rejects_prepared_media_directory(tmp_path: Path) -> None:
    pipeline = _PipelineHarness()
    media_path = tmp_path / "media-dir"
    media_path.mkdir()
    state = _storage_state(media_path, tmp_path / "data")

    with (
        patch.object(pipeline, "_prepare_entry_storage", return_value=state),
        pytest.raises(ValueError, match="not a file"),
    ):
        pipeline.prepare_input([DataEntry(media_path="input.mp4", data_path="data")])


def test_prepare_input_creates_missing_prepared_data_directory(tmp_path: Path) -> None:
    pipeline = _PipelineHarness()
    media_path = _media_file(tmp_path)
    data_path = tmp_path / "prepared-data"
    state = _storage_state(media_path, data_path)

    with patch.object(pipeline, "_prepare_entry_storage", return_value=state):
        prepared = pipeline.prepare_input([DataEntry(media_path="input.mp4", data_path="data")])

    assert prepared[0].data_path == str(data_path)
    assert data_path.is_dir()


def test_prepare_input_rejects_prepared_data_file(tmp_path: Path) -> None:
    pipeline = _PipelineHarness()
    media_path = _media_file(tmp_path)
    data_path = tmp_path / "data-file"
    data_path.write_text("not a directory", encoding="utf-8")
    state = _storage_state(media_path, data_path)

    with (
        patch.object(pipeline, "_prepare_entry_storage", return_value=state),
        pytest.raises(ValueError, match="not a directory"),
    ):
        pipeline.prepare_input([DataEntry(media_path="input.mp4", data_path="data")])


def test_prepare_output_rejects_untracked_entries() -> None:
    pipeline = _PipelineHarness()
    data_entry = DataEntry(media_path="input.mp4", data_path="data")

    with pytest.raises(RuntimeError, match="has no pipeline storage state"):
        pipeline.prepare_output([data_entry])


def test_prepare_stage_output_leaves_untracked_entries_unchanged() -> None:
    pipeline = _PipelineHarness()
    data_entry = DataEntry(media_path="input.mp4", data_path="data")

    assert pipeline.prepare_stage_output([data_entry]) == [data_entry]


def test_prepare_entry_storage_requires_data_path() -> None:
    pipeline = _PipelineHarness()

    with pytest.raises(ValueError, match="data_path is required"):
        pipeline._prepare_entry_storage(DataEntry(media_path="input.mp4", data_path=""), [])


def test_prepare_data_directory_rejects_local_file(tmp_path: Path) -> None:
    pipeline = _PipelineHarness()
    data_path = tmp_path / "data-file"
    data_path.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValueError, match="not a directory"):
        pipeline._prepare_data_directory(str(data_path), [])


def test_preserve_raw_media_rejects_existing_raw_directory(tmp_path: Path) -> None:
    pipeline = _PipelineHarness()
    media_path = _media_file(tmp_path)
    raw_path = tmp_path / "data" / "sidecars" / "raw.mp4"
    raw_path.mkdir(parents=True)

    with pytest.raises(ValueError, match="Raw media path"):
        pipeline._preserve_raw_media(str(media_path), str(tmp_path / "data"))


def test_cleanup_temp_dir_list_ignores_missing_paths(tmp_path: Path) -> None:
    pipeline = _PipelineHarness()

    pipeline._cleanup_temp_dir_list([tmp_path / "missing"])
