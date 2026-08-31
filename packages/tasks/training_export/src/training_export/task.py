# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch task for exporting completed DAFT scenes to training formats."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, override

from core import DataEntry
from core.interfaces import TaskInterface
from core.utils.multistorage import MSCStorage

from training_export.formats import (
    COSMOS_REASON_VERSION,
    SUPPORTED_TRAINING_TASKS,
    TrainingConversionResult,
    convert_metropolis_scenes_to_cosmos_reason,
    convert_metropolis_scenes_to_tao_vl_reason,
)

TrainingExportFormat = Literal["cosmos-reason-v1.0", "tao-vl-reason-v1.0"]
TRAINING_EXPORT_FORMATS: tuple[TrainingExportFormat, ...] = (
    "cosmos-reason-v1.0",
    "tao-vl-reason-v1.0",
)
TRAINING_EXPORT_TASKS: tuple[str, ...] = tuple(sorted(SUPPORTED_TRAINING_TASKS))


class _TrainingExportStorage(Protocol):
    def is_remote_storage_url(self, url: str) -> bool: ...

    def download_if_remote(
        self,
        path: str,
        local_path: str | None = None,
        *,
        is_file: bool,
    ) -> str: ...

    def upload_if_remote(
        self,
        local_path: str,
        remote_path: str,
        *,
        delete_unmatched_files: bool = False,
    ) -> str: ...


@dataclass(frozen=True)
class TrainingExportConfig:
    """Configuration for training-format dataset export."""

    formats: tuple[TrainingExportFormat, ...] = ()
    output_dir: str | Path | None = None
    task_types: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)
    copy_media: bool = True
    emit_media_root_as_null: bool = False
    enabled: bool = True


class TrainingExportTask(TaskInterface):
    """Export a batch of completed DAFT scene directories to training datasets."""

    def __init__(
        self,
        *,
        config: TrainingExportConfig | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or "training_export")
        self.config = config or TrainingExportConfig()
        self.last_results: tuple[TrainingConversionResult, ...] = ()

    @override
    def run(self, data_entry: DataEntry) -> DataEntry:
        return self.run_batch([data_entry])[0]

    @override
    def run_batch(self, data_entries: list[DataEntry]) -> list[DataEntry]:
        if not self.config.enabled or not self.config.formats:
            self.logger.info("Training export disabled; skipping.")
            self.last_results = ()
            return data_entries

        self.last_results = run_training_exports(data_entries, self.config, logger=self.logger)
        return data_entries


def run_training_exports(
    data_entries: list[DataEntry],
    config: TrainingExportConfig,
    *,
    logger: logging.Logger,
    storage: _TrainingExportStorage | None = None,
) -> tuple[TrainingConversionResult, ...]:
    """Export completed DAFT scene outputs to configured training formats."""
    formats = _selected_formats(config)
    if not config.enabled or not formats:
        return ()

    output_root_value = config.output_dir
    if output_root_value is None or not str(output_root_value).strip():
        raise ValueError("training export output_dir is required when formats are set")

    export_storage = storage or MSCStorage(logger)
    temp_dirs: list[Path] = []
    try:
        scene_dirs, has_remote_scene = _scene_dirs(data_entries, export_storage, temp_dirs)
        if not scene_dirs:
            raise ValueError("Training export requires at least one DataEntry data_path")
        if has_remote_scene and not config.copy_media:
            raise ValueError(
                "Training export with remote DataEntry data_path values requires copy_media=True "
                "so exported datasets do not reference temporary staged media paths."
            )

        output_root, remote_output_dir = _output_root(output_root_value, export_storage, temp_dirs)
        task_types = _selected_task_types(config)
        metadata = dict(config.metadata)

        results: list[TrainingConversionResult] = []
        media_anchor = _common_parent(scene_dirs)
        for format_name in formats:
            output_dir = output_root / format_name
            if format_name == COSMOS_REASON_VERSION:
                result = convert_metropolis_scenes_to_cosmos_reason(
                    scene_dirs,
                    output_dir,
                    media_anchor=media_anchor,
                    task_types=task_types,
                    copy_media=config.copy_media,
                    metadata=metadata,
                )
            else:
                result = convert_metropolis_scenes_to_tao_vl_reason(
                    scene_dirs,
                    output_dir,
                    media_anchor=media_anchor,
                    task_types=task_types,
                    copy_media=config.copy_media,
                    metadata=metadata,
                    emit_media_root_as_null=config.emit_media_root_as_null,
                )
            _log_export_result(logger, format_name, output_dir, result)
            if result.errors:
                joined = "; ".join(result.errors)
                raise ValueError(f"{format_name} training export failed: {joined}")
            results.append(result)

        if remote_output_dir is not None:
            export_storage.upload_if_remote(str(output_root), remote_output_dir)
        return tuple(results)
    finally:
        _cleanup_temp_dirs(temp_dirs)


def _selected_formats(config: TrainingExportConfig) -> tuple[TrainingExportFormat, ...]:
    selected: list[TrainingExportFormat] = []
    for raw_format in config.formats:
        if raw_format not in TRAINING_EXPORT_FORMATS:
            choices = ", ".join(TRAINING_EXPORT_FORMATS)
            raise ValueError(
                f"Unsupported training export format {raw_format!r}; expected {choices}"
            )
        if raw_format not in selected:
            selected.append(raw_format)
    return tuple(selected)


def _selected_task_types(config: TrainingExportConfig) -> tuple[str, ...] | None:
    selected: list[str] = []
    for task_type in config.task_types:
        if task_type not in TRAINING_EXPORT_TASKS:
            choices = ", ".join(TRAINING_EXPORT_TASKS)
            raise ValueError(f"Unsupported training export task {task_type!r}; expected {choices}")
        if task_type not in selected:
            selected.append(task_type)
    return tuple(selected) or None


def _scene_dirs(
    data_entries: list[DataEntry],
    storage: _TrainingExportStorage,
    temp_dirs: list[Path],
) -> tuple[tuple[Path, ...], bool]:
    scenes: list[Path] = []
    has_remote_scene = False
    for entry in data_entries:
        data_path = entry.data_path
        if storage.is_remote_storage_url(data_path):
            has_remote_scene = True
            local_dir = Path(tempfile.mkdtemp(prefix="training_export_scene_"))
            temp_dirs.append(local_dir)
            downloaded = storage.download_if_remote(data_path, str(local_dir), is_file=False)
            scenes.append(Path(downloaded).resolve())
        else:
            scenes.append(Path(data_path).expanduser().resolve())
    return tuple(dict.fromkeys(scenes)), has_remote_scene


def _output_root(
    output_root_value: str | Path,
    storage: _TrainingExportStorage,
    temp_dirs: list[Path],
) -> tuple[Path, str | None]:
    output_root_text = str(output_root_value)
    if storage.is_remote_storage_url(output_root_text):
        local_dir = Path(tempfile.mkdtemp(prefix="training_export_output_"))
        temp_dirs.append(local_dir)
        return local_dir, output_root_text
    return Path(output_root_value).expanduser().resolve(), None


def _cleanup_temp_dirs(temp_dirs: list[Path]) -> None:
    for temp_dir in reversed(temp_dirs):
        shutil.rmtree(temp_dir, ignore_errors=True)


def _log_export_result(
    logger: logging.Logger,
    format_name: str,
    output_dir: Path,
    result: TrainingConversionResult,
) -> None:
    logger.info(
        "Training export %s wrote %d samples to %s.",
        format_name,
        result.samples_written,
        output_dir,
    )
    for warning in result.warnings:
        logger.warning("[%s] %s", format_name, warning)


def _common_parent(paths: tuple[Path, ...]) -> Path:
    if len(paths) == 1:
        return paths[0]
    return Path(os.path.commonpath([str(path) for path in paths]))


__all__ = [
    "TRAINING_EXPORT_FORMATS",
    "TRAINING_EXPORT_TASKS",
    "TrainingExportConfig",
    "TrainingExportFormat",
    "TrainingExportTask",
    "run_training_exports",
]
