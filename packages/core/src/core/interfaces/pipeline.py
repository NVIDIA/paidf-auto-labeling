# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
This module contains the interface to be implemented by all pipeline packages. It provides:
- An interface for pipeline implementations to adhere to.
- Shared input/output preparation for DAFT scene directories, active media handoff, and
  local/remote storage.

See core/pipelines.py for existing implementations. Existing pipelines exist for:
- Linear pipeline: A linear pipeline is a pipeline that runs each task in sequence on the output
  of the previous task.

To create a new pipeline for a custom use case, subclass the PipelineInterface and implement the
run method.
"""

import logging
import shutil
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from core.models import DataEntry, read_pipeline_state, write_pipeline_state
from core.scene import (
    active_media_path_for_input,
    ensure_scene_skeleton,
    find_active_media_path,
    raw_media_path,
    scene_media_id_from_path,
)
from core.utils.logging import get_logger
from core.utils.multistorage import MSCStorage


@dataclass
class _PipelineStorageState:
    """
    Bookkeeping for one data entry's local DAFT working directory.

    Attributes:
        original_media_path: Original local or remote media path from the input entry.
        working_media_path: Local ``sidecars/active.*`` path tasks should read.
        original_data_path: Original local or remote DAFT data directory from the input entry.
        working_data_path: Local DAFT directory tasks should mutate during execution.
        final_data_path: Caller-visible DAFT data directory path.
        temp_dirs: Framework-owned temporary directories for this data entry.
    """

    original_media_path: str
    working_media_path: str
    original_data_path: str
    working_data_path: str
    final_data_path: str
    temp_dirs: list[Path]


class PipelineInterface(ABC):
    """
    Base class for pipeline packages.

    Subclasses implement ``run()`` and should call ``prepare_input()`` before the first task,
    ``prepare_stage_output()`` between tasks that may update media, and ``prepare_output()`` before
    returning results to the caller. The base class keeps task execution local by staging remote
    DAFT directories and media into framework-owned working paths.
    """

    logger: logging.Logger

    def __init__(self, name: str | None = None) -> None:
        """
        Initialize shared pipeline logging and storage state.

        Args:
            name: Optional human-readable pipeline name used in logs. Defaults to the class name.
        """
        self.name = name or self.__class__.__name__
        self.logger = get_logger(self.name, kind="pipeline")
        self.storage = MSCStorage(self.logger)
        self.original_data_entries: list[DataEntry] = []
        self._storage_state_by_entry_id: dict[str, _PipelineStorageState] = {}
        self.logger.info(f"Initialized Pipeline {self.name}.")

    @abstractmethod
    def run(self, data_entries: list[DataEntry]) -> list[DataEntry]:
        """
        Run the pipeline against a batch of data entries.

        Args:
            data_entries: The data entries to be annotated.
        Returns:
            A list of annotated data entries.
        """
        ...

    def prepare_input(self, data_entries: list[DataEntry]) -> list[DataEntry]:
        """
        Prepare data entries for task execution.

        The returned entries point at local working paths. ``media_path`` is rewritten to the
        active media sidecar under the DAFT directory, and ``data_path`` is rewritten to the local
        DAFT directory used by tasks. Original caller-facing paths are kept in pipeline state so
        ``prepare_output()`` can restore them later.

        Args:
            data_entries: The data entries to be prepared.
        Returns:
            Data entries with local task-facing ``media_path`` and ``data_path`` values.
        Raises:
            FileNotFoundError: If staged media is missing.
            ValueError: If a staged media path is not a file, a staged data path is not a
                directory, or duplicate data entry IDs are provided.
        """
        self.original_data_entries = data_entries
        self._storage_state_by_entry_id = {}
        local_data_entries: list[DataEntry] = []
        try:
            for data_entry in data_entries:
                if data_entry.id in self._storage_state_by_entry_id:
                    msg = f"Duplicate data entry id {data_entry.id!r}."
                    raise ValueError(msg)

                temp_dirs: list[Path] = []
                local_data_entry = data_entry.model_copy(deep=True)
                try:
                    storage_state = self._prepare_entry_storage(data_entry, temp_dirs)
                    local_data_entry.media_path = storage_state.working_media_path
                    local_data_entry.data_path = storage_state.working_data_path

                    media = Path(local_data_entry.media_path)
                    if not media.exists():
                        self.logger.error(
                            f"Media file {local_data_entry.media_path} does not exist."
                        )
                        raise FileNotFoundError(
                            f"Media file {local_data_entry.media_path} does not exist."
                        )
                    if not media.is_file():
                        self.logger.error(
                            f"Media path {local_data_entry.media_path} is not a file."
                        )
                        raise ValueError(f"Media path {local_data_entry.media_path} is not a file.")

                    data_dir = Path(local_data_entry.data_path)
                    if not data_dir.exists():
                        self.logger.debug(f"Creating data directory {local_data_entry.data_path}.")
                        data_dir.mkdir(parents=True, exist_ok=True)
                    elif not data_dir.is_dir():
                        self.logger.error(
                            f"Data path {local_data_entry.data_path} is not a directory."
                        )
                        raise ValueError(
                            f"Data path {local_data_entry.data_path} is not a directory."
                        )
                except Exception:
                    self._cleanup_temp_dir_list(temp_dirs)
                    raise

                self._storage_state_by_entry_id[data_entry.id] = storage_state
                local_data_entries.append(local_data_entry)
        except Exception:
            try:
                self._cleanup_temp_dirs()
            finally:
                self._storage_state_by_entry_id = {}
            raise

        return local_data_entries

    def prepare_output(self, data_entries: list[DataEntry]) -> list[DataEntry]:
        """
        Restore caller-facing paths and publish any remote DAFT directory updates.

        Entries that were prepared by ``prepare_input()`` have their active media promoted, remote
        DAFT working directories uploaded, and original ``media_path`` restored. Entries without
        storage state are returned unchanged.

        Args:
            data_entries: The data entries to be prepared.
        Returns:
            Data entries with caller-facing ``media_path`` and final ``data_path`` values.
        """
        prepared_data_entries: list[DataEntry] = []
        try:
            for data_entry in data_entries:
                state = self._storage_state_by_entry_id.get(data_entry.id)
                if state is None:
                    raise RuntimeError(f"Data entry {data_entry.id} has no pipeline storage state.")

                prepared_data_entry = self._promote_active_media(data_entry, state)
                if self.storage.is_remote_storage_url(state.final_data_path):
                    self.storage.upload_if_remote(state.working_data_path, state.final_data_path)

                prepared_data_entry.media_path = state.original_media_path
                prepared_data_entry.data_path = state.final_data_path
                prepared_data_entries.append(prepared_data_entry)
        finally:
            self._cleanup_temp_dirs()
        return prepared_data_entries

    def prepare_stage_output(self, data_entries: list[DataEntry]) -> list[DataEntry]:
        """
        Normalize task outputs back to the local DAFT active-media contract.

        Tasks may either update ``data_entry.media_path`` to point at a newly
        produced media file or update the active file in place. Between tasks,
        the pipeline promotes any returned media path into ``sidecars/active.*``
        under the DAFT directory and resets ``media_path`` so downstream tasks
        and later containers consume the same handoff file.

        Args:
            data_entries: Task outputs to normalize before the next task runs.
        Returns:
            Data entries with ``media_path`` pointing at the active media sidecar.
        """
        prepared_data_entries: list[DataEntry] = []
        for data_entry in data_entries:
            state = self._storage_state_by_entry_id.get(data_entry.id)
            if state is None:
                prepared_data_entries.append(data_entry)
                continue
            prepared_data_entries.append(self._promote_active_media(data_entry, state))
        return prepared_data_entries

    def _prepare_entry_storage(
        self,
        data_entry: DataEntry,
        temp_dirs: list[Path],
    ) -> _PipelineStorageState:
        """Prepare a local DAFT directory and active media for one data entry.

        The original ``media_path`` is never rewritten back to the caller.
        ``media_path`` is also preserved once under ``data_path/sidecars`` as
        ``raw<source suffix>`` if that file does not already exist. If
        ``data_path/sidecars`` already contains an active media sidecar, that
        file is used as the task input. Otherwise the original media is copied
        or downloaded into ``sidecars`` as the active file.

        Args:
            data_entry: The original input data entry.
            temp_dirs: Mutable collection of temp directories created for this data entry.

        Returns:
            Storage state describing local working paths and final destinations.
        """
        if not data_entry.data_path:
            msg = "data_path is required."
            raise ValueError(msg)

        working_data_path = self._prepare_data_directory(data_entry.data_path, temp_dirs)
        self._preserve_scene_identity(data_entry, working_data_path)
        preserved_raw_media_path, raw_was_created = self._preserve_raw_media(
            data_entry.media_path,
            working_data_path,
        )
        active_media_path = find_active_media_path(working_data_path)
        if active_media_path is None:
            active_media_path = active_media_path_for_input(
                working_data_path,
                data_entry.media_path,
            )
            if raw_was_created or self.storage.is_remote_storage_url(data_entry.media_path):
                active_source_path = str(preserved_raw_media_path)
            elif Path(data_entry.media_path).exists():
                active_source_path = data_entry.media_path
            else:
                active_source_path = str(preserved_raw_media_path)
            self._stage_media_file(active_source_path, str(active_media_path))
        else:
            self.logger.info(f"Using existing active media {active_media_path}.")

        return _PipelineStorageState(
            original_media_path=data_entry.media_path,
            working_media_path=str(active_media_path),
            original_data_path=data_entry.data_path,
            working_data_path=working_data_path,
            final_data_path=data_entry.data_path,
            temp_dirs=temp_dirs,
        )

    def _prepare_data_directory(
        self,
        data_path: str,
        temp_dirs: list[Path],
    ) -> str:
        """Return a local DAFT directory for task execution.

        Remote DAFT directories are synced into a framework-owned temp
        directory and uploaded back during ``prepare_output``. Local DAFT
        directories are used in place and created if missing.

        Args:
            data_path: Original local or remote DAFT data directory.
            temp_dirs: Mutable collection of temp directories created for this data entry.

        Returns:
            A local DAFT directory path.
        """
        if self.storage.is_remote_storage_url(data_path):
            working_data_path = self._make_temp_dir("pipeline_data_", temp_dirs)
            self.storage.download_if_remote(data_path, working_data_path, is_file=False)
        else:
            working_data_path = data_path
            data_dir = Path(working_data_path)
            if data_dir.exists() and not data_dir.is_dir():
                self.logger.error(f"Data path {working_data_path} is not a directory.")
                raise ValueError(f"Data path {working_data_path} is not a directory.")

        ensure_scene_skeleton(working_data_path)
        return working_data_path

    def _preserve_scene_identity(self, data_entry: DataEntry, working_data_path: str) -> None:
        """Persist the caller-facing scene media ID before any temp-dir staging leaks in."""
        media_id = scene_media_id_from_path(data_entry.data_path)
        if not media_id:
            return

        pipeline_state = read_pipeline_state(working_data_path)
        if pipeline_state.media_id:
            return

        pipeline_state.media_id = media_id
        write_pipeline_state(working_data_path, pipeline_state)

    def _preserve_raw_media(self, media_path: str, data_path: str) -> tuple[Path, bool]:
        """Copy ``media_path`` to the raw sidecar if it is not already present.

        Args:
            media_path: Local or remote original media path from the input entry.
            data_path: Local DAFT directory containing the ``sidecars`` directory.

        Returns:
            The raw-media path and a flag indicating whether this call created it.
        """
        destination = raw_media_path(data_path, media_path)
        if destination.exists():
            if not destination.is_file():
                msg = f"Raw media path {destination} is not a file."
                raise ValueError(msg)
            self.logger.info(f"Using existing raw media {destination}.")
            return destination, False

        self._stage_media_file(media_path, str(destination))
        return destination, True

    def _promote_active_media(
        self,
        data_entry: DataEntry,
        state: _PipelineStorageState,
    ) -> DataEntry:
        """Copy a task-returned media path into ``sidecars/active.*`` and normalize fields.

        Args:
            data_entry: Data entry returned by a task.
            state: Storage state created for the entry during input preparation.

        Returns:
            A copied data entry pointing at the local working media and DAFT directory.
        """
        prepared_data_entry: DataEntry = data_entry.model_copy(deep=True)
        source_path = prepared_data_entry.media_path
        source_suffix = Path(source_path).suffix
        current_suffix = Path(state.working_media_path).suffix
        destination_path = state.working_media_path
        if (
            source_path != state.working_media_path
            and source_suffix.lower() != current_suffix.lower()
        ):
            destination_path = str(
                active_media_path_for_input(state.working_data_path, source_path)
            )

        if source_path != destination_path:
            self._stage_media_file(source_path, destination_path)

        if destination_path != state.working_media_path:
            previous_active = Path(state.working_media_path)
            if previous_active.exists() and previous_active.is_file():
                previous_active.unlink()
            state.working_media_path = destination_path

        prepared_data_entry.media_path = state.working_media_path
        prepared_data_entry.data_path = state.working_data_path
        return prepared_data_entry

    def _stage_media_file(self, source_path: str, destination_path: str) -> None:
        """Copy or download a media file into a local working destination.

        Args:
            source_path: Local or remote media file to stage.
            destination_path: Local file path where the media should be available.
        """
        if self.storage.is_remote_storage_url(source_path):
            self.storage.download_if_remote(source_path, destination_path, is_file=True)
            return

        self.storage.copy_local_file(source_path, destination_path)

    def _make_temp_dir(self, prefix: str, temp_dirs: list[Path]) -> str:
        """Create and track a framework-owned temporary directory.

        Args:
            prefix: Prefix to use for the created temp directory name.
            temp_dirs: Mutable collection where the created directory should be recorded.

        Returns:
            The created temp directory path as a string.
        """
        temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
        temp_dirs.append(temp_dir)
        return str(temp_dir)

    def _cleanup_temp_dirs(self) -> None:
        """Remove all framework-owned temp directories from the latest pipeline run."""
        for state in self._storage_state_by_entry_id.values():
            self._cleanup_temp_dir_list(state.temp_dirs)

    def _cleanup_temp_dir_list(self, temp_dirs: list[Path]) -> None:
        """Remove framework-owned temp directories, tolerating already-removed paths."""
        for temp_dir in temp_dirs:
            try:
                shutil.rmtree(temp_dir)
            except FileNotFoundError:
                continue
