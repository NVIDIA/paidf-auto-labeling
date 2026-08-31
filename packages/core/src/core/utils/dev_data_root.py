# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Helpers for staging service inputs into a development DAFT data root.

``DevDataRootCopier`` owns the ``--dev-data-root`` behavior shared by services:
copy each input DAFT directory to ``dev_data_root/<data_entry.id>`` and rewrite
``data_path`` so the service annotates the copy. The root and sources may be
local paths or supported remote storage URLs.
"""

import logging
import shutil
import tempfile
from pathlib import Path, PurePosixPath

from core.models import DataEntry
from core.utils.multistorage import MSCStorage


class DevDataRootCopier:
    """
    Copy input DAFT directories to a local or remote development root.

    Existing destination directories are replaced only after every entry has
    been validated, so duplicate IDs or invalid local sources fail before any
    destination is overwritten.

    Attributes:
        logger: Logger used to report copy, download, and upload operations.
        storage: Storage helper used for local and remote filesystem operations.
    """

    logger: logging.Logger
    storage: MSCStorage

    def __init__(self, logger: logging.Logger, storage: MSCStorage | None = None) -> None:
        """
        Initialize the copier.

        Args:
            logger: Logger used to report copy, download, and upload operations.
            storage: Optional storage helper. If omitted, a default ``MSCStorage`` instance is
                created with ``logger``.
        """
        self.logger = logger
        self.storage = storage if storage is not None else MSCStorage(logger)

    def copy_entries(
        self,
        data_entries: list[DataEntry],
        dev_data_root: str,
    ) -> list[DataEntry]:
        """
        Copy input DAFT directories to a dev root and rewrite ``data_path``.

        Args:
            data_entries: Input entries whose DAFT directories should be copied.
            dev_data_root: Local or remote directory that will contain one child directory per
                entry ID.

        Returns:
            Data entries with ``data_path`` rewritten to their dev-copy directories.

        Raises:
            ValueError: If an ID is duplicate or unsafe, a local source is not a directory, or a
                destination path is invalid.
        """
        if self.storage.is_remote_storage_url(dev_data_root):
            return self._copy_entries_to_remote_root(data_entries, dev_data_root)

        return self._copy_entries_to_local_root(data_entries, dev_data_root)

    def _copy_entries_to_local_root(
        self,
        data_entries: list[DataEntry],
        dev_data_root: str,
    ) -> list[DataEntry]:
        """
        Copy input DAFT directories to a local dev root and rewrite ``data_path``.

        Args:
            data_entries: Input entries whose DAFT directories should be copied.
            dev_data_root: Local directory that will contain one child directory per entry ID.

        Returns:
            Data entries with ``data_path`` rewritten to local dev-copy directories.

        Raises:
            ValueError: If the local root is a file, an ID is duplicate or unsafe, a local
                source is not a directory, or a destination path is invalid.
        """
        root = Path(dev_data_root)
        if root.exists() and not root.is_dir():
            msg = f"Dev data root {dev_data_root!r} is not a directory."
            raise ValueError(msg)
        root.mkdir(parents=True, exist_ok=True)

        destination_by_id = self._local_destinations(data_entries, root)
        rewritten_entries: list[DataEntry] = []
        for data_entry in data_entries:
            destination = destination_by_id[data_entry.id]
            if destination.exists():
                if not destination.is_dir():
                    msg = f"Dev data destination {destination} is not a directory."
                    raise ValueError(msg)
                shutil.rmtree(destination)

            self.logger.info(
                "Copying DAFT data directory for entry %s from %s to %s.",
                data_entry.id,
                data_entry.data_path,
                destination,
            )
            if self.storage.is_remote_storage_url(data_entry.data_path):
                self.storage.download_if_remote(
                    data_entry.data_path,
                    str(destination),
                    is_file=False,
                )
            elif Path(data_entry.data_path).exists():
                self.storage.copy_local_directory(data_entry.data_path, str(destination))
            else:
                self.logger.info(
                    "Source DAFT data directory for entry %s does not exist; creating empty dev "
                    "directory %s.",
                    data_entry.id,
                    destination,
                )
                destination.mkdir(parents=True, exist_ok=True)
            rewritten_entries.append(data_entry.model_copy(update={"data_path": str(destination)}))

        return rewritten_entries

    def _copy_entries_to_remote_root(
        self,
        data_entries: list[DataEntry],
        dev_data_root: str,
    ) -> list[DataEntry]:
        """
        Copy input DAFT directories to a remote dev root and rewrite ``data_path``.

        The copy is staged through a temporary local directory so local, remote, and missing
        sources share the same behavior. The remote destination sync deletes unmatched files to
        preserve the dev-copy overwrite contract.

        Args:
            data_entries: Input entries whose DAFT directories should be copied.
            dev_data_root: Remote directory URL that will contain one child directory per
                entry ID.

        Returns:
            Data entries with ``data_path`` rewritten to remote dev-copy directory URLs.

        Raises:
            ValueError: If an ID is duplicate or unsafe, a local source is not a directory, or
                multiple IDs map to the same destination.
        """
        destination_by_id = self._remote_destinations(data_entries, dev_data_root)
        rewritten_entries: list[DataEntry] = []
        for data_entry in data_entries:
            destination = destination_by_id[data_entry.id]
            self.logger.info(
                "Copying DAFT data directory for entry %s from %s to %s.",
                data_entry.id,
                data_entry.data_path,
                destination,
            )
            with tempfile.TemporaryDirectory(prefix="dev_data_") as temp_dir:
                if self.storage.is_remote_storage_url(data_entry.data_path):
                    self.storage.download_if_remote(data_entry.data_path, temp_dir, is_file=False)
                elif Path(data_entry.data_path).exists():
                    self.storage.copy_local_directory(data_entry.data_path, temp_dir)
                else:
                    self.logger.info(
                        "Source DAFT data directory for entry %s does not exist; uploading empty "
                        "dev directory to %s.",
                        data_entry.id,
                        destination,
                    )
                self.storage.upload_if_remote(
                    temp_dir,
                    destination,
                    delete_unmatched_files=True,
                )
            rewritten_entries.append(data_entry.model_copy(update={"data_path": destination}))

        return rewritten_entries

    def _local_destinations(
        self,
        data_entries: list[DataEntry],
        root: Path,
    ) -> dict[str, Path]:
        """
        Validate entry IDs and return their local dev-copy destinations.

        Args:
            data_entries: Input entries to validate before copying.
            root: Local dev root path.

        Returns:
            Mapping from data entry ID to its destination under ``root``.

        Raises:
            ValueError: If IDs are duplicate, resolve outside ``root``, map to the same
                destination, or point a local source at its destination.
        """
        root_resolved = root.resolve()
        seen_ids: set[str] = set()
        seen_destinations: dict[Path, str] = {}
        destinations: dict[str, Path] = {}

        for data_entry in data_entries:
            if data_entry.id in seen_ids:
                msg = f"Duplicate data entry id {data_entry.id!r}."
                raise ValueError(msg)
            seen_ids.add(data_entry.id)

            destination = root / data_entry.id
            destination_resolved = destination.resolve()
            try:
                destination_resolved.relative_to(root_resolved)
            except ValueError as exc:
                msg = f"Dev data destination {destination} escapes dev data root {root_resolved}."
                raise ValueError(msg) from exc
            if destination_resolved == root_resolved:
                msg = f"Data entry id {data_entry.id!r} does not identify a child directory."
                raise ValueError(msg)

            existing_id = seen_destinations.get(destination_resolved)
            if existing_id is not None:
                msg = (
                    f"Data entry ids {existing_id!r} and {data_entry.id!r} map to the same "
                    f"dev data destination {destination}."
                )
                raise ValueError(msg)
            seen_destinations[destination_resolved] = data_entry.id

            if not self.storage.is_remote_storage_url(data_entry.data_path):
                source = Path(data_entry.data_path)
                src_resolved = source.resolve()
                if source.exists() and not source.is_dir():
                    msg = f"Local data path {data_entry.data_path!r} is not a directory."
                    raise ValueError(msg)
                if destination_resolved == src_resolved or destination_resolved.is_relative_to(
                    src_resolved
                ):
                    msg = (
                        f"Dev data destination {destination} is the same as source data_path "
                        "or inside it; choose a different --dev-data-root."
                    )
                    raise ValueError(msg)

            destinations[data_entry.id] = destination

        return destinations

    def _remote_destinations(
        self,
        data_entries: list[DataEntry],
        root: str,
    ) -> dict[str, str]:
        """
        Validate entry IDs and return their remote dev-copy destinations.

        Args:
            data_entries: Input entries to validate before copying.
            root: Remote dev root URL.

        Returns:
            Mapping from data entry ID to its destination under ``root``.
        """
        remote_root = root.rstrip("/")
        seen_ids: set[str] = set()
        seen_destinations: dict[str, str] = {}
        destinations: dict[str, str] = {}

        for data_entry in data_entries:
            if data_entry.id in seen_ids:
                msg = f"Duplicate data entry id {data_entry.id!r}."
                raise ValueError(msg)
            seen_ids.add(data_entry.id)

            relative_id = self._normalized_id_path(data_entry.id)
            destination = f"{remote_root}/{relative_id.as_posix()}"

            existing_id = seen_destinations.get(destination)
            if existing_id is not None:
                msg = (
                    f"Data entry ids {existing_id!r} and {data_entry.id!r} map to the same "
                    f"dev data destination {destination}."
                )
                raise ValueError(msg)
            seen_destinations[destination] = data_entry.id

            if not self.storage.is_remote_storage_url(data_entry.data_path):
                source = Path(data_entry.data_path)
                if source.exists() and not source.is_dir():
                    msg = f"Local data path {data_entry.data_path!r} is not a directory."
                    raise ValueError(msg)

            destinations[data_entry.id] = destination

        return destinations

    def _normalized_id_path(self, data_entry_id: str) -> PurePosixPath:
        """
        Normalize a data entry ID into a safe relative POSIX path.

        Args:
            data_entry_id: Data entry ID to normalize as a remote destination child path.

        Returns:
            Normalized relative POSIX path for the ID.

        Raises:
            ValueError: If the ID is absolute, escapes the root, or normalizes to the root itself.
        """
        raw_path = PurePosixPath(data_entry_id)
        if raw_path.is_absolute():
            msg = f"Dev data destination {data_entry_id!r} escapes dev data root."
            raise ValueError(msg)

        parts: list[str] = []
        for part in raw_path.parts:
            if part in ("", "."):
                continue
            if part == "..":
                if not parts:
                    msg = f"Dev data destination {data_entry_id!r} escapes dev data root."
                    raise ValueError(msg)
                parts.pop()
                continue
            parts.append(part)

        if not parts:
            msg = f"Data entry id {data_entry_id!r} does not identify a child directory."
            raise ValueError(msg)

        return PurePosixPath(*parts)
