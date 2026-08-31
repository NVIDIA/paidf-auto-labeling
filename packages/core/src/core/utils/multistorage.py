# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Storage helpers for local paths and multistorageclient-backed remote URLs.

``MSCStorage`` centralizes the small amount of framework policy around MSC:
which URL prefixes are treated as remote, how remote files and prefixes are
downloaded into local working paths, and how local task outputs are uploaded
back when the caller requested a remote destination.
"""

import atexit
import logging
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import cast

import multistorageclient as msc
import multistorageclient.shortcuts as msc_shortcuts
from core.utils.logging import get_logger

SUPPORTED_REMOTE_URL_PREFIXES = (
    "msc://",
    "s3://",
    "gs://",
    "ais://",
    "http://",
    "https://",
    "gcp://",
    "azure://",
)


def _cleanup_inline_msc_config(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass

    if os.environ.get("MSC_CONFIG") == path:
        os.environ.pop("MSC_CONFIG", None)


class MSCStorage:
    """
    Shared wrapper around multistorageclient for one process run.

    The MSC module is imported once during startup and any remote URLs that are
    known up front can be resolved immediately so later file transfers reuse the
    cached MSC client(s) instead of rebuilding them at each call site.
    """

    logger: logging.Logger

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """
        Initialize the storage helper.

        Args:
            logger: Optional logger used for transfer messages. Defaults to a framework logger.
        """
        self.logger = logger or get_logger("multistorage")
        self._msc = msc
        self._configure_inline_config()

    def _configure_inline_config(self) -> None:
        """
        Expose inline MSC config through the file-based MSC loader.

        MSC expects ``MSC_CONFIG`` to point at a config file. Services commonly receive inline
        JSON/YAML via ``MULTISTORAGECLIENT_CONFIGURATION``, so this writes that payload to a temp
        file and points MSC at it when no explicit ``MSC_CONFIG`` is already set.
        """
        if os.getenv("MSC_CONFIG"):
            return

        config_str = os.getenv("MULTISTORAGECLIENT_CONFIGURATION")
        if not config_str:
            return

        suffix = ".json" if config_str.lstrip().startswith("{") else ".yaml"
        fd, path = tempfile.mkstemp(prefix="msc_config_", suffix=suffix)
        os.close(fd)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(config_str)
        os.environ["MSC_CONFIG"] = path
        atexit.register(_cleanup_inline_msc_config, path)
        msc_shortcuts._reinitialize_after_fork()
        self.logger.debug("Configured MSC from MULTISTORAGECLIENT_CONFIGURATION.")

    def is_remote_storage_url(self, url: str) -> bool:
        """
        Return True when the path is a supported remote storage URL.

        Args:
            url: Path or URL to inspect.
        Returns:
            True if ``url`` starts with one of ``SUPPORTED_REMOTE_URL_PREFIXES``.
        """
        return url.startswith(SUPPORTED_REMOTE_URL_PREFIXES)

    def get_client_and_path(self, url: str) -> tuple[msc.StorageClient, str]:
        """
        Resolve a supported URL or local path into an MSC client and client-local path.

        Args:
            url: Local path or MSC-supported remote URL.
        Returns:
            Storage client and path pair returned by MSC.
        """
        return cast(tuple[msc.StorageClient, str], self._msc.resolve_storage_client(url))

    def _local_file_destination(self, remote_path: str, local_path: str | None) -> str:
        """
        Resolve the local file destination for a remote file download.

        Args:
            remote_path: Client-local remote file path.
            local_path: Optional local file or directory destination.

        Returns:
            Local file path where the remote file should be downloaded.

        Raises:
            ValueError: If ``remote_path`` does not include a file name.
        """
        filename = PurePosixPath(remote_path).name
        if not filename:
            msg = f"Remote path {remote_path!r} does not include a file name."
            raise ValueError(msg)

        if local_path is None:
            return str(Path(tempfile.mkdtemp(prefix="msc_download_")) / filename)

        destination = Path(local_path)
        if destination.exists() and destination.is_dir():
            return str(destination / filename)
        return str(destination)

    def _remote_file_destination(self, remote_path: str, local_path: str) -> str:
        """
        Resolve the remote file destination for a local file upload.

        Args:
            remote_path: Client-local remote path or prefix.
            local_path: Local file path being uploaded.

        Returns:
            Client-local remote file path to upload to.

        Raises:
            ValueError: If ``local_path`` does not include a file name.
        """
        filename = Path(local_path).name
        if not filename:
            msg = f"Local path {local_path!r} does not include a file name."
            raise ValueError(msg)

        if not remote_path or remote_path.endswith("/"):
            return str(PurePosixPath(remote_path) / filename)
        return remote_path

    def _remote_file_url(self, remote_url: str, remote_path: str, destination: str) -> str:
        """
        Build the caller-facing remote URL for an uploaded file.

        Args:
            remote_url: Original remote destination URL.
            remote_path: Client-local path MSC resolved from ``remote_url``.
            destination: Client-local file path actually uploaded.

        Returns:
            ``remote_url`` when it already names the file, otherwise a URL under the remote prefix.
        """
        if destination == remote_path:
            return remote_url

        return f"{remote_url.rstrip('/')}/{PurePosixPath(destination).name}"

    def _download_remote_file(
        self,
        path: str,
        client: msc.StorageClient,
        remote_path: str,
        local_path: str | None,
    ) -> str:
        """
        Download one remote file into a local file path.

        Args:
            path: Original remote URL, used for logging.
            client: Source storage client.
            remote_path: Client-local remote file path.
            local_path: Optional local file or directory destination.
        Returns:
            Local file path that was downloaded.
        """
        destination = self._local_file_destination(remote_path, local_path)
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Downloading file {path} to {destination}")
        client.download_file(remote_path, destination)
        return destination

    def _sync_remote_directory(
        self,
        path: str,
        client: msc.StorageClient,
        remote_path: str,
        local_path: str,
    ) -> str:
        """
        Sync one remote directory/prefix into a local directory.

        Args:
            path: Original remote URL, used for logging.
            client: Source storage client.
            remote_path: Client-local remote directory or prefix.
            local_path: Local destination directory.
        Returns:
            Local directory path that was synced.
        """
        Path(local_path).mkdir(parents=True, exist_ok=True)
        if client.is_empty(remote_path):
            return local_path

        target_client, target_path = self.get_client_and_path(local_path)
        self.logger.info(f"Syncing {path} to {local_path}")
        target_client.sync_from(client, remote_path, target_path)
        return local_path

    def download_if_remote(
        self,
        path: str,
        local_path: str | None = None,
        *,
        is_file: bool,
    ) -> str:
        """
        Download or sync a remote path locally when needed.

        Local filesystem paths are returned unchanged. Remote paths are validated
        against ``is_file`` before transfer. The caller is responsible for
        cleaning up any temporary path created when ``local_path`` is omitted.

        Args:
            path: Local path or remote URL to make available locally.
            local_path: Optional local file or directory destination for remote sources.
            is_file: True when the remote source must be a file; False when it must be a
                directory/prefix.

        Returns:
            Local path that should be used by callers.

        Raises:
            ValueError: If the remote source kind does not match ``is_file`` or a remote
                file path does not include a file name.
        """
        if not self.is_remote_storage_url(path):
            return path

        client, remote_path = self.get_client_and_path(path)
        if client.is_file(remote_path):
            if not is_file:
                msg = f"Remote data path {path!r} is a file, expected a directory."
                raise ValueError(msg)
            return self._download_remote_file(path, client, remote_path, local_path)

        if is_file:
            msg = f"Remote media path {path!r} is not a file."
            raise ValueError(msg)

        if local_path is None:
            local_path = tempfile.mkdtemp(prefix="msc_download_")
        return self._sync_remote_directory(path, client, remote_path, local_path)

    def copy_local_directory(self, source_path: str, destination_path: str) -> str:
        """Copy a local directory into another local directory.

        Existing destination contents are preserved and overwritten only where
        source files share the same relative paths. Copying a directory onto
        itself is treated as a no-op.

        Args:
            source_path: Existing local directory to copy from.
            destination_path: Local directory to create or merge into.

        Returns:
            The destination path.

        Raises:
            FileNotFoundError: If the source directory does not exist.
            ValueError: If the source path exists but is not a directory, or the destination is
                inside the source directory.
        """
        source = Path(source_path)
        if not source.exists():
            msg = f"Local directory {source_path!r} does not exist."
            raise FileNotFoundError(msg)
        if not source.is_dir():
            msg = f"Local data path {source_path!r} is not a directory."
            raise ValueError(msg)

        destination = Path(destination_path)
        resolved_source = source.resolve()
        resolved_destination = destination.resolve()
        if resolved_source == resolved_destination:
            return str(destination)
        if resolved_destination.is_relative_to(resolved_source):
            msg = (
                f"Cannot copy local directory {source_path!r} into nested destination "
                f"{destination_path!r}."
            )
            raise ValueError(msg)

        destination.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination, dirs_exist_ok=True)
        return str(destination)

    def copy_local_file(self, source_path: str, destination_path: str) -> str:
        """Copy a local file to another local file path.

        Copying a file onto itself is treated as a no-op. Parent directories
        for the destination are created automatically.

        Args:
            source_path: Existing local file to copy from.
            destination_path: Local file path to create or replace.

        Returns:
            The destination path.

        Raises:
            FileNotFoundError: If the source file does not exist.
            ValueError: If the source path exists but is not a file.
        """
        source = Path(source_path)
        if not source.exists():
            msg = f"Local file {source_path!r} does not exist."
            raise FileNotFoundError(msg)
        if not source.is_file():
            msg = f"Local media path {source_path!r} is not a file."
            raise ValueError(msg)

        destination = Path(destination_path)
        if source.resolve() == destination.resolve():
            return str(destination)

        destination.parent.mkdir(parents=True, exist_ok=True)
        copy_result = shutil.copy2(source, destination)
        return str(copy_result)

    def upload_if_remote(
        self,
        local_path: str,
        remote_path: str,
        *,
        delete_unmatched_files: bool = False,
    ) -> str:
        """
        Upload a local file or directory when the destination is a remote URL.

        If remote_path is a plain filesystem path, return local_path unchanged.

        Args:
            local_path: Existing local file or directory to upload.
            remote_path: Destination path. Remote URLs trigger upload; local paths are ignored.
            delete_unmatched_files: When uploading a directory, delete files at the remote
                destination that are not present in the local source.

        Returns:
            Remote URL for uploaded files, remote directory URL for synced directories, or
            ``local_path`` when ``remote_path`` is local.

        Raises:
            FileNotFoundError: If ``local_path`` does not exist for a remote destination.
            ValueError: If ``local_path`` exists but is neither a file nor a directory.
        """
        if not self.is_remote_storage_url(remote_path):
            return local_path

        source = Path(local_path)
        if not source.exists():
            msg = f"Local path {local_path!r} does not exist."
            raise FileNotFoundError(msg)

        target_client, target_path = self.get_client_and_path(remote_path)
        if source.is_file():
            destination = self._remote_file_destination(target_path, local_path)
            self.logger.info(f"Uploading file {local_path} to {remote_path}")
            target_client.upload_file(destination, local_path)
            return self._remote_file_url(remote_path, target_path, destination)

        if not source.is_dir():
            msg = f"Local path {local_path!r} is not a file or directory."
            raise ValueError(msg)

        source_client, source_path = self.get_client_and_path(local_path)
        self.logger.info(f"Syncing {local_path} to {remote_path}")
        sync_kwargs: dict[str, bool] = {}
        if delete_unmatched_files:
            sync_kwargs["delete_unmatched_files"] = True
        target_client.sync_from(source_client, source_path, target_path, **sync_kwargs)
        return remote_path
