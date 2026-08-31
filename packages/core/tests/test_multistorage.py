# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from core.utils.multistorage import MSCStorage, _cleanup_inline_msc_config


def test_storage_initializes_without_inline_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MSC_CONFIG", raising=False)
    monkeypatch.delenv("MULTISTORAGECLIENT_CONFIGURATION", raising=False)

    MSCStorage()


def test_storage_skips_inline_config_when_explicit_config_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MSC_CONFIG", "/etc/msc/config.yaml")
    monkeypatch.setenv("MULTISTORAGECLIENT_CONFIGURATION", '{"profiles": {}}')

    with (
        patch("core.utils.multistorage.tempfile.mkstemp") as mkstemp,
        patch("core.utils.multistorage.atexit.register") as cleanup_register,
        patch("core.utils.multistorage.msc_shortcuts._reinitialize_after_fork") as reinit,
    ):
        MSCStorage()

    mkstemp.assert_not_called()
    cleanup_register.assert_not_called()
    reinit.assert_not_called()
    assert os.environ["MSC_CONFIG"] == "/etc/msc/config.yaml"


def test_storage_configures_inline_json_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("MSC_CONFIG", raising=False)
    config = ' {"profiles": {}}'
    config_path = tmp_path / "msc_config.json"
    fd = os.open(config_path, os.O_CREAT | os.O_RDWR)
    monkeypatch.setenv("MULTISTORAGECLIENT_CONFIGURATION", config)

    with (
        patch(
            "core.utils.multistorage.tempfile.mkstemp",
            return_value=(fd, str(config_path)),
        ) as mkstemp,
        patch("core.utils.multistorage.atexit.register") as cleanup_register,
        patch("core.utils.multistorage.msc_shortcuts._reinitialize_after_fork") as reinit,
    ):
        MSCStorage()

    mkstemp.assert_called_once_with(prefix="msc_config_", suffix=".json")
    cleanup_register.assert_called_once_with(_cleanup_inline_msc_config, str(config_path))
    reinit.assert_called_once_with()
    assert os.environ["MSC_CONFIG"] == str(config_path)
    assert config_path.read_text(encoding="utf-8") == config


def test_inline_config_cleanup_removes_file_and_clears_matching_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "msc_config.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("MSC_CONFIG", str(config_path))

    _cleanup_inline_msc_config(str(config_path))

    assert not config_path.exists()
    assert "MSC_CONFIG" not in os.environ


def test_inline_config_cleanup_keeps_changed_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "msc_config.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("MSC_CONFIG", "/etc/msc/config.yaml")

    _cleanup_inline_msc_config(str(config_path))

    assert not config_path.exists()
    assert os.environ["MSC_CONFIG"] == "/etc/msc/config.yaml"


def test_inline_config_cleanup_ignores_remove_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "msc_config.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("MSC_CONFIG", str(config_path))

    with patch("core.utils.multistorage.os.remove", side_effect=OSError):
        _cleanup_inline_msc_config(str(config_path))

    assert os.environ.get("MSC_CONFIG") is None


def test_download_if_remote_returns_local_path_unchanged(tmp_path: Path) -> None:
    storage = MSCStorage()
    local_path = str(tmp_path / "input.mp4")

    with patch.object(storage, "get_client_and_path") as resolve:
        assert storage.download_if_remote(local_path, is_file=True) == local_path

    resolve.assert_not_called()


def test_download_if_remote_downloads_file_to_temp_path(tmp_path: Path) -> None:
    client = MagicMock()
    client.is_file.return_value = True

    with (
        patch("core.utils.multistorage.tempfile.mkdtemp", return_value=str(tmp_path)),
        patch(
            "core.utils.multistorage.msc.resolve_storage_client",
            return_value=(client, "videos/file.mp4"),
        ),
    ):
        downloaded = MSCStorage().download_if_remote(
            "s3://bucket/videos/file.mp4",
            is_file=True,
        )

    assert downloaded == str(tmp_path / "file.mp4")
    client.is_file.assert_called_once_with("videos/file.mp4")
    client.download_file.assert_called_once_with("videos/file.mp4", str(tmp_path / "file.mp4"))


def test_download_if_remote_downloads_file_to_existing_directory(tmp_path: Path) -> None:
    client = MagicMock()
    client.is_file.return_value = True
    destination = tmp_path / "downloads"
    destination.mkdir()

    with patch(
        "core.utils.multistorage.msc.resolve_storage_client",
        return_value=(client, "videos/file.mp4"),
    ):
        downloaded = MSCStorage().download_if_remote(
            "s3://bucket/videos/file.mp4",
            str(destination),
            is_file=True,
        )

    assert downloaded == str(destination / "file.mp4")
    client.download_file.assert_called_once_with("videos/file.mp4", str(destination / "file.mp4"))


def test_download_if_remote_rejects_remote_file_without_filename() -> None:
    client = MagicMock()
    client.is_file.return_value = True

    with patch(
        "core.utils.multistorage.msc.resolve_storage_client",
        return_value=(client, ""),
    ):
        with pytest.raises(ValueError, match="does not include a file name"):
            MSCStorage().download_if_remote("s3://bucket", is_file=True)


def test_download_if_remote_returns_local_file_path_unchanged(tmp_path: Path) -> None:
    storage = MSCStorage()
    local_path = str(tmp_path / "input.mp4")

    with patch.object(storage, "get_client_and_path") as resolve:
        assert storage.download_if_remote(local_path, is_file=True) == local_path

    resolve.assert_not_called()


def test_download_if_remote_rejects_remote_prefix_when_file_expected() -> None:
    client = MagicMock()
    client.is_file.return_value = False

    with patch(
        "core.utils.multistorage.msc.resolve_storage_client",
        return_value=(client, "videos/"),
    ):
        with pytest.raises(ValueError, match="not a file"):
            MSCStorage().download_if_remote("s3://bucket/videos/", is_file=True)

    client.is_file.assert_called_once_with("videos/")


def test_download_if_remote_downloads_file_to_explicit_destination(tmp_path: Path) -> None:
    client = MagicMock()
    client.is_file.return_value = True
    destination = tmp_path / "active.mp4"

    with patch(
        "core.utils.multistorage.msc.resolve_storage_client",
        return_value=(client, "videos/file.mp4"),
    ):
        downloaded = MSCStorage().download_if_remote(
            "s3://bucket/videos/file.mp4",
            str(destination),
            is_file=True,
        )

    assert downloaded == str(destination)
    client.is_file.assert_called_once_with("videos/file.mp4")
    client.download_file.assert_called_once_with("videos/file.mp4", str(destination))


def test_download_if_remote_returns_local_directory_path_unchanged(tmp_path: Path) -> None:
    storage = MSCStorage()
    local_path = str(tmp_path / "annotations")

    with patch.object(storage, "get_client_and_path") as resolve:
        assert (
            storage.download_if_remote(local_path, str(tmp_path / "download"), is_file=False)
            == local_path
        )

    resolve.assert_not_called()


def test_download_if_remote_syncs_remote_prefix_to_temp_dir(tmp_path: Path) -> None:
    source_client = MagicMock()
    source_client.is_file.return_value = False
    source_client.is_empty.return_value = False
    target_client = MagicMock()
    destination = tmp_path / "download"

    with (
        patch("core.utils.multistorage.tempfile.mkdtemp", return_value=str(destination)),
        patch(
            "core.utils.multistorage.msc.resolve_storage_client",
            side_effect=[
                (source_client, "annotations"),
                (target_client, str(destination)),
            ],
        ),
    ):
        downloaded = MSCStorage().download_if_remote("s3://bucket/annotations", is_file=False)

    assert downloaded == str(destination)
    source_client.is_file.assert_called_once_with("annotations")
    target_client.sync_from.assert_called_once_with(
        source_client,
        "annotations",
        str(destination),
    )


def test_upload_returns_local_path_when_destination_is_local(tmp_path: Path) -> None:
    storage = MSCStorage()
    local_path = tmp_path / "annotations"
    local_path.mkdir()

    with patch.object(storage, "get_client_and_path") as resolve:
        uploaded = storage.upload_if_remote(str(local_path), str(tmp_path / "output"))

    assert uploaded == str(local_path)
    resolve.assert_not_called()


def test_upload_uploads_file_to_remote_path(tmp_path: Path) -> None:
    local_path = tmp_path / "labels.json"
    local_path.write_text("{}", encoding="utf-8")
    target_client = MagicMock()

    with patch(
        "core.utils.multistorage.msc.resolve_storage_client",
        return_value=(target_client, "annotations/labels.json"),
    ):
        uploaded = MSCStorage().upload_if_remote(
            str(local_path),
            "s3://bucket/annotations/labels.json",
        )

    assert uploaded == "s3://bucket/annotations/labels.json"
    target_client.upload_file.assert_called_once_with(
        "annotations/labels.json",
        str(local_path),
    )


def test_upload_uploads_file_to_remote_prefix(tmp_path: Path) -> None:
    local_path = tmp_path / "labels.json"
    local_path.write_text("{}", encoding="utf-8")
    target_client = MagicMock()

    with patch(
        "core.utils.multistorage.msc.resolve_storage_client",
        return_value=(target_client, "annotations/"),
    ):
        uploaded = MSCStorage().upload_if_remote(str(local_path), "s3://bucket/annotations/")

    assert uploaded == "s3://bucket/annotations/labels.json"
    target_client.upload_file.assert_called_once_with(
        "annotations/labels.json",
        str(local_path),
    )


def test_upload_syncs_local_directory_to_remote_prefix(tmp_path: Path) -> None:
    local_path = tmp_path / "annotations"
    local_path.mkdir()
    target_client = MagicMock()
    source_client = MagicMock()

    with patch(
        "core.utils.multistorage.msc.resolve_storage_client",
        side_effect=[
            (target_client, "outputs/annotations"),
            (source_client, str(local_path)),
        ],
    ):
        uploaded = MSCStorage().upload_if_remote(
            str(local_path),
            "s3://bucket/outputs/annotations",
        )

    assert uploaded == "s3://bucket/outputs/annotations"
    target_client.sync_from.assert_called_once_with(
        source_client,
        str(local_path),
        "outputs/annotations",
    )


def test_upload_syncs_local_directory_to_remote_prefix_with_delete_unmatched(
    tmp_path: Path,
) -> None:
    local_path = tmp_path / "annotations"
    local_path.mkdir()
    target_client = MagicMock()
    source_client = MagicMock()

    with patch(
        "core.utils.multistorage.msc.resolve_storage_client",
        side_effect=[
            (target_client, "outputs/annotations"),
            (source_client, str(local_path)),
        ],
    ):
        uploaded = MSCStorage().upload_if_remote(
            str(local_path),
            "s3://bucket/outputs/annotations",
            delete_unmatched_files=True,
        )

    assert uploaded == "s3://bucket/outputs/annotations"
    target_client.sync_from.assert_called_once_with(
        source_client,
        str(local_path),
        "outputs/annotations",
        delete_unmatched_files=True,
    )


def test_download_if_remote_syncs_remote_prefix_to_local_dir(tmp_path: Path) -> None:
    source_client = MagicMock()
    source_client.is_file.return_value = False
    source_client.is_empty.return_value = False
    target_client = MagicMock()
    destination = tmp_path / "download"

    with patch(
        "core.utils.multistorage.msc.resolve_storage_client",
        side_effect=[
            (source_client, "annotations"),
            (target_client, str(destination)),
        ],
    ):
        downloaded = MSCStorage().download_if_remote(
            "s3://bucket/annotations",
            str(destination),
            is_file=False,
        )

    assert downloaded == str(destination)
    source_client.is_file.assert_called_once_with("annotations")
    source_client.is_empty.assert_called_once_with("annotations")
    target_client.sync_from.assert_called_once_with(
        source_client,
        "annotations",
        str(destination),
    )


def test_download_if_remote_creates_empty_dir_for_empty_remote(
    tmp_path: Path,
) -> None:
    source_client = MagicMock()
    source_client.is_file.return_value = False
    source_client.is_empty.return_value = True
    destination = tmp_path / "download"

    with patch(
        "core.utils.multistorage.msc.resolve_storage_client",
        return_value=(source_client, "annotations"),
    ):
        downloaded = MSCStorage().download_if_remote(
            "s3://bucket/annotations",
            str(destination),
            is_file=False,
        )

    assert downloaded == str(destination)
    assert destination.exists() and destination.is_dir()
    source_client.is_file.assert_called_once_with("annotations")
    source_client.is_empty.assert_called_once_with("annotations")


def test_download_if_remote_rejects_remote_file_when_directory_expected(tmp_path: Path) -> None:
    source_client = MagicMock()
    source_client.is_file.return_value = True

    with patch(
        "core.utils.multistorage.msc.resolve_storage_client",
        return_value=(source_client, "annotations/labels.json"),
    ):
        with pytest.raises(ValueError, match="expected a directory"):
            MSCStorage().download_if_remote(
                "s3://bucket/annotations/labels.json",
                str(tmp_path / "download"),
                is_file=False,
            )


def test_copy_local_directory_copies_existing_contents(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "labels.json").write_text("{}", encoding="utf-8")
    destination = tmp_path / "destination"

    copied = MSCStorage().copy_local_directory(str(source), str(destination))

    assert copied == str(destination)
    assert (destination / "labels.json").read_text(encoding="utf-8") == "{}"


def test_copy_local_directory_rejects_missing_source(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        MSCStorage().copy_local_directory(
            str(tmp_path / "missing"),
            str(tmp_path / "destination"),
        )


def test_copy_local_directory_rejects_file_source(tmp_path: Path) -> None:
    source = tmp_path / "labels.json"
    source.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="not a directory"):
        MSCStorage().copy_local_directory(str(source), str(tmp_path / "destination"))


def test_copy_local_directory_returns_same_path_for_self_copy(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()

    copied = MSCStorage().copy_local_directory(str(source), str(source))

    assert copied == str(source)


def test_copy_local_directory_rejects_nested_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "labels.json").write_text("{}", encoding="utf-8")
    destination = source / "nested" / "destination"

    with pytest.raises(ValueError, match="nested destination"):
        MSCStorage().copy_local_directory(str(source), str(destination))

    assert not destination.exists()


def test_copy_local_file_copies_existing_file(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"media")
    destination = tmp_path / "nested" / "destination.mp4"

    copied = MSCStorage().copy_local_file(str(source), str(destination))

    assert copied == str(destination)
    assert destination.read_bytes() == b"media"


def test_copy_local_file_returns_actual_path_for_directory_destination(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"media")
    destination = tmp_path / "nested"
    destination.mkdir()

    copied = MSCStorage().copy_local_file(str(source), str(destination))

    copied_path = destination / source.name
    assert copied == str(copied_path)
    assert copied_path.read_bytes() == b"media"


def test_copy_local_file_rejects_missing_source(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        MSCStorage().copy_local_file(
            str(tmp_path / "missing.mp4"),
            str(tmp_path / "destination.mp4"),
        )


def test_copy_local_file_rejects_directory_source(tmp_path: Path) -> None:
    source = tmp_path / "media"
    source.mkdir()

    with pytest.raises(ValueError, match="not a file"):
        MSCStorage().copy_local_file(str(source), str(tmp_path / "destination.mp4"))


def test_copy_local_file_returns_same_path_for_self_copy(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"media")

    copied = MSCStorage().copy_local_file(str(source), str(source))

    assert copied == str(source)


def test_remote_file_destination_rejects_local_path_without_filename() -> None:
    with pytest.raises(ValueError, match="does not include a file name"):
        MSCStorage()._remote_file_destination("annotations/", "")


def test_upload_if_remote_rejects_missing_source(tmp_path: Path) -> None:
    storage = MSCStorage()

    with patch.object(storage, "get_client_and_path") as resolve:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            storage.upload_if_remote(str(tmp_path / "missing.json"), "s3://bucket/labels.json")

    resolve.assert_not_called()


def test_upload_if_remote_rejects_source_that_is_not_file_or_directory(tmp_path: Path) -> None:
    source = tmp_path / "source.pipe"
    os.mkfifo(source)
    target_client = MagicMock()

    with patch(
        "core.utils.multistorage.msc.resolve_storage_client",
        return_value=(target_client, "outputs/source.pipe"),
    ):
        with pytest.raises(ValueError, match="not a file or directory"):
            MSCStorage().upload_if_remote(str(source), "s3://bucket/outputs/source.pipe")

    target_client.upload_file.assert_not_called()
    target_client.sync_from.assert_not_called()
