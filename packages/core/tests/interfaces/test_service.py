# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from core.cost_performance import model_usage_entry, record_model_call
from core.interfaces import ServiceInterface
from core.models import DataEntry
from pydantic import ValidationError


class _StubService(ServiceInterface):
    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)
        self.observed_args: argparse.Namespace | None = None
        self.observed_data_entries: list[DataEntry] | None = None

    def execute(self, args: argparse.Namespace, data_entries: list[DataEntry]) -> None:
        self.observed_args = args
        self.observed_data_entries = data_entries


class _ModelCallService(ServiceInterface):
    def __init__(self) -> None:
        super().__init__("model-call-service")

    def execute(self, args: argparse.Namespace, data_entries: list[DataEntry]) -> None:
        for data_entry in data_entries:
            with model_usage_entry(data_entry):
                record_model_call(
                    kind="llm",
                    provider="openai-compatible",
                    model="test-llm",
                    endpoint_url="http://localhost:8080/v1",
                    elapsed_s=0.5,
                    retry_count=2,
                    success=True,
                    usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                )


def _seed_remote_data(
    _storage: object,
    _remote_path: str,
    local_path: str,
    *,
    is_file: bool,
) -> str:
    assert not is_file
    destination = Path(local_path)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "seed.txt").write_text("seed", encoding="utf-8")
    return local_path


def test_common_args_added_to_parser() -> None:
    cli = _StubService("test")
    parser = argparse.ArgumentParser()
    cli._add_common_args(parser)
    payload = json.dumps(
        [{"id": "x", "media_path": "path/to/media/file.mp4", "data_path": "path/to/data/directory"}]
    )
    args = parser.parse_args(["--input", payload])

    assert args.input == payload
    assert args.input_file is None
    assert args.dev_data_root is None
    assert args.log_level == "INFO"
    assert not hasattr(args, "verbose")


def test_common_args_log_level() -> None:
    cli = _StubService("test")
    parser = argparse.ArgumentParser()
    cli._add_common_args(parser)
    payload = json.dumps([{"media_path": "m.mp4", "data_path": "d/"}])
    args = parser.parse_args(["--log-level", "DEBUG", "--input", payload])
    assert args.log_level == "DEBUG"


def test_get_data_entries_from_input() -> None:
    cli = _StubService("test")
    parser = argparse.ArgumentParser()
    cli._add_common_args(parser)
    payload = json.dumps(
        [
            {"id": "a", "media_path": "m1.mp4", "data_path": "d1/"},
            {"media_path": "m2.mp4", "data_path": "d2/"},
        ]
    )
    args = parser.parse_args(["--input", payload])
    entries = cli._get_data_entries(args)
    assert len(entries) == 2
    assert entries[0].id == "a"
    assert entries[0].media_path == "m1.mp4"
    assert entries[0].data_path == "d1/"
    assert entries[1].media_path == "m2.mp4"
    assert entries[1].data_path == "d2/"


def test_get_data_entries_rejects_unknown_fields() -> None:
    """``DataEntry`` is locked down (extras=forbid). A typo'd field
    should be caught at input parse time, not silently swallowed."""
    cli = _StubService("test")
    parser = argparse.ArgumentParser()
    cli._add_common_args(parser)
    payload = json.dumps([{"media_path": "m.mp4", "data_path": "d/", "note": "extra"}])
    args = parser.parse_args(["--input", payload])
    with pytest.raises(ValidationError, match="extra_forbidden"):
        cli._get_data_entries(args)


def test_get_data_entries_from_input_file(tmp_path: Path) -> None:
    cli = _StubService("test")
    parser = argparse.ArgumentParser()
    cli._add_common_args(parser)
    path = tmp_path / "entries.jsonl"
    path.write_text(
        '{"media_path": "a.mp4", "data_path": "out/a/"}\n'
        "\n"
        '{"id": "b", "media_path": "b.mp4", "data_path": "out/b/"}\n',
        encoding="utf-8",
    )
    args = parser.parse_args(["--input-file", str(path)])
    entries = cli._get_data_entries(args)
    assert entries[0].media_path == "a.mp4"
    assert entries[0].data_path == "out/a/"
    assert entries[1].id == "b"
    assert entries[1].media_path == "b.mp4"
    assert entries[1].data_path == "out/b/"


def test_get_data_entries_rejects_data_directory_path_alias() -> None:
    cli = _StubService("test")
    parser = argparse.ArgumentParser()
    cli._add_common_args(parser)
    payload = json.dumps([{"media_path": "m.mp4", "data_directory_path": "daft/"}])
    args = parser.parse_args(["--input", payload])

    with pytest.raises(ValidationError, match="data_path"):
        cli._get_data_entries(args)


def test_get_data_entries_without_input_returns_empty() -> None:
    cli = _StubService("test")
    parser = argparse.ArgumentParser()
    cli._add_common_args(parser)
    args = parser.parse_args([])

    assert cli._get_data_entries(args) == []


def test_dev_data_root_rewrites_entries_and_copies_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "seed.txt").write_text("seed", encoding="utf-8")
    dev_root = tmp_path / "dev-root"
    payload = json.dumps([{"id": "entry-1", "media_path": "m.mp4", "data_path": str(source)}])
    cli = _StubService("test")

    with patch.object(
        sys,
        "argv",
        ["test_service", "--dev-data-root", str(dev_root), "--input", payload],
    ):
        cli.run()

    assert cli.observed_data_entries is not None
    destination = dev_root / "entry-1"
    assert cli.observed_data_entries[0].data_path == str(destination)
    assert cli.observed_data_entries[0].media_path == "m.mp4"
    assert (destination / "seed.txt").read_text(encoding="utf-8") == "seed"
    assert (source / "seed.txt").read_text(encoding="utf-8") == "seed"


def test_run_emits_generic_cost_performance_report_for_model_calls(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    payload = json.dumps([{"id": "entry-1", "media_path": "m.mp4", "data_path": str(scene)}])
    service = _ModelCallService()

    with (
        patch.object(sys, "argv", ["svc", "--input", payload]),
        patch("core.interfaces.service.write_cost_performance_report") as write_report,
    ):
        service.run()

    write_report.assert_called_once()
    kwargs = write_report.call_args.kwargs
    assert kwargs["service_name"] == "model-call-service"
    assert kwargs["model_usage"].summary_json()["llm_calls_observed"] == 1
    assert kwargs["model_usage"].summary_json()["retry_count_observed"] == 2
    assert kwargs["model_usage"].summary_json()["token_counts_observed"]["total_tokens"] == 5
    assert not (scene / "sidecars" / "cost_performance" / "report.json").exists()


def test_dev_data_root_overwrites_existing_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "seed.txt").write_text("fresh", encoding="utf-8")
    dev_root = tmp_path / "dev-root"
    destination = dev_root / "entry-1"
    destination.mkdir(parents=True)
    (destination / "old.txt").write_text("old", encoding="utf-8")
    cli = _StubService("test")

    rewritten = cli._copy_data_entries_to_dev_root(
        [DataEntry(id="entry-1", media_path="m.mp4", data_path=str(source))],
        str(dev_root),
    )

    assert rewritten[0].data_path == str(destination)
    assert (destination / "seed.txt").read_text(encoding="utf-8") == "fresh"
    assert not (destination / "old.txt").exists()


def test_dev_data_root_downloads_remote_source(tmp_path: Path) -> None:
    dev_root = tmp_path / "dev-root"
    cli = _StubService("test")
    data_entry = DataEntry(id="entry-1", media_path="m.mp4", data_path="s3://bucket/data")

    with patch(
        "core.utils.dev_data_root.MSCStorage.download_if_remote",
        autospec=True,
        side_effect=_seed_remote_data,
    ) as download_if_remote:
        rewritten = cli._copy_data_entries_to_dev_root([data_entry], str(dev_root))

    destination = dev_root / "entry-1"
    assert rewritten[0].data_path == str(destination)
    assert (destination / "seed.txt").read_text(encoding="utf-8") == "seed"
    download_if_remote.assert_called_once()


def test_dev_data_root_uploads_local_source_to_remote_root(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "seed.txt").write_text("seed", encoding="utf-8")
    cli = _StubService("test")

    def assert_upload(
        _storage: object,
        local_path: str,
        remote_path: str,
        *,
        delete_unmatched_files: bool,
    ) -> str:
        assert (Path(local_path) / "seed.txt").read_text(encoding="utf-8") == "seed"
        assert remote_path == "s3://bucket/dev-root/entry-1"
        assert delete_unmatched_files is True
        return remote_path

    with patch(
        "core.utils.dev_data_root.MSCStorage.upload_if_remote",
        autospec=True,
        side_effect=assert_upload,
    ) as upload_if_remote:
        rewritten = cli._copy_data_entries_to_dev_root(
            [DataEntry(id="entry-1", media_path="m.mp4", data_path=str(source))],
            "s3://bucket/dev-root/",
        )

    assert rewritten[0].data_path == "s3://bucket/dev-root/entry-1"
    upload_if_remote.assert_called_once()


def test_dev_data_root_remote_root_downloads_remote_source_before_uploading() -> None:
    cli = _StubService("test")

    def assert_upload(
        _storage: object,
        local_path: str,
        remote_path: str,
        *,
        delete_unmatched_files: bool,
    ) -> str:
        assert (Path(local_path) / "seed.txt").read_text(encoding="utf-8") == "seed"
        assert remote_path == "s3://bucket/dev-root/entry-1"
        assert delete_unmatched_files is True
        return remote_path

    with (
        patch(
            "core.utils.dev_data_root.MSCStorage.download_if_remote",
            autospec=True,
            side_effect=_seed_remote_data,
        ) as download_if_remote,
        patch(
            "core.utils.dev_data_root.MSCStorage.upload_if_remote",
            autospec=True,
            side_effect=assert_upload,
        ) as upload_if_remote,
    ):
        rewritten = cli._copy_data_entries_to_dev_root(
            [DataEntry(id="entry-1", media_path="m.mp4", data_path="s3://source/data")],
            "s3://bucket/dev-root",
        )

    assert rewritten[0].data_path == "s3://bucket/dev-root/entry-1"
    download_if_remote.assert_called_once()
    upload_if_remote.assert_called_once()


def test_dev_data_root_rejects_file_root(tmp_path: Path) -> None:
    dev_root = tmp_path / "dev-root"
    dev_root.write_text("not a directory", encoding="utf-8")
    cli = _StubService("test")

    with pytest.raises(ValueError, match="not a directory"):
        cli._copy_data_entries_to_dev_root([], str(dev_root))


def test_dev_data_root_duplicate_ids_raise_before_copying(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    dev_root = tmp_path / "dev-root"
    destination = dev_root / "entry-1"
    destination.mkdir(parents=True)
    marker = destination / "old.txt"
    marker.write_text("old", encoding="utf-8")
    entries = [
        DataEntry(id="entry-1", media_path="m1.mp4", data_path=str(source)),
        DataEntry(id="entry-1", media_path="m2.mp4", data_path=str(source)),
    ]
    cli = _StubService("test")

    with pytest.raises(ValueError, match="Duplicate data entry id"):
        cli._copy_data_entries_to_dev_root(entries, str(dev_root))

    assert marker.read_text(encoding="utf-8") == "old"


def test_dev_data_root_rejects_id_that_escapes_root(tmp_path: Path) -> None:
    cli = _StubService("test")
    entry = DataEntry(id="../outside", media_path="m.mp4", data_path="s3://bucket/data")

    with pytest.raises(ValueError, match="escapes dev data root"):
        cli._copy_data_entries_to_dev_root([entry], str(tmp_path / "dev-root"))


def test_dev_data_root_rejects_id_that_maps_to_root(tmp_path: Path) -> None:
    cli = _StubService("test")
    entry = DataEntry(id=".", media_path="m.mp4", data_path="s3://bucket/data")

    with pytest.raises(ValueError, match="does not identify a child directory"):
        cli._copy_data_entries_to_dev_root([entry], str(tmp_path / "dev-root"))


def test_remote_dev_data_root_rejects_id_that_escapes_root() -> None:
    cli = _StubService("test")
    entry = DataEntry(id="../outside", media_path="m.mp4", data_path="s3://bucket/data")

    with pytest.raises(ValueError, match="escapes dev data root"):
        cli._copy_data_entries_to_dev_root([entry], "s3://bucket/dev-root")


def test_remote_dev_data_root_rejects_id_that_maps_to_root() -> None:
    cli = _StubService("test")
    entry = DataEntry(id=".", media_path="m.mp4", data_path="s3://bucket/data")

    with pytest.raises(ValueError, match="does not identify a child directory"):
        cli._copy_data_entries_to_dev_root([entry], "s3://bucket/dev-root")


def test_dev_data_root_rejects_ids_that_map_to_same_destination(tmp_path: Path) -> None:
    cli = _StubService("test")
    entries = [
        DataEntry(id="entry-1", media_path="m1.mp4", data_path="s3://bucket/one"),
        DataEntry(id="nested/../entry-1", media_path="m2.mp4", data_path="s3://bucket/two"),
    ]

    with pytest.raises(ValueError, match="map to the same"):
        cli._copy_data_entries_to_dev_root(entries, str(tmp_path / "dev-root"))


def test_remote_dev_data_root_rejects_ids_that_map_to_same_destination() -> None:
    cli = _StubService("test")
    entries = [
        DataEntry(id="entry-1", media_path="m1.mp4", data_path="s3://bucket/one"),
        DataEntry(id="nested/../entry-1", media_path="m2.mp4", data_path="s3://bucket/two"),
    ]

    with (
        patch("core.utils.dev_data_root.MSCStorage.upload_if_remote") as upload_if_remote,
        pytest.raises(ValueError, match="map to the same"),
    ):
        cli._copy_data_entries_to_dev_root(entries, "s3://bucket/dev-root")

    upload_if_remote.assert_not_called()


def test_dev_data_root_rejects_existing_non_directory_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    dev_root = tmp_path / "dev-root"
    dev_root.mkdir()
    destination = dev_root / "entry-1"
    destination.write_text("not a directory", encoding="utf-8")
    cli = _StubService("test")

    with pytest.raises(ValueError, match="not a directory"):
        cli._copy_data_entries_to_dev_root(
            [DataEntry(id="entry-1", media_path="m.mp4", data_path=str(source))],
            str(dev_root),
        )

    assert destination.read_text(encoding="utf-8") == "not a directory"


def test_dev_data_root_rejects_local_source_file(tmp_path: Path) -> None:
    source = tmp_path / "source-file"
    source.write_text("not a directory", encoding="utf-8")
    cli = _StubService("test")

    with pytest.raises(ValueError, match="Local data path"):
        cli._copy_data_entries_to_dev_root(
            [DataEntry(id="entry-1", media_path="m.mp4", data_path=str(source))],
            str(tmp_path / "dev-root"),
        )


def test_remote_dev_data_root_rejects_local_source_file_before_uploading(tmp_path: Path) -> None:
    source = tmp_path / "source-file"
    source.write_text("not a directory", encoding="utf-8")
    cli = _StubService("test")

    with (
        patch("core.utils.dev_data_root.MSCStorage.upload_if_remote") as upload_if_remote,
        pytest.raises(ValueError, match="Local data path"),
    ):
        cli._copy_data_entries_to_dev_root(
            [DataEntry(id="entry-1", media_path="m.mp4", data_path=str(source))],
            "s3://bucket/dev-root",
        )

    upload_if_remote.assert_not_called()


def test_dev_data_root_rejects_source_equal_to_destination(tmp_path: Path) -> None:
    dev_root = tmp_path / "dev-root"
    source = dev_root / "entry-1"
    source.mkdir(parents=True)
    cli = _StubService("test")

    with pytest.raises(ValueError, match="same as source data_path"):
        cli._copy_data_entries_to_dev_root(
            [DataEntry(id="entry-1", media_path="m.mp4", data_path=str(source))],
            str(dev_root),
        )


def test_dev_data_root_rejects_destination_inside_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    dev_root = source / "dev-root"
    cli = _StubService("test")

    with pytest.raises(ValueError, match="--dev-data-root"):
        cli._copy_data_entries_to_dev_root(
            [DataEntry(id="entry-1", media_path="m.mp4", data_path=str(source))],
            str(dev_root),
        )


def test_dev_data_root_missing_source_creates_empty_destination(
    tmp_path: Path,
) -> None:
    missing_source = tmp_path / "missing_source"
    dev_root = tmp_path / "dev-root"
    destination = dev_root / "entry-1"
    destination.mkdir(parents=True)
    marker = destination / "old.txt"
    marker.write_text("old", encoding="utf-8")
    cli = _StubService("test")

    rewritten = cli._copy_data_entries_to_dev_root(
        [DataEntry(id="entry-1", media_path="m.mp4", data_path=str(missing_source))],
        str(dev_root),
    )

    assert rewritten[0].data_path == str(destination)
    assert destination.is_dir()
    assert not marker.exists()
    assert list(destination.iterdir()) == []


def test_get_data_entries_input_not_array() -> None:
    cli = _StubService("test")
    parser = argparse.ArgumentParser()
    cli._add_common_args(parser)
    args = parser.parse_args(["--input", '{"media_path": "x", "data_path": "y"}'])
    with pytest.raises(ValueError, match="JSON array"):
        cli._get_data_entries(args)


def test_get_data_entries_rejects_non_object_records() -> None:
    cli = _StubService("test")
    parser = argparse.ArgumentParser()
    cli._add_common_args(parser)
    args = parser.parse_args(["--input", json.dumps(["not an object"])])

    with pytest.raises(ValueError, match="Each entry must be a JSON object"):
        cli._get_data_entries(args)


def test_get_data_entries_missing_keys() -> None:
    cli = _StubService("test")
    parser = argparse.ArgumentParser()
    cli._add_common_args(parser)
    args = parser.parse_args(["--input", json.dumps([{"media_path": "only"}])])
    with pytest.raises(ValidationError, match="data_path"):
        cli._get_data_entries(args)


def test_add_service_args_is_overridable() -> None:
    class _CustomService(ServiceInterface):
        def add_service_args(self, parser: argparse.ArgumentParser) -> None:
            parser.add_argument("--threshold", type=float, default=0.5)

        def execute(self, args: argparse.Namespace, data_entries: list[DataEntry]) -> None:
            pass

    cli = _CustomService("test")
    payload = json.dumps([])
    argv = ["test_service", "--threshold", "0.8", "--input", payload]
    with patch.object(sys, "argv", argv):
        cli.run()
