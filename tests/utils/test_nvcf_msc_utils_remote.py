# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import nvcf_msc_utils as utils
import pytest
from al_utils.media_paths import IMAGE_EXTS, VIDEO_EXTS


@pytest.fixture
def mock_msc():
    with patch("nvcf_msc_utils.msc") as m:
        yield m


def test_convert_to_msc_url_with_mapping(monkeypatch):
    # Setup env
    config = {"path_mapping": {"https://mybucket/": "msc://profile/mybucket/"}}
    monkeypatch.setenv("MULTISTORAGECLIENT_CONFIGURATION", json.dumps(config))

    url = "https://mybucket/video.mp4"
    converted = utils.convert_to_msc_url(url)
    assert converted == "msc://profile/mybucket/video.mp4"

    # Non-matching
    assert utils.convert_to_msc_url("https://other/file.mp4") == "https://other/file.mp4"


def test_is_remote_path():
    assert utils.is_remote_path("s3://bucket")
    assert utils.is_remote_path("msc://profile")
    assert utils.is_remote_path("gs://bucket")
    assert utils.is_remote_path("http://host")
    assert utils.is_remote_path("https://host")
    assert not utils.is_remote_path("/local/path")
    assert not utils.is_remote_path("relative/path")


def test_sync_remote_to_local_file_direct(mock_msc, tmp_path):
    # Case: URL looks like a file (ends with extension in VIDEO_EXTENSIONS)
    remote = "s3://bucket/video.mp4"
    local = tmp_path / "downloads"

    # Mock download to verify path
    utils.sync_remote_to_local(remote, str(local))

    mock_msc.download_file.assert_called_once_with(remote, str(local / "video.mp4"))


def test_video_extensions_match_media_path_policy():
    assert set(utils.VIDEO_EXTENSIONS) == VIDEO_EXTS | IMAGE_EXTS
    assert ".mkv" not in utils.VIDEO_EXTENSIONS
    assert ".avi" not in utils.VIDEO_EXTENSIONS
    assert ".webm" not in utils.VIDEO_EXTENSIONS


def test_sync_remote_to_local_http_file_with_query(mock_msc, tmp_path):
    # Case: URL has query params
    remote = "https://host/video.mp4?token=123"
    local = tmp_path / "downloads"

    with patch("nvcf_msc_utils._download_http_file") as download_http:
        utils.sync_remote_to_local(remote, str(local))

    # Should treat as video.mp4
    download_http.assert_called_once_with(remote, str(local / "video.mp4"))
    mock_msc.download_file.assert_not_called()


def test_sync_remote_to_local_decodes_http_basename(mock_msc, tmp_path):
    remote = "https://host/input/lv/C000025%20%E4%B8%AD%E8%8F%AF.mp4?token=123"
    local = tmp_path / "downloads"

    with patch("nvcf_msc_utils._download_http_file") as download_http:
        utils.sync_remote_to_local(remote, str(local), extensions=(".mp4",))

    download_http.assert_called_once_with(remote, str(local / "C000025 中華.mp4"))
    mock_msc.download_file.assert_not_called()


def test_sync_remote_to_local_decodes_percent_encoded_slash_in_basename(mock_msc, tmp_path):
    remote = "https://host/input/lv/folder%2Fclip%20name.mp4"
    local = tmp_path / "downloads"

    with patch("nvcf_msc_utils._download_http_file") as download_http:
        utils.sync_remote_to_local(remote, str(local), extensions=(".mp4",))

    download_http.assert_called_once_with(remote, str(local / "clip name.mp4"))
    mock_msc.download_file.assert_not_called()


def test_sync_remote_to_local_unmapped_http_directory_fails_clearly(mock_msc, tmp_path):
    with pytest.raises(RuntimeError, match=r"Direct HTTP\(S\) remote input must point to a single supported file"):
        utils.sync_remote_to_local("http://host/input/lv/", str(tmp_path), extensions=(".mp4",))

    mock_msc.list.assert_not_called()
    mock_msc.download_file.assert_not_called()


def test_localize_path_to_dir_decodes_remote_basename(tmp_path, monkeypatch):
    remote = "https://host/input/lv/C000025%20%E4%B8%AD%E8%8F%AF.mp4?token=123"

    def fake_sync(remote_path: str, local_dir: str, extensions=None, verbose=True) -> str:
        assert remote_path == remote
        Path(local_dir, "C000025 中華.mp4").write_bytes(b"video")
        return local_dir

    monkeypatch.setattr(utils, "sync_remote_to_local", fake_sync)

    out = utils.localize_path_to_dir(
        remote,
        dst_dir=tmp_path,
        logger=None,
        extensions=(".mp4",),
        config_dir=tmp_path,
        repo_root=tmp_path,
    )

    assert out == tmp_path / "C000025 中華.mp4"


def test_sync_remote_to_local_mapped_https_uses_msc(mock_msc, tmp_path, monkeypatch):
    config = {"path_mapping": {"https://host/": "msc://profile/bucket/"}}
    monkeypatch.setenv("MULTISTORAGECLIENT_CONFIGURATION", json.dumps(config))
    remote = "https://host/video.mp4?token=123"
    local = tmp_path / "downloads"

    with patch("nvcf_msc_utils._download_http_file") as download_http:
        utils.sync_remote_to_local(remote, str(local))

    mock_msc.download_file.assert_called_once_with("msc://profile/bucket/video.mp4?token=123", str(local / "video.mp4"))
    download_http.assert_not_called()


def test_sync_remote_to_local_dir(mock_msc, tmp_path):
    # Case: URL is directory, list returns items
    remote = "s3://bucket/videos/"
    local = tmp_path / "downloads"

    # Mock msc.list items
    item1 = MagicMock()
    item1.key = "s3://bucket/videos/v1.mp4"
    item1.type = "file"

    item2 = MagicMock()
    item2.key = "s3://bucket/videos/ignore.txt"  # Wrong extension
    item2.type = "file"

    item3 = MagicMock()
    item3.key = "s3://bucket/videos/sub/v2.mp4"  # Nested
    item3.type = "file"

    # Simulate a directory placeholder that should be ignored
    item4 = MagicMock()
    item4.key = "s3://bucket/videos/sub/"
    item4.type = "directory"

    mock_msc.list.return_value = [item1, item2, item3, item4]

    utils.sync_remote_to_local(remote, str(local), extensions=(".mp4",))

    # Check calls
    assert mock_msc.download_file.call_count == 2
    mock_msc.download_file.assert_has_calls(
        [
            call("s3://bucket/videos/v1.mp4", str(local / "v1.mp4")),
            call("s3://bucket/videos/sub/v2.mp4", str(local / "sub/v2.mp4")),
        ],
        any_order=True,
    )


def test_sync_local_to_remote(mock_msc, tmp_path):
    # Create local files
    (tmp_path / "f1.txt").touch()
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "f2.txt").touch()

    remote = "s3://bucket/dest"
    utils.sync_local_to_remote(str(tmp_path), remote)

    # Verify uploads
    expected_calls = [
        call(f"{remote}/f1.txt", str(tmp_path / "f1.txt")),
        call(f"{remote}/sub/f2.txt", str(tmp_path / "sub/f2.txt")),
    ]
    mock_msc.upload_file.assert_has_calls(expected_calls, any_order=True)


def test_sync_local_to_remote_http_failure(tmp_path, monkeypatch):
    monkeypatch.delenv("MULTISTORAGECLIENT_CONFIGURATION", raising=False)
    monkeypatch.delenv("MSC_CONFIG", raising=False)
    # Upload to HTTP(S) without mapping should fail.
    remote = "http://host/dest"
    with pytest.raises(RuntimeError, match=r"Cannot upload to HTTP\(S\)"):
        utils.sync_local_to_remote(str(tmp_path), remote)


def test_setup_msc_config(monkeypatch, tmp_path):
    # Test config writing
    cfg = {"path_mapping": {"a": "b"}}
    monkeypatch.setenv("MULTISTORAGECLIENT_CONFIGURATION", json.dumps(cfg))

    utils.setup_msc_config()

    # Verify file written to /tmp/msc_config.json
    # Note: the module hardcodes "/tmp/msc_config.json"
    p = Path("/tmp/msc_config.json")
    if p.exists():
        content = json.loads(p.read_text())
        assert content == cfg
        assert os.environ["MSC_CONFIG"] == str(p)


def test_nvcf_progress_tracker(monkeypatch, tmp_path):
    progress_file = tmp_path / "progress.json"
    monkeypatch.setenv("NVCT_TASK_ID", "task1")
    monkeypatch.setenv("NVCT_PROGRESS_FILE_PATH", str(progress_file))

    tracker = utils.NVCFProgressTracker()
    assert tracker.is_nvcf

    tracker.update(50, "Halfway")

    assert progress_file.exists()
    data = json.loads(progress_file.read_text())
    assert data["taskId"] == "task1"
    assert data["percentComplete"] == 50
    assert data["message"] == "Halfway"


def test_detect_nvcf_vlm_endpoint(monkeypatch):
    monkeypatch.setenv("VLM_BASE_URL", "http://vlm")
    monkeypatch.setenv("VLM_MODEL", "model1")

    url, model = utils.detect_nvcf_vlm_endpoint()
    assert url == "http://vlm"
    assert model == "model1"


def test_detect_nvcf_vlm_endpoint_missing(monkeypatch):
    monkeypatch.delenv("VLM_BASE_URL", raising=False)
    monkeypatch.delenv("VLM_MODEL", raising=False)

    url, model = utils.detect_nvcf_vlm_endpoint()
    assert url is None
    assert model is None


def test_detect_nvcf_llm_endpoint(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://llm")
    monkeypatch.setenv("LLM_MODEL", "llm-model1")

    url, model = utils.detect_nvcf_llm_endpoint()
    assert url == "http://llm"
    assert model == "llm-model1"


def test_detect_nvcf_llm_endpoint_missing(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    url, model = utils.detect_nvcf_llm_endpoint()
    assert url is None
    assert model is None


def test_sync_remote_to_local_no_files_found(mock_msc, tmp_path):
    remote = "s3://bucket/videos/"
    local = tmp_path / "downloads"

    item1 = MagicMock()
    item1.key = "s3://bucket/videos/ignore.txt"
    item1.type = "file"

    mock_msc.list.return_value = [item1]

    with pytest.raises(RuntimeError, match="No files with extensions"):
        utils.sync_remote_to_local(remote, str(local), extensions=(".mp4",))


def test_sync_remote_to_local_skip_placeholders(mock_msc, tmp_path, capsys):
    remote = "s3://bucket/videos/"
    local = tmp_path / "downloads"

    # Placeholders
    item1 = MagicMock()
    item1.key = "s3://bucket/videos/."
    item1.type = "file"

    item2 = MagicMock()
    item2.key = "s3://bucket/videos/subdir/."
    item2.type = "file"

    mock_msc.list.return_value = [item1, item2]

    utils.sync_remote_to_local(remote, str(local), verbose=True)

    # Should not download anything
    mock_msc.download_file.assert_not_called()

    # Check stdout for skipping messages
    captured = capsys.readouterr()
    assert "Skipping" in captured.out
