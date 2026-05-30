# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for nvcf_msc_utils (MSC config, path_mapping, remote detection). No real network download."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from nvcf_msc_utils import (
    _download_http_file,
    convert_to_msc_url,
    get_msc_path_mapping,
    http_timeout_seconds,
    is_remote_path,
    normalize_remote_prefix,
    normalize_remote_prefix_for_compare,
    remote_child_prefix,
    setup_msc_config,
)


def test_setup_msc_config_no_env_no_crash():
    """With MULTISTORAGECLIENT_CONFIGURATION unset, setup_msc_config does nothing and does not crash."""
    orig = os.environ.pop("MULTISTORAGECLIENT_CONFIGURATION", None)
    orig_msc = os.environ.pop("MSC_CONFIG", None)
    try:
        setup_msc_config()
        assert "MSC_CONFIG" not in os.environ or os.environ.get("MSC_CONFIG") == orig_msc
    finally:
        if orig is not None:
            os.environ["MULTISTORAGECLIENT_CONFIGURATION"] = orig
        if orig_msc is not None:
            os.environ["MSC_CONFIG"] = orig_msc


def test_setup_msc_config_with_minimal_json(tmp_path):
    """With minimal JSON, setup writes file and sets MSC_CONFIG; get_msc_path_mapping works."""
    config = '{"path_mapping": {"https://bucket.s3.amazonaws.com/": "s3://bucket/"}}'
    orig = os.environ.pop("MULTISTORAGECLIENT_CONFIGURATION", None)
    orig_msc = os.environ.pop("MSC_CONFIG", None)
    try:
        os.environ["MULTISTORAGECLIENT_CONFIGURATION"] = config
        setup_msc_config()
        assert "MSC_CONFIG" in os.environ
        config_path = os.environ["MSC_CONFIG"]
        assert Path(config_path).exists()
        mapping = get_msc_path_mapping()
        assert "https://bucket.s3.amazonaws.com/" in mapping
        assert mapping["https://bucket.s3.amazonaws.com/"] == "s3://bucket/"
    finally:
        if orig is not None:
            os.environ["MULTISTORAGECLIENT_CONFIGURATION"] = orig
        elif "MULTISTORAGECLIENT_CONFIGURATION" in os.environ:
            del os.environ["MULTISTORAGECLIENT_CONFIGURATION"]
        if orig_msc is not None:
            os.environ["MSC_CONFIG"] = orig_msc
        elif "MSC_CONFIG" in os.environ:
            del os.environ["MSC_CONFIG"]


def test_convert_to_msc_url_with_mapping():
    """convert_to_msc_url rewrites https prefix to msc when path_mapping is set."""
    config = '{"path_mapping": {"https://b.s3.amazonaws.com/": "s3://b/"}}'
    orig = os.environ.get("MULTISTORAGECLIENT_CONFIGURATION")
    orig_msc = os.environ.get("MSC_CONFIG")
    try:
        os.environ["MULTISTORAGECLIENT_CONFIGURATION"] = config
        setup_msc_config()
        out = convert_to_msc_url("https://b.s3.amazonaws.com/prefix/video.mp4")
        assert out == "s3://b/prefix/video.mp4"
        assert convert_to_msc_url("msc://profile/path") == "msc://profile/path"
        assert convert_to_msc_url("s3://other/key") == "s3://other/key"
    finally:
        if orig is not None:
            os.environ["MULTISTORAGECLIENT_CONFIGURATION"] = orig
        else:
            os.environ.pop("MULTISTORAGECLIENT_CONFIGURATION", None)
        if orig_msc is not None:
            os.environ["MSC_CONFIG"] = orig_msc
        else:
            os.environ.pop("MSC_CONFIG", None)


def test_is_remote_path():
    """is_remote_path recognizes s3/msc/gs/http(s) and rejects local paths."""
    assert is_remote_path("s3://bucket/key") is True
    assert is_remote_path("msc://p/k") is True
    assert is_remote_path("https://example.com/f") is True
    assert is_remote_path("http://example.com/f") is True
    assert is_remote_path("/local/path") is False
    assert is_remote_path("./relative") is False
    assert is_remote_path("") is False


def test_normalize_remote_prefix_collapses_path_slashes():
    """normalize_remote_prefix preserves scheme and trailing slash while cleaning path separators."""
    assert normalize_remote_prefix("s3://bucket//prefix///") == "s3://bucket/prefix/"
    assert normalize_remote_prefix("msc://profile///path//file.mp4") == "msc://profile/path/file.mp4"
    assert normalize_remote_prefix("/local//path") == "/local//path"


def test_normalize_remote_prefix_for_compare_ignores_trailing_slash():
    assert normalize_remote_prefix_for_compare("s3://bucket//out/logs/") == "s3://bucket/out/logs"
    assert normalize_remote_prefix_for_compare("s3://bucket/out/logs") == "s3://bucket/out/logs"


def test_remote_child_prefix_joins_without_local_path_semantics():
    assert remote_child_prefix("s3://bucket//out/", "logs") == "s3://bucket/out/logs"
    assert remote_child_prefix("https://example.test/base", "/logs/") == "https://example.test/base/logs"


def test_http_timeout_seconds_env_override(monkeypatch):
    monkeypatch.delenv("HTTP_TIMEOUT_S", raising=False)
    assert http_timeout_seconds() == 120.0

    monkeypatch.setenv("HTTP_TIMEOUT_S", "7.5")
    assert http_timeout_seconds() == 7.5

    monkeypatch.setenv("HTTP_TIMEOUT_S", "not-a-number")
    assert http_timeout_seconds() == 120.0

    monkeypatch.setenv("HTTP_TIMEOUT_S", "0")
    assert http_timeout_seconds() == 120.0

    monkeypatch.setenv("HTTP_TIMEOUT_S", "-5")
    assert http_timeout_seconds() == 120.0


def test_download_http_file_rejects_non_http_scheme(tmp_path: Path):
    with pytest.raises(ValueError, match="http:// or https://"):
        _download_http_file("file:///tmp/config.yaml", str(tmp_path / "config.yaml"))
