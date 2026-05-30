# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for config loader: data[] auto-extend via dotlist overrides."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import config.loader
import pytest

PIPELINE_EXAMPLE_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "pipeline_example.yaml"


class _HttpResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def _base_overrides(tmp_path: Path) -> list[str]:
    return [
        "super_resolution.enabled=false",
        "detection_and_tracking.enabled=false",
        "vlm_json.enabled=false",
        "mcq_generation.enabled=false",
        f"data.0.inputs.video_path={tmp_path / 'v0.mp4'}",
        f"data.0.output.out_dir={tmp_path / 'out0'}",
        "endpoints.vlm.url=http://fake/v1",
        "endpoints.vlm.model=fake-vlm",
        "endpoints.llm.url=http://fake/v1",
        "endpoints.llm.model=fake-llm",
    ]


def test_single_sample_baseline(tmp_path: Path) -> None:
    """Base case: single data[0] override works as before."""
    cfg, _ = config.loader.load_config_with_overrides(
        str(PIPELINE_EXAMPLE_CONFIG),
        _base_overrides(tmp_path),
    )
    assert len(cfg["data"]) == 1
    assert str(tmp_path / "v0.mp4") in cfg["data"][0]["inputs"]["video_path"]


def test_two_samples_via_dotlist(tmp_path: Path) -> None:
    """data.1.* override auto-extends the list to 2 entries."""
    overrides = _base_overrides(tmp_path) + [
        f"data.1.inputs.video_path={tmp_path / 'v1.mp4'}",
        f"data.1.output.out_dir={tmp_path / 'out1'}",
    ]
    cfg, _ = config.loader.load_config_with_overrides(str(PIPELINE_EXAMPLE_CONFIG), overrides)
    assert len(cfg["data"]) == 2
    assert str(tmp_path / "v1.mp4") in cfg["data"][1]["inputs"]["video_path"]
    assert str(tmp_path / "out1") in cfg["data"][1]["output"]["out_dir"]


def test_http_config_downloads_without_msc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MULTISTORAGECLIENT_CONFIGURATION", raising=False)
    monkeypatch.delenv("MSC_CONFIG", raising=False)
    payload = b"pipeline:\n  gpu_ids: '0'\ndata: []\n"

    for scheme in ("http", "https"):
        with (
            patch("nvcf_msc_utils.urllib.request.urlopen", return_value=_HttpResponse(payload)) as urlopen,
            patch("config.loader.msc.download_file") as download_file,
        ):
            cfg, _ = config.loader.load_config_with_overrides(f"{scheme}://example.test/pipeline.yaml", [])

        assert cfg["pipeline"]["gpu_ids"] == "0"
        urlopen.assert_called_once()
        download_file.assert_not_called()


def test_mapped_http_config_uses_msc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "MULTISTORAGECLIENT_CONFIGURATION",
        '{"path_mapping": {"https://example.test/": "msc://profile/bucket/"}}',
    )
    payload = "pipeline:\n  gpu_ids: '1'\ndata: []\n"

    def fake_download(_remote: str, local: str) -> None:
        Path(local).write_text(payload, encoding="utf-8")

    with (
        patch("nvcf_msc_utils.urllib.request.urlopen") as urlopen,
        patch("config.loader.msc.download_file", side_effect=fake_download) as download_file,
    ):
        cfg, _ = config.loader.load_config_with_overrides("https://example.test/pipeline.yaml", [])

    assert cfg["pipeline"]["gpu_ids"] == "1"
    assert download_file.call_args.args[0] == "msc://profile/bucket/pipeline.yaml"
    urlopen.assert_not_called()


def test_three_samples_via_dotlist(tmp_path: Path) -> None:
    """data.2.* auto-extends to 3 entries; data[1] inherits data[0] defaults."""
    overrides = _base_overrides(tmp_path) + [
        f"data.1.inputs.video_path={tmp_path / 'v1.mp4'}",
        f"data.1.output.out_dir={tmp_path / 'out1'}",
        f"data.2.inputs.video_path={tmp_path / 'v2.mp4'}",
        f"data.2.output.out_dir={tmp_path / 'out2'}",
    ]
    cfg, _ = config.loader.load_config_with_overrides(str(PIPELINE_EXAMPLE_CONFIG), overrides)
    assert len(cfg["data"]) == 3
    assert str(tmp_path / "v2.mp4") in cfg["data"][2]["inputs"]["video_path"]


def test_extended_samples_are_independent(tmp_path: Path) -> None:
    """Each extended sample gets its own out_dir, not a shared reference."""
    overrides = _base_overrides(tmp_path) + [
        f"data.1.inputs.video_path={tmp_path / 'v1.mp4'}",
        f"data.1.output.out_dir={tmp_path / 'out1'}",
    ]
    cfg, _ = config.loader.load_config_with_overrides(str(PIPELINE_EXAMPLE_CONFIG), overrides)
    assert cfg["data"][0]["output"]["out_dir"] != cfg["data"][1]["output"]["out_dir"]
