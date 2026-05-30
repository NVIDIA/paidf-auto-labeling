# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from al_utils.media_paths import IMAGE_EXTS, VIDEO_EXTS


def _install_seedvr_import_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    modules = {
        "common": types.ModuleType("common"),
        "common.distributed": types.ModuleType("common.distributed"),
        "common.distributed.advanced": types.ModuleType("common.distributed.advanced"),
        "common.partition": types.ModuleType("common.partition"),
        "common.seed": types.ModuleType("common.seed"),
        "data": types.ModuleType("data"),
        "data.image": types.ModuleType("data.image"),
        "data.image.transforms": types.ModuleType("data.image.transforms"),
        "data.image.transforms.divisible_crop": types.ModuleType("data.image.transforms.divisible_crop"),
        "data.image.transforms.na_resize": types.ModuleType("data.image.transforms.na_resize"),
        "data.video": types.ModuleType("data.video"),
        "data.video.transforms": types.ModuleType("data.video.transforms"),
        "data.video.transforms.rearrange": types.ModuleType("data.video.transforms.rearrange"),
    }

    modules["common.distributed"].get_device = lambda: "cpu"
    for name in (
        "get_data_parallel_rank",
        "get_data_parallel_world_size",
        "get_sequence_parallel_rank",
        "get_sequence_parallel_world_size",
    ):
        setattr(modules["common.distributed.advanced"], name, lambda: 1)
    modules["common.partition"].partition_by_groups = lambda xs, _n: [xs]
    modules["common.partition"].partition_by_size = lambda xs, _n: [xs]
    modules["common.seed"].set_seed = lambda *_args, **_kwargs: None

    class _IdentityTransform:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def __call__(self, value):
            return value

    modules["data.image.transforms.divisible_crop"].DivisibleCrop = _IdentityTransform
    modules["data.image.transforms.na_resize"].NaResize = _IdentityTransform
    modules["data.video.transforms.rearrange"].Rearrange = _IdentityTransform

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture()
def seedvr_window(monkeypatch: pytest.MonkeyPatch):
    _install_seedvr_import_stubs(monkeypatch)
    sys.modules.pop("sr_runner.inference_seedvr2_window", None)
    return importlib.import_module("sr_runner.inference_seedvr2_window")


def test_plain_avcodec_open2_error_is_not_nvenc_error(seedvr_window) -> None:
    assert not seedvr_window._is_nvenc_error(RuntimeError("avcodec_open2 failed for codec mpeg4"))


def test_standalone_media_exts_match_shared_policy(seedvr_window) -> None:
    assert seedvr_window.IMAGE_EXTS == IMAGE_EXTS
    assert seedvr_window.VIDEO_EXTS == VIDEO_EXTS

    for ext in IMAGE_EXTS:
        assert seedvr_window.is_image_file(f"sample{ext}")
    assert not seedvr_window.is_image_file("sample.tiff")


def test_writer_retries_mpeg4_when_nvenc_setup_fails(
    seedvr_window, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    add_stream_codecs: list[str] = []
    closed_containers: list[MagicMock] = []

    class _Stream:
        width = 0
        height = 0
        pix_fmt = ""
        options = {}

        def encode(self, _frame=None):
            return []

    def _open(_path, mode="r"):
        container = MagicMock()
        closed_containers.append(container)

        def _add_stream(codec, rate):
            add_stream_codecs.append(codec)
            if codec == "h264_nvenc":
                raise RuntimeError("h264_nvenc failed during stream setup")
            return _Stream()

        container.add_stream.side_effect = _add_stream
        return container

    monkeypatch.setattr(seedvr_window, "_select_video_encoder", lambda: ("h264_nvenc", {"preset": "p4"}))
    monkeypatch.setattr(seedvr_window.av, "open", _open)
    monkeypatch.setattr(
        seedvr_window.torch, "load", lambda *_args, **_kwargs: torch.zeros((1, 3, 2, 2), dtype=torch.uint8)
    )
    seedvr_window._NVENC_PROBE_RESULT = True
    seedvr_window._MPEG4_FALLBACK_WARNED = False

    seedvr_window._write_video_streaming_from_segments_u8(
        ["segment.pt"],
        str(tmp_path / "out.mp4"),
        fps=5,
        overlap_frames=0,
        blend=False,
    )

    assert add_stream_codecs == ["h264_nvenc", "mpeg4"]
    assert seedvr_window._NVENC_PROBE_RESULT is False
    assert seedvr_window._MPEG4_FALLBACK_WARNED is True
    assert len(closed_containers) == 2
    for container in closed_containers:
        container.close.assert_called_once()
