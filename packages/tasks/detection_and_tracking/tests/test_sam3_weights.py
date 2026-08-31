# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SAM3 weight and runtime resolution."""

from __future__ import annotations

from pathlib import Path

import pytest
from detection_and_tracking.backends.sam3_weights import (
    resolve_sam3_runtime,
    resolve_sam3_weights,
)
from detection_and_tracking.config import DetectionAndTrackingConfig


def test_auto_runtime_selects_transformers_for_sam3() -> None:
    config = DetectionAndTrackingConfig(sam3_version="sam3", sam3_runtime="auto")
    assert resolve_sam3_runtime(config) == "transformers"


def test_auto_runtime_selects_native_for_sam31() -> None:
    config = DetectionAndTrackingConfig(sam3_version="sam3.1", sam3_runtime="auto")
    assert resolve_sam3_runtime(config) == "native"


def test_transformers_runtime_rejects_sam31() -> None:
    config = DetectionAndTrackingConfig(sam3_version="sam3.1", sam3_runtime="transformers")
    with pytest.raises(ValueError, match="requires sam3_runtime='native'"):
        resolve_sam3_runtime(config)


def test_resolve_transformers_dir_from_config_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_root = tmp_path / "sam3"
    model_root.mkdir()
    (model_root / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.delenv("SAM3_MODEL_PATH", raising=False)
    config = DetectionAndTrackingConfig(
        model_cache_path=str(tmp_path),
        sam3_version="sam3",
        sam3_runtime="transformers",
    )
    weights = resolve_sam3_weights(config)
    assert weights.runtime == "transformers"
    assert weights.path_for_runtime == model_root.resolve()


def test_resolve_native_checkpoint_sam31(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_root = tmp_path / "sam3"
    model_root.mkdir()
    checkpoint = model_root / "sam3.1_multiplex.pt"
    checkpoint.write_bytes(b"fake")
    monkeypatch.delenv("SAM3_MODEL_PATH", raising=False)
    config = DetectionAndTrackingConfig(
        model_cache_path=str(tmp_path),
        sam3_version="sam3.1",
        sam3_runtime="native",
    )
    weights = resolve_sam3_weights(config)
    assert weights.runtime == "native"
    assert weights.path_for_runtime == checkpoint.resolve()


def test_resolve_native_checkpoint_from_env_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "custom.pt"
    checkpoint.write_bytes(b"fake")
    monkeypatch.setenv("SAM3_MODEL_PATH", str(checkpoint))
    config = DetectionAndTrackingConfig(sam3_version="sam3.1", sam3_runtime="native")
    weights = resolve_sam3_weights(config)
    assert weights.path_for_runtime == checkpoint.resolve()


def test_missing_native_checkpoint_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_root = tmp_path / "sam3"
    model_root.mkdir()
    (model_root / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.delenv("SAM3_MODEL_PATH", raising=False)
    config = DetectionAndTrackingConfig(
        model_cache_path=str(tmp_path),
        sam3_version="sam3.1",
        sam3_runtime="native",
    )
    with pytest.raises(FileNotFoundError, match="native checkpoint missing"):
        resolve_sam3_weights(config)


def test_missing_transformers_dir_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_root = tmp_path / "sam3"
    model_root.mkdir()
    (model_root / "sam3.pt").write_bytes(b"fake")
    monkeypatch.delenv("SAM3_MODEL_PATH", raising=False)
    config = DetectionAndTrackingConfig(
        model_cache_path=str(tmp_path),
        sam3_version="sam3",
        sam3_runtime="transformers",
    )
    with pytest.raises(FileNotFoundError, match="Transformers weights missing"):
        resolve_sam3_weights(config)
