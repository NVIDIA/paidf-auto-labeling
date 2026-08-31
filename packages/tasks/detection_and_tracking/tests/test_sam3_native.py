# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for native Meta SAM3 runtime selection and output conversion."""

from __future__ import annotations

import builtins
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from core import SceneContext, ScenePaths
from detection_and_tracking.backends import sam3 as sam3_module
from detection_and_tracking.backends.sam3 import SAM3Tracker
from detection_and_tracking.backends.sam3_native import SAM3NativeRuntime
from detection_and_tracking.config import DetectionAndTrackingConfig


class _FakeNativeRuntime:
    def __init__(self, *, config: DetectionAndTrackingConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def run(
        self,
        *,
        media_path: Path,
        scene_paths: ScenePaths,
        scene_ctx: SceneContext,
        prompts: list[str],
    ) -> Path:
        _ = media_path, prompts
        return scene_paths.sidecars_dir / "sam3" / f"{scene_ctx.media_id}_annotated_id.mp4"


def test_sam3_tracker_selects_native_runtime_for_sam31(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "detection_and_tracking.backends.sam3_native.SAM3NativeRuntime",
        _FakeNativeRuntime,
    )
    tracker = SAM3Tracker(
        logging.getLogger("test"),
        DetectionAndTrackingConfig(
            tracker="sam3",
            sam3_prompts=("person",),
            sam3_version="sam3.1",
            sam3_runtime="auto",
        ),
    )
    assert isinstance(tracker._runtime, _FakeNativeRuntime)


def test_sam3_tracker_keeps_transformers_runtime_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sam3_module, "_SAM3Runtime", _FakeNativeRuntime)
    tracker = SAM3Tracker(
        logging.getLogger("test"),
        DetectionAndTrackingConfig(tracker="sam3", sam3_prompts=("person",)),
    )
    assert isinstance(tracker._runtime, _FakeNativeRuntime)


def test_native_detections_from_outputs_happy_path() -> None:
    runtime = SAM3NativeRuntime.__new__(SAM3NativeRuntime)
    runtime.np = np
    runtime.cv2 = _FakeCv2()
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    outputs = {
        "out_obj_ids": [7],
        "out_binary_masks": [mask],
        "out_boxes_xywh": [[1.0, 1.0, 2.0, 2.0]],
        "out_probs": [0.91],
    }
    detections = runtime._detections_from_native_outputs(
        outputs,
        prompt="person",
        chunk_idx=0,
        obj_id_offset=10,
    )
    assert len(detections) == 1
    det = detections[0]
    assert det["prompt"] == "person"
    assert det["instance_id"] == 17
    assert det["object_id"] == "person_0_17"
    assert det["box_xyxy"] == [1.0, 1.0, 3.0, 3.0]
    assert det["score"] == 0.91


def test_native_detections_skip_empty_masks() -> None:
    runtime = SAM3NativeRuntime.__new__(SAM3NativeRuntime)
    runtime.np = np
    runtime.cv2 = _FakeCv2()
    empty = np.zeros((3, 3), dtype=bool)
    outputs = {
        "out_obj_ids": [1],
        "out_binary_masks": [empty],
        "out_boxes_xywh": [[0.0, 0.0, 1.0, 1.0]],
    }
    assert (
        runtime._detections_from_native_outputs(
            outputs,
            prompt="person",
            chunk_idx=0,
            obj_id_offset=0,
        )
        == []
    )


def test_native_missing_package_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "sam3.1_multiplex.pt"
    checkpoint.write_bytes(b"fake")
    monkeypatch.setenv("SAM3_MODEL_PATH", str(checkpoint))

    class _Torch:
        class cuda:  # noqa: N801 - mirrors torch.cuda API surface
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def set_device(gpu_id: int) -> None:
                _ = gpu_id

    runtime = SAM3NativeRuntime.__new__(SAM3NativeRuntime)
    runtime.config = DetectionAndTrackingConfig(
        tracker="sam3",
        sam3_prompts=("person",),
        sam3_version="sam3.1",
        sam3_runtime="native",
        gpu_ids="0",
    )
    runtime.logger = logging.getLogger("test")
    runtime.torch = _Torch()
    runtime._predictor = None

    real_import = builtins.__import__

    def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "sam3" or name.startswith("sam3."):
            raise ImportError("No module named sam3")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(RuntimeError, match="requires the Meta `sam3` package"):
        runtime._ensure_predictor_loaded()


class _FakeCv2:
    RETR_EXTERNAL = 0
    CHAIN_APPROX_SIMPLE = 1

    def findContours(self, *_args: Any, **_kwargs: Any) -> tuple[list[Any], None]:  # noqa: N802
        return [], None
